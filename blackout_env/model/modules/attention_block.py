"""
Grouped Query Attention (GQA) — PyTorch F.scaled_dot_product_attention 기반,
Flash Attention 백엔드로 디스패치. Exclusive Self Attention (XSA, Zhai 2026,
arXiv:2603.09078)도 옵션으로 지원.

요구사항:
- torch >= 2.5  (enable_gqa 네이티브 지원. 이전 버전은 자동 fallback)
- Flash Attention 커널이 실제로 켜지려면 CUDA + fp16/bf16 필요
  (CPU나 fp32에서는 자동으로 다른 SDPA 백엔드로 fallback됨)

XSA
---
표준 SA의 출력 y_i = sum_j a_ij * v_j 는 자기 자신의 value 벡터 v_i와 코사인 유사도가
높아지는 경향이 있다(attention similarity bias) — attention이 문맥 정보뿐 아니라
포인트와이즈 변환(FFN의 역할)까지 일부 떠맡는다는 뜻. XSA는 y_i에서 v_i 방향 성분을
제거해서(y_i가 이미 residual로 v_i에 접근 가능하므로) attention이 순수하게 "문맥"만
담당하도록 강제한다:

    z_i = y_i - (y_i · v̂_i) v̂_i,   v̂_i = v_i / ||v_i||

GQA에서는 Q가 KV보다 head 수가 많아 "자기 자신의 v_i"가 head별로 유일하지 않으므로,
V를 Q의 head 수만큼 repeat_interleave 해서 각 쿼리 head가 자기 그룹의 v_i를 쓰도록 한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .ffn_block import SwiGLUBlock
from .rotary import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        exclusive: bool = True,
    ):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # 하나의 KV head를 공유하는 Q head 수
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.exclusive = exclusive

        # Diagnostic-only: F.scaled_dot_product_attention is a fused kernel that never
        # exposes the pre-softmax QK^T logits, so there's nothing to log by default. Setting
        # log_attention_stats=True makes forward() pay for one extra (unfused, no_grad) QK^T
        # matmul to populate last_logit_norm -- left off the hot path and toggled on only for
        # TB-logging steps by the trainer (see QMIXTrainer._forward_and_loss).
        self.log_attention_stats = False
        self.last_logit_norm: float | None = None

        self.norm = nn.RMSNorm(d_model)

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, T, head_dim) / k, v: (B, num_kv_heads, T, head_dim)

        if rope is not None:
            # rotate q/k only (standard RoPE) — cos/sin: [T, head_dim], broadcasts over (B, heads)
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.log_attention_stats:
            self.last_logit_norm = self._compute_logit_rms(q, k)

        out = self._flash_gqa(q, k, v, attn_mask, is_causal)  # (B, num_heads, T, head_dim)

        if self.exclusive:
            out = self._exclude_self(out, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def _exclude_self(self, out: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        XSA: subtract each position's projection onto its own (unit-normalized) value vector.
        v has num_kv_heads heads (fewer than out's num_heads under GQA) — repeat it to match
        out's query-head count first, so "self value vector" means the same v_i every query
        head in that KV group actually attended with.
        """
        v_full = v.repeat_interleave(self.num_groups, dim=1)  # (B, num_heads, T, head_dim)
        v_hat = F.normalize(v_full, dim=-1)
        return out - (out * v_hat).sum(dim=-1, keepdim=True) * v_hat

    @torch.no_grad()
    def _compute_logit_rms(self, q: torch.Tensor, k: torch.Tensor) -> float:
        """RMS (not raw Frobenius norm -- that would just grow with B*T*heads and tell you
        nothing) of the pre-softmax QK^T/sqrt(head_dim) logits SDPA computes internally but
        never exposes. The standard "is attention saturating/about to blow up" diagnostic:
        RMS growing over training means softmax is sharpening toward near-one-hot (vanishing
        attention gradient), independent of how many logits went into the average."""
        k_full = k.repeat_interleave(self.num_groups, dim=1)  # (B, num_heads, T, head_dim), matches q
        logits = torch.matmul(q, k_full.transpose(-2, -1)) * (self.head_dim ** -0.5)
        return logits.pow(2).mean().sqrt().item()

    def _flash_gqa(self, q, k, v, attn_mask, is_causal):
        try:
            # torch>=2.5: enable_gqa=True 이면 K/V를 미리 repeat 하지 않아도
            # SDPA가 내부적으로 broadcast 하여 Flash Attention 커널로 바로 전달
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
        except (RuntimeError, TypeError):
            # torch<2.5 이거나 Flash 백엔드를 못 쓰는 환경(예: CPU, 지원 안 하는
            # head_dim/dtype)에서는 K/V head를 수동으로 repeat_interleave 해서 사용
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, exclusive: bool = True) -> None:
        super(AttentionBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=num_heads//4, exclusive=exclusive)
        self.ffn = SwiGLUBlock(d_model, d_model*3, d_model)


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = x + self.gqa(x, attn_mask, is_causal, rope)
        x = x + self.ffn(x)

        return x


class AttentionLayers(nn.Module):
    def __init__(self, d_model: int, num_heads: int, depth: int, exclusive: bool = True) -> None:
        super(AttentionLayers, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = depth

        self.layers = nn.ModuleList(
            [AttentionBlock(d_model, num_heads, exclusive=exclusive)
            for _ in range(depth)]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, is_causal, rope)

        return x