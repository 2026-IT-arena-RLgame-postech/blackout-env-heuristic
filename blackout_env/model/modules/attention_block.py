"""
Grouped Query Attention (GQA) — PyTorch F.scaled_dot_product_attention 기반,
Flash Attention 백엔드로 디스패치.

요구사항:
- torch >= 2.5  (enable_gqa 네이티브 지원. 이전 버전은 자동 fallback)
- Flash Attention 커널이 실제로 켜지려면 CUDA + fp16/bf16 필요
  (CPU나 fp32에서는 자동으로 다른 SDPA 백엔드로 fallback됨)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .ffn_block import SwiGLUBlock


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # 하나의 KV head를 공유하는 Q head 수
        self.head_dim = d_model // num_heads
        self.dropout = dropout

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
    ) -> torch.Tensor:
        B, T, _ = x.shape

        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, T, head_dim) / k, v: (B, num_kv_heads, T, head_dim)

        out = self._flash_gqa(q, k, v, attn_mask, is_causal)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)

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
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(AttentionBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=num_heads//4)
        self.ffn = SwiGLUBlock(d_model, d_model*3, d_model)


    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        x = x + self.gqa(x, attn_mask, is_causal)
        x = x + self.ffn(x)

        return x


class AttentionLayers(nn.Module):
    def __init__(self, d_model: int, num_heads: int, depth: int) -> None:
        super(AttentionLayers, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = depth

        self.layers = nn.ModuleList(
            [AttentionBlock(d_model, num_heads)
            for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, is_causal)

        return x