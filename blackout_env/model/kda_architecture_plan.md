# KDA(Kimi Delta Attention) 도입 계획 — [공간 Encoder → 시간 KDA → 공간 Decoder]

> 이번 세션에서 KDA 구현은 다음 작업으로 미뤘다 (`unit 임베딩` 수정과 offline pretrain의 `BBF 반복
> 리셋+재-annealing` 수정을 먼저 반영). 이 문서는 사용자가 정리해 준 아키텍처 전체 설계를 그대로
> 보존한 것 — 다음에 KDA를 실제로 붙일 때 여기서부터 시작한다.

## 1. 아키텍처 핵심 컨셉

공간 처리(Full Attention)와 시간 처리(KDA)를 층(Layer) 단위로 완전 분리(decouple)하여, 강화학습의
**공간 문맥 파악 + 장기 기억(credit assignment) + 초고속 학습 병렬성**을 모두 챙기는 하이브리드 설계.

> **2026-09-14 갱신**: 사용자가 인코더/디코더 구조를 더 단순하게 확정했다 — 아래 §1/§2가 최신
> 버전이고, 이전에 적었던 "패치만 별도로 self-attention → pooling" 인코더 설계는 폐기.

- **공간 Encoder (Full Attn):** **지금 `MyModel`의 trunk를 그대로 유지** (vision 36 + unit 10 +
  team_state 1 = 47 토큰이 한 번에 full self-attention, `token_type_emb`/`vec_slot_emb`/RoPE 전부
  그대로). 여기에 **학습 가능한 CLS(memory) 토큰 5개만 추가로 concat**해서 같은 full attention에
  같이 태운다 (입력 52 토큰). 별도의 "패치만 attention → pooling" 단계를 두지 않는다 — 지금 구조에
  토큰 5개 얹는 것 말고는 인코더 쪽을 바꾸지 않는다는 뜻.
- **시간 전파 (KDA):** 인코더 출력 중 CLS 슬롯 5개(`h_t`)만 뽑아서 시퀀스로 모으고, per-channel
  gated delta rule로 시간적 맥락(history)을 누적/정제해 시간 메모리 토큰(`M_{1:T}`) 생성.
- **공간 Decoder (Full Attn):** 인코더와 같은 구성(현재 프레임의 vision+unit+team_state 47 토큰)에
  이번엔 raw CLS 대신 **KDA로 정제된 M_t 5개**를 concat해서 또 한 번 full attention. **이 디코더
  full attention이 끝나면 M_t 슬롯 5개의 출력에서 바로 Q-value를 뽑는다** — 지금처럼 `unit_out`(유닛
  토큰 10개)에서 뽑는 게 아니라, decoder를 거친 5개 메모리 슬롯이 Q-head의 입력이 됨.

  **(2026-09-14 확정)** 5는 "우리 팀 유닛 수(5)에 맞춘 것"일 뿐 — 슬롯 i = 유닛 i 전용 영구 메모리로
  쓴다. KDA를 두는 이유 자체가 **유닛별 장기 전략(그동안의 위치/교전/역할 이력)을 저장**하는 것이라,
  범용 장면 요약이 아니라 유닛별 전용 슬롯이 맞는 방향 — `vec_slot_emb`가 "동일 상태 유닛 구분
  불가" 문제를 정적 임베딩으로 임시 봉합했던 것을, KDA 슬롯이 시간축까지 포함해서 근본적으로
  대체하게 된다.

## 2. 전체 연산 파이프라인 (학습 / prefill 기준)

학습 시에는 replay buffer에 전체 시퀀스 `X_{1:T}`가 존재하므로 **루프(`for`) 없는 100% 텐서 병렬
연산**으로 동작한다.

```
[입력 데이터] X_{1:T}  (Shape: [B, T, 47, D])  <-- 지금 MyModel trunk 그대로: vision 36 + unit 10 + team_state 1
       │
       ▼
[1단계: Spatial Encoder (Full Attention) = 지금 MyModel.forward의 self.attention 그대로]
  - T개 프레임을 Batch 차원으로 통합: [B*T, 47, D]
  - 학습 가능한 CLS(memory) 토큰 5개를 concat: [B*T, 52, D] 로 같은 full self-attention에 통과
    (token_type_emb/vec_slot_emb/RoPE 등 기존 로직은 전부 그대로, CLS 토큰 5개만 추가)
  - 결과: attention 출력 중 CLS 슬롯 5개만 추출 = 프레임당 5개의 글로벌 요약 토큰
    h_{1:T} (Shape: [B, T, 5, D])
       │
       ▼
[2단계: Temporal Processing (Chunk-Parallel KDA)]  <-- 병렬화의 핵심
  - 요약 토큰 h_{1:T} 입력 (Shape: [B, T, 5, D], 5개 슬롯을 독립 시퀀스로 보고 KDA 적용)
  - 시퀀스 T를 Chunk(예: 64)로 분할하여 DPLR(하삼각 행렬곱) 알고리즘 적용
  - 과거 시간적 기억이 정제된 시간 메모리 토큰 M_{1:T} 출력 (Shape: [B, T, 5, D])
       │
       ▼
[3단계: Spatial Decoder (Full Attention) = 같은 trunk를 한 번 더, CLS 자리에 M_t를 넣어서]
  - 같은 프레임의 vision+unit+team_state 47 토큰 + 이번엔 raw CLS 대신 M_t 5개를 concat: [B*T, 52, D]
  - 또 한 번 full self-attention (encoder와 같은 종류의 attention, 가중치 공유 여부는 미정)
  - 결과: attention 출력 중 M_t 슬롯 5개 위치의 출력 = Q-head 입력 (unit_out 10개 대신)
       │
       ▼
[4단계: Action & Value Head]
  - 디코더의 M_t 슬롯 5개 출력에서 바로 Q-value(또는 Action Distribution & Value) 추출
    -- "5"가 우리 팀 유닛 5명 각각의 전용 메모리인지, 범용 요약 5개인지는 미해결 (위 §1 참고)
```

## 3. 학습(Training) vs 추론(Rollout) 동작 방식 비교

| 구분 | 오프라인 학습 / epoch update | 실시간 환경 인터랙션 (rollout / test) |
| --- | --- | --- |
| 입력 조건 | 전체 시퀀스 `X_{1:T}`가 replay buffer에 존재 | 매 스텝마다 `t=1` 프레임만 도착 |
| KDA 연산 모드 | **Chunk Parallel (DPLR) 병렬 연산** | **Single-step recurrent update** |
| 시간 복잡도 | O(1) GPU kernel call (병렬 matmul) | O(1) 스텝당 계산량 (상수 시간) |
| 메모리(KV cache) | 필요 없음 (chunk 내 행렬곱으로 처리) | KDA state matrix `S_t` (`H x d_k x d_k`) 고정 크기 유지 |
| 특징 | `for` 루프 없는 극도로 빠른 학습 속도 | 시퀀스가 아무리 길어져도 추론 속도/메모리 유지 |

## 4. 병렬화가 가능한 이유

1. **단방향 feed-forward 연산 흐름**: Encoder → KDA → Decoder로 정보가 한 방향으로만 흐르고, `t`
   스텝 decoder의 action 출력이 `t+1` 스텝 encoder의 입력으로 들어가는 recurrent feedback loop를
   제거했다.
2. **KDA 입력의 사전 확정**: 프레임 `X_{1:T}` 전체가 사전에 주어진 상태에서 encoder가 `h_{1:T}`
   전체를 한 번에 만들어 주므로, KDA가 T 전체 시퀀스를 한 번에 받아 chunk parallelization
   (Triton/CUDA GEMM)을 100% 활용할 수 있다.
3. **Bottleneck compression을 통한 연산 효율화**: 공간 패치 N개를 KDA에 전부 넣지 않고 요약 토큰
   `h_t`로 압축해서 넣기 때문에 KDA의 시간축 연산량과 메모리 사용량이 극도로 가볍다.

## 5. 아키텍처 요약 카드

- **Encoder:** 지금 `MyModel` trunk의 full self-attention 그대로(47 토큰) + 학습 가능한 CLS
  토큰 5개만 추가(52 토큰) — 별도 pooling 연산 없이 CLS 슬롯 출력을 그대로 요약 토큰으로 사용
- **Temporal:** Kimi Delta Attention (KDA) (per-channel diagonal decay gating), CLS 5슬롯 각각에
  독립 적용
- **Decoder:** 같은 trunk의 full self-attention을 한 번 더, 이번엔 CLS 자리에 M_t 5개를 넣어서 —
  이 두 번째 attention 출력의 M_t 슬롯 5개에서 Q-value를 직접 추출 (unit_out 10개 대신)
- **병렬성:** 공간축 full matmul + 시간축 chunk-DPLR KDA = 100% 루프-프리 학습

## 6. 이 프로젝트(BlackOut)에 붙일 때 미리 알아야 할 것

지난 논의에서 확인된, 이 설계를 실제로 붙이기 전에 반드시 짚어야 할 제약들:

- **Triton 없음**: 이 Mac은 `--device mps`(Apple Silicon)이고 `triton` 패키지 자체가 설치돼 있지
  않다(`pip list`/`import triton` 둘 다 확인 완료). `fla-org/flash-linear-attention` 등 대부분의
  KDA/DeltaNet 구현체는 Triton chunk kernel 기반이라 여기선 그대로 못 쓴다. **순수 PyTorch로
  chunk-parallel delta rule을 직접 구현**해야 한다 (위 표의 "Chunk Parallel (DPLR)"을 Triton 없이
  matmul로). Rollout 쪽 single-step recurrent update는 elementwise+matmul이라 MPS에서 그대로 돌아간다.
- **현재 MyModel은 완전 stateless per-step**: `qmix_trainer.py`/`MyPolicy.act()`는 스텝마다 독립
  feedforward라 recurrent state를 들고 다니는 경로가 없다. KDA 도입 시:
  - **rollout**: `MyPolicy`와 `qmix_trainer.py`의 rollout 루프가 에피소드(=흡수구간, 이 프로젝트의
    "episode" 정의 — [[project_heuristic_variation_cadence]] 참고) 동안 KDA state matrix `S_t`를
    들고 다니다 경계에서 리셋해야 함.
  - **학습**: 지금 `train_step()`은 `SequentialReplayBuffer`에서 개별 transition을 랜덤 샘플링한다.
    KDA를 학습에 반영하려면 순서가 있는 **윈도우 단위 BPTT**(R2D2 스타일 burn-in)가 필요 — SPR의
    k-step future window 샘플링(`window = max(n_step, spr_k)`, `qmix_trainer.py:769`)이 이미
    윈도우 샘플링을 지원하니 이걸 재활용할 수 있다.
- **체크포인트 호환성**: 이번 세션에 이미 한 번(`vec_slot_emb` 유닛 식별 임베딩 추가) 아키텍처를
  바꿔서 기존 체크포인트가 깨졌다. KDA를 붙이면 또 한 번 깨지므로, 유닛 임베딩 fix로 새로 시작한
  오프라인 프리트레인 결과를 어느 정도 검증한 뒤에 KDA 작업을 얹는 순서를 권장.
- **유닛 임베딩과의 관계**: 이 설계의 "공간 요약 토큰 h_t로 압축"이 그대로 가면, unit 토큰별
  identity가 다시 요약 단계에서 뭉개지지 않도록 주의 — `vec_slot_emb`가 해결한 "동일 상태 유닛
  구분 불가" 문제가 압축(K개 요약 토큰) 단계에서 재발하지 않는지 설계 시 확인 필요.

## 7. 다음에 KDA 작업을 시작할 때 첫 스텝

1. 순수 PyTorch delta-rule(Kimi의 fine-grained per-channel gate 버전) 레이어만 모델에 붙이기 전에
   단독으로 작게 구현 → 수치적으로 검증(그래디언트 체크, 간단한 synthetic recall task로 실제 시간적
   기억이 동작하는지 확인).
2. 검증되면 **학습 가능한 CLS(memory) 토큰 5개**를 encoder 입력(패치 토큰과 concat)에 추가 — 이
   5개가 encoder의 요약 토큰(h_t)이자 KDA 입출력이자 decoder의 메모리 앵커(M_t)를 겸함(§1, §2).
3. rollout 쪽 상태 유지(5개 슬롯 각각의 KDA state matrix) + train_step() 윈도우 샘플링 재사용까지
   붙여서 end-to-end로 완성.

CLS 슬롯 수를 5로 굳힌 이유(§1)는 서로 다른 종류의 공간 정보(아군 위치, 적 위협, 자원 상태 등)를
슬롯별로 분담해서 요약하게 하려는 것 — 1개짜리 단일 풀링 벡터보다 정보 손실이 적을 것으로 기대.
`vec_slot_emb`가 고친 "동일 상태 유닛 구분 불가" 문제(§6)가 이 CLS 압축 단계에서 재발하지 않는지는
실제 구현 시 반드시 확인.
