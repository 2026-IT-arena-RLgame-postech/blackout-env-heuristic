# 오프라인 프리트레인 런 기록

`blackout_env/train/offline_pretrain.py` 실행 시 매번 헷갈리지 않도록, 런마다 쓴 하이퍼파라미터/코드
상태/결과를 여기 기록한다. 새 런을 돌릴 때는 이 문서 맨 위에 새 섹션을 추가할 것.

## 참고: BBF 공식 하이퍼파라미터 (google-research/bigger_better_faster, `BBF.gin`)

우리 코드가 BBF 논문(Schwarzer et al. 2023)을 어디까지 따르고 있는지 헷갈리지 않도록, 2026-09-14에
공식 저장소에서 직접 확인한 값. 이 프로젝트는 Atari 픽셀이 아니라 24x24 시맨틱 맵 + 5v5 유닛 상태라
전부 그대로 이식하는 게 최선인지는 별개 문제 — 뭐가 의도적으로 다르게 튜닝된 값이고 뭐가 그냥 기본값을
안 맞춘 건지 구분하는 용도.

| 하이퍼파라미터 | 우리 코드 (`QMIXConfig` 기본값) | BBF 공식 값 | 비고 |
|---|---|---|---|
| `spr_k` / `jumps` | 5 | `jumps=5` | ✅ 일치 |
| n_step (start→end) | 10→3 | `max_update_horizon=10`→`update_horizon=3` | ✅ 일치 |
| gamma (start→end) | 0.97→0.997 | `min_gamma=0.97`→`gamma=0.997` | ✅ 일치 |
| `spr_loss_weight` | **1.0** | **`spr_weight=5`** | ❌ 5배 차이 (2026-09-14 발견) |
| `lr` | 3e-4 | `learning_rate=1e-4` | ❌ 3배 차이, 아직 안 바꿈 |
| `weight_decay` | 1e-2 | `weight_decay=0.1` | ❌ 10배 차이, 아직 안 바꿈 |
| reset shrink/perturb (CNN/attention) | keep 80%/92.5% (`reset_alpha_cnn=0.8`/`reset_alpha_attention=0.925`) | keep 50%/50% (`shrink_factor=perturb_factor=0.5`) | ❌ BBF가 훨씬 공격적, 아직 안 바꿈 |
| reset 주기 | `steps // 5` (오프라인 스크립트 기본값) | `reset_every=20_000`, 전체 100k 중 5회 | 사이클 횟수는 비슷 |

lr/weight_decay/reset 강도는 아직 실험 안 해봄 — spr_loss_weight부터 먼저 시도하기로 함 (2026-09-14).

---

## graphic_encoder 그래디언트 소실 진단 + 코드 수정 (2026-09-14, Run 2 완료 직후)

Run 2가 200,000스텝 완료된 직후 GUI로 휴리스틱 대결을 돌려보니 **6전 6무, 전부 0-0** — 유닛끼리
겹치는 락스텝 버그는 사라졌지만, 대신 벽에 계속 박고 거의 움직이지 않는 새로운 증상 발견. 텐서보드
분석으로 원인 진단:

- `grad_norm/graphic_encoder`가 step 0의 0.82에서 지수적으로 감소해 step 50,000 이후로는
  1e-6~1e-7대에 고정 — 다른 모든 컴포넌트(attention/q_head/vector_encoder 등, 훈련 내내
  1e-2~1e-1대 유지)보다 4~5자리 낮음. BBF 리셋(40k/80k/120k/160k)도 잠깐 반등만 시키고
  1000~5000스텝 안에 재붕괴.
- weight_decay 침식(스텝당 `lr×wd`≈3e-6, 200k스텝 전체로도 절반 정도밖에 못 깎음) 속도로는
  리셋 직후 며칠천 스텝 만에 벌어지는 재붕괴 속도를 설명 못 함 — `maybe_reset()`이 리셋 시
  Adam optimizer state도 같이 지우는 걸 확인했으므로([qmix_trainer.py:1113](../blackout_env/train/qmix_trainer.py:1113)),
  stale momentum 때문도 아님.
- 구조 분석: `q_head`는 `unit_out`만 입력받고 `graphic_encoder`의 출력(`vis_out`)은 오직
  `spr_head`→SPR loss로만 연결. `AttentionBlock`이 `x = x + gqa(x)` 형태의 residual이라
  구조적으로 identity 경로는 살아있지만, SPR 경로 자체가 (구) `vision_latent.mean(dim=1)` —
  36개 vision 토큰을 고정 가중치(1/36)로 평균 풀링 — 를 거치면서 그래디언트가 위치별로
  희석되는 것도 한 원인일 것으로 추정.
- 결론: 침식(weight_decay)과 희석(mean-pool) 두 메커니즘이 겹쳐서 작용했을 가능성이 높고,
  둘 다 별개로 고칠 수 있는 코드 변경이라 아래 두 가지를 모두 적용:

**1. graphic_encoder 전용 AdamW 파라미터 그룹** ([qmix_trainer.py](../blackout_env/train/qmix_trainer.py))
- `QMIXConfig.encoder_lr` / `encoder_weight_decay` 필드 추가 (기본값 `None` = 기존과 동일,
  no-op). `self.net.graphic_encoder.parameters()`만 별도 AdamW param group으로 분리.
- `offline_pretrain.py`에 `--encoder-lr`, `--encoder-weight-decay` CLI 플래그 추가.
- `QMIXTrainer.load()`가 옵티마이저 param group 개수 불일치(구버전 체크포인트) 시 net/mixer/spr
  가중치는 정상 로드하고 옵티마이저 state만 새로 초기화하도록 방어 처리 — `ValueError`를 잡아서
  경고만 출력.

**2. SPR 전용 CLS 토큰 추가 (mean-pool 제거)** ([my_model.py](../blackout_env/model/my_model.py))
- ViT/BERT 스타일 학습 가능한 쿼리 토큰(`spr_cls_emb`, 데이터에 의존하지 않는 순수 파라미터) 1개를
  트렁크 시퀀스에 추가 — `[36 vision] + [10 unit] + [1 team_state] + [1 spr_cls]` = 48토큰
  (기존 47). `token_type_emb`도 3→4종류로 확장.
- `spr_head`가 이제 이 CLS 토큰의 attention 출력 하나만 입력받음 (`vision_latent` shape
  `[B, 256]`, 기존 `[B, 36, 256]`을 트레이너에서 평균 내던 것 대체) — attention이 어떤 vision
  토큰이 중요한지 "학습된 가중치"로 직접 골라 담당하게 함.
- `qmix_trainer.py`의 `pooled_vision = vision_latent.mean(dim=1)` / `target_pooled = ...mean(dim=1)`
  두 곳 모두 mean-pool 제거 (이미 CLS 토큰이 pooled 형태로 나옴).
- **주의: 이 변경으로 체크포인트 아키텍처 호환이 또 깨짐** — `token_type_emb.weight`
  shape(3→4)와 `spr_cls_emb.weight`(신규) 때문에 Run 1/Run 2 체크포인트 전부 `net.load_state_dict()`
  에서 즉시 실패 (`RuntimeError: size mismatch`). Run 2의 `final.pt`(200k스텝)는 이제 재사용 불가.

검증: `MyModel` forward/backward 단독 테스트 + `offline_pretrain.py` 20스텝(리셋 4번 포함) 풀
학습 루프 스모크 테스트로 새 아키텍처 정상 동작 확인. 실제 grad_norm/graphic_encoder가
회복되는지는 다음 런을 돌려봐야 알 수 있음 — 아직 검증 전.

---

## Run 4 (진행 중, PID 81173, 시작 2026-09-14 22:2x) — 인코더 축소 + BC loss + 온라인/오프라인 비교

**배경**: Run 3(spr_loss_weight=5.0 + encoder_weight_decay=1e-4만 적용, PID 79933)을 4000스텝까지
돌려본 결과 `grad_norm/graphic_encoder` 붕괴 속도가 Run 2와 **거의 동일**(step 4000 기준 Run2=2.6e-4,
Run3=2.1e-4) — weight_decay 완화도 mean-pool→CLS 토큰 교체도 붕괴 속도에 유의미한 영향을 못 줌 →
Run 3 중단, 아래 2가지를 추가로 적용해서 재시작.

기존 온라인 런(`runs/20260913-014025_53890`, qmix_trainer.py)의 `grad_norm/graphic_encoder`를
대조해보니: 온라인은 step 1000~5000 근처에서 같이 떨어졌다가 **10,000스텝부터 반등해서
0.01~0.03대에서 249,000스텝까지 유지**됨 — 오프라인처럼 1e-7까지 안 죽음. 추정 원인: 온라인은
지금 학습 중인 net 자신의 self-play(+ epsilon 탐험)로 버퍼가 계속 채워지므로, CNN이 벽을 잘못
이해하면 실제로 나쁜 결과(큰 TD-error)가 나고 그게 다시 그래디언트 압력이 되는 피드백 루프가 있음.
오프라인은 버퍼가 유능한 휴리스틱이 만든 고정 데이터라 이 피드백 루프가 없고, 휴리스틱 데이터 안에서는
agent_states만으로도 결과 예측이 잘 되는 지름길이 존재해서 vision을 굳이 정밀하게 안 봐도 loss가
잘 줄어듦(모방학습의 "causal confusion"과 같은 계열 현상으로 추정).

**1. GraphicEncoder 채널 축소** ([graphic_encoder.py](../blackout_env/model/modules/graphic_encoder.py))
- 입력 13채널 중 8개(base 지형/벽 one-hot)는 에피소드 내내 고정, 실제로 매 틱 변하는 건 배터리+아이템
  4채널뿐 — 입력 엔트로피가 낮은데 ImageNet 백본급(64→128→256채널) 용량을 쓰고 있었던 것 아니냐는
  지적으로 축소. `pre_conv` 32→64(기존 64→128), `conv64` 블록 2개→1개, `pool` 64→128(기존 128→256),
  `conv128` 블록 3개→2개, `tokenize_ffn` 입력 128(기존 256). 파라미터 수 1,292,288 → 405,184 (약 3.2배 감소).

**2. BC(behavior cloning) 보조 loss** ([qmix_trainer.py](../blackout_env/train/qmix_trainer.py))
- `QMIXConfig.bc_loss_alpha` 추가 (기본 0.0=off). `cross_entropy(q_values, dataset_action)`을
  own-team 유닛별로 계산해서 total_loss에 더함 — "휴리스틱이 실제로 고른 방향"을 직접 맞히도록
  강제해서, 벽 회피/아이템 탐색처럼 vision에 의존하는 결정을 명시적으로 설명하게 만듦
  (Discrete BCQ 스타일, Fujimoto et al. 2019, `sfujim/BCQ`의 `discrete_BCQ.py`).
- 단순 고정 가중치가 아니라 **TD3+BC 스타일 적응형 정규화**(Fujimoto & Gu 2021): 매 스텝
  `bc_loss`를 `iqn_loss`의 현재 스케일에 맞춰 리스케일한 뒤 `bc_loss_alpha`를 곱함 — Discrete BCQ는
  가중치 1.0을 그냥 고정으로 쓰지만, 그건 Atari Huber TD loss 스케일 기준이고 우리 IQN quantile
  loss는 이 환경에서 훨씬 작아서(~0.01~0.05, cross-entropy 초기값 ln(8)=2.08과 2자리 차이) 그대로
  가져오면 BC가 학습을 통째로 삼킴. `bc_loss_alpha=1.0`이 "스케일 보정 후 BC와 TD를 대등하게 취급"
  (Discrete BCQ의 가중치=1 철학과 동등)에 해당.
- Discrete BCQ의 나머지 두 메커니즘은 의도적으로 안 가져옴: (a) 별도 imitation 헤드 대신 기존
  `q_values`를 그대로 분류기 로짓으로 재사용(구조 단순화 목적), (b) `1e-2 * i.pow(2).mean()` 로짓
  L2 정규화는 안 씀 — BCQ는 imitation 헤드가 전용이라 로짓을 눌러도 무방하지만, 우리는 `q_values`가
  실제 Q값이기도 해서 크기를 억지로 누르면 가치 추정 자체를 해칠 수 있음. (c) argmax 시점 액션
  마스킹(threshold=0.3)도 안 씀 — 우리 목적은 OOD 억제가 아니라 graphic_encoder 그래디언트 확보라
  학습 loss 경로에 두는 게 맞다고 판단.

**3. 10k스텝마다 휴리스틱 대결 eval 훅** ([periodic_eval.py](../blackout_env/train/periodic_eval.py),
[movement_monitor.py](../blackout_env/train/movement_monitor.py))
- `examples/benchmark_heuristics.py`가 쓰던 idle/blocked(벽에 막힘) 판정 로직을 `MovementMonitor`/
  `FailureRuns`로 공용 모듈화, `offline_pretrain.py`가 `--eval-interval`(기본 10,000)마다 헤드리스로
  `RecommendedStrategicHeuristic` 상대 매치를 몇 판 돌려서 win/loss/draw rate, margin, idle/blocked
  incidents를 TensorBoard(`eval/*`)에 기록. 중간에 이상 징후(벽 박기 급증 등) 보이면 200k스텝 다
  기다리지 않고 바로 중단할 수 있게 하는 용도.

```bash
python -m blackout_env.train.offline_pretrain \
    --dataset-dir datasets/heuristic_mixv2_20260913 \
    --steps 200000 \
    --device mps \
    --spr-loss-weight 5.0 \
    --encoder-weight-decay 1e-4 \
    --bc-loss-alpha 1.0 \
    --eval-interval 10000
```

- 체크포인트/텐서보드 디렉토리는 `offline_pretrain.py`가 PID 기준으로 자동 생성 (`checkpoints/offline/`,
  `runs/offline/`).
- **주의: 이번에도 체크포인트 아키텍처 호환 깨짐** — GraphicEncoder 채널 수가 바뀌어서 Run 1/2/3
  체크포인트 전부 로드 불가 (앞선 CLS 토큰 변경으로 이미 깨져있었음, 이번 건 별개로 추가 파손).
- 로그: `datasets/offline_pretrain_logs/run4_full_20260914.log`.

## Run 3 (중단, PID 79933, 4000스텝에서 킬) — spr_loss_weight + encoder_weight_decay만 단독 적용

Run 2에서 관찰된 "`weight_norm/spr_head`가 리셋과 무관하게 계속 우상향(20.6→38+, 안 꺾임)" 및
`grad_norm/graphic_encoder` 붕괴에 대해, `spr_loss_weight=5.0`(BBF 값) + `encoder_weight_decay=1e-4`
두 가지만 적용해서 시도. **결과: grad_norm/graphic_encoder 붕괴 속도가 Run 2와 거의 동일 — 두 변경
다 지배적 원인이 아니었음을 확인** (위 Run 4 섹션 참고) → SPR CLS 토큰 + 인코더 축소 + BC loss를
추가해서 Run 4로 재시작.

## Run 2 (완료, 체크포인트 호환 깨짐 — 재사용 불가) — 유닛 임베딩 + BBF 리셋 픽스

**2026-09-14 추가**: 이후 SPR CLS 토큰 아키텍처 변경(`token_type_emb` 3→4, `spr_cls_emb` 신규)으로
`final.pt` 포함 이 런의 모든 체크포인트가 `net.load_state_dict()`에서 shape mismatch로 로드 불가
— 위 "graphic_encoder 그래디언트 소실 진단 + 코드 수정" 섹션 참고.

```bash
python -m blackout_env.train.offline_pretrain \
    --dataset-dir datasets/heuristic_mixv2_20260913 \
    --steps 200000 \
    --device mps
```

- 체크포인트: `checkpoints/offline/20260914-131107_57374/` (완료 시 `final.pt`)
- 텐서보드: `runs/offline/20260914-131107_57374/`
- 코드 변경 (Run 1 대비):
  - `my_model.py`에 `vec_slot_emb` per-unit 슬롯 정체성 임베딩 추가 — 유닛 10개가 완전히 동일한
    `agent_states`를 가질 때(스폰 직후 등) Q-value가 byte-identical하게 나와서 팀 전체가 락스텝으로
    움직이던 버그 수정.
  - `offline_pretrain.py`가 `maybe_reset()`을 학습 루프에서 실제로 호출하도록 수정 — 이전엔 n_step/gamma
    가 전체 런에 걸쳐 딱 한 번만 어닐링되고 끝값에 고정됐었음. `--reset-interval` 기본값 `steps // 5`
    (=40,000, 5회 리셋).
- 하이퍼파라미터: 전부 `QMIXConfig` 기본값 그대로 — `lr=3e-4`, `batch_size=64`, `weight_decay=1e-2`,
  `spr_loss_weight=1.0`, `reset_alpha_cnn=0.8`, `reset_alpha_attention=0.925`.
- 결과 (step ~190,000/200,000 시점 기준):
  - Run 1에서 봤던 발산 패턴(`td_error`/`q_value` 계속 재상승) **재현 안 됨** — 유닛 임베딩 픽스가
    실제 원인이었던 것으로 보임. 리셋 3회(40k/80k/120k) 모두 설계대로 정상 발동.
  - 새로 관찰된 것: `weight_norm/spr_head`가 리셋과 무관하게(리셋이 spr_head는 안 건드림) 계속
    우상향 (20.6→38+, step 144000 기준). `grad_norm/spr_head`는 낮고 평평(~0.02-0.03), `loss/spr`도
    ~0.01 부근에서 정체 — capacity 부족 시그니처(grad_norm이 높거나 계속 커짐)는 아니고, AdamW +
    weight_decay 불균형에 의한 norm drift에 더 가까워 보임. → Run 3에서 `spr_loss_weight` 조정으로
    먼저 확인.

## Run 1 (완료, 폐기) — 픽스 전, `checkpoints/offline/mixv2_20260913/`

- 유닛 임베딩 없음 (락스텝 버그 있는 상태), `offline_pretrain.py`가 `maybe_reset()`을 호출하지 않아서
  n_step/gamma가 딱 한 번만 어닐링됨.
- 결과: step ~20,000-23,000에서 `td_error`/`loss`가 바닥을 찍고 이후 199,000까지 3-4배로 재상승,
  `weight_norm`/`q_value/std`는 계속 단조 증가 — 전형적 오프라인 Q-value 발산. 이 체크포인트로 GUI
  대결 돌려보다가 "5유닛이 전부 같이 움직임" 버그(락스텝) 발견 → 근본 원인이 Q-발산이 아니라 유닛
  임베딩 누락이었음을 알게 됨. **모든 체크포인트가 `vec_slot_emb` 추가로 아키텍처 호환 깨짐 — 재사용 불가.**
