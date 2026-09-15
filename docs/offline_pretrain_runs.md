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

## Run 5 (중단, PID 23899, 2026-09-15 15:55 ~ 21:30, step 약 184k에서 종료) — 벽충돌 페널티 + 온폴리시 주입 + 리셋 웜업

```bash
python -m blackout_env.train.offline_pretrain \
    --dataset-dir datasets/heuristic_mixv2_20260913 --steps 200000 --device mps \
    --spr-loss-weight 5.0 --encoder-weight-decay 1e-4 --bc-loss-alpha 1.0 \
    --blocked-penalty 0.02 --onpolicy-self-vs-heuristic-frac 0.3 --onpolicy-self-play-frac 0.1 \
    --reset-warmup-steps 2000 --eval-interval 10000
```

- 체크포인트 `checkpoints/offline/20260915-155519_23899/` (마지막 `step_180000.pt`), TB
  `runs/offline/20260915-155519_23899/`.
- 0-0 무승부 버그 수정(`0f48f95`) 후 첫 런이라 **eval 수치를 신뢰할 수 있는 첫 런**이다.

**결과**
| 지표 | 값 |
|---|---|
| `eval/win_rate` | 19개 윈도우(0~180k) 전부 **0.0**, 매 판 패배 |
| `eval/mean_margin` | −56 ~ −92, 추세 없음 |
| `eval/mean_steps` | 395~462결정 (게임 시간 약 16~18초 만에 휴리스틱이 100점 도달) |
| `eval/candidate_blocked_per_1000_ticks` | 6.5 → 1.9~2.8 (Run 4의 2.3~4.0보다 약간 낮음) |
| `grad_norm/graphic_encoder` | 약 0.9, 붕괴 재발 없음 |
| `loss/total` | 약 5만 스텝 이후 0.06~0.07에서 정체 |

**실제 온폴리시 주입량 (윈도우당)**
- 휴리스틱 상대: 약 15,200틱 / 35~37경기 (경기당 약 425틱).
- 셀프플레이: 매번 정확히 **10,501틱 / 1경기**. 시간 만료까지 풀타임이고, 경기 단위로 자르기 때문에 목표(약
  5,200틱)의 2배가 들어갔다.

**이상 증상** (자세한 분석은 [reward_hypotheses.md](reward_hypotheses.md) §5, H8)
- `q_value/mean`이 +0.77 → −3.3으로 거의 직선 하락하고 `q_value/std`는 0.19 → 6.5로 증가했다. Run 4는 40k
  이후 0.3~0.6에서 안정.
- 10k 주입 직후마다 그래디언트가 튄다 (평상시의 약 7배, 130k~150k엔 11~15로 `grad_clip=10` 초과,
  graphic_encoder 19.8배). 같은 순간 Q가 계단식으로 떨어졌다가 회복한다.

**중단 사유와 다음 결정**: eval 전패와 loss 정체가 계속돼 남은 약 1.6만 스텝으로는 결론이 바뀌지 않는다고 판단해
중단했다. Run 6는 셀프플레이 주입을 제거한다 (`--onpolicy-self-play-frac 0`).

**운영 메모**: `nohup ... &`로 띄운 프로세스는 SIGINT를 무시해서 `KeyboardInterrupt` 핸들러(현재 스텝 저장)가
작동하지 않았다. SIGTERM으로 종료했기 때문에 마지막 체크포인트(180k) 이후 약 4천 스텝(TB 기준 184k까지)은 저장되지 않았다.

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

## Run 4 완료 후 진단 (2026-09-15) — GUI 대결, blocked-tick 42% 발견, reward/Q값 조사, Run 5 준비

**Run 4 완료 결과**: 200,000스텝 정상 종료 (`[offline] done.`, 에러 없음). 핵심 성과 —
`grad_norm/graphic_encoder`가 **처음으로 전체 구간 내내 붕괴 없이 유지됨** (0.04~1.3 범위에서
흔들림, step 0부터 200,000까지), `weight_norm/graphic_encoder`도 32.1→54.0으로 매끄럽게 계속 증가.
4가지 조치(SPR CLS 토큰 + encoder 전용 weight_decay + 인코더 축소 + BC loss)가 그래디언트 붕괴
자체는 확실히 해결한 것으로 판단.

다만 `eval/win_rate`는 21개 체크포인트(step 0~200,000) 전부 정확히 0.0 (126게임 무승), GUI로 직접
6판(시드 3개×스왑) 돌려봐도 **0승 2패 4무** — TensorBoard 수치와 일치. `candidate_blocked_per_1000_ticks`는
초기 8.5→10,000스텝 만에 1.8~3.3으로 급감 후 200,000스텝까지 2.3~4.0에서 정체 (더 이상 개선 없음).
사용자가 GUI로 직접 보고 "vs 이전 대비 확실히 개선됐지만 중간중간 멈추는 행동이 있고 전략적으로
비효율적"이라고 보고.

> **정정 (2026-09-15, 커밋 `0f48f95` 이후)**: Run 4의 `eval/mean_margin`(−1.67~0)과 GUI 대결의 "4무"는
> **0-0 무승부 버그의 아티팩트**였다. `BlackOutEnv._collect_obs()`의 리셋 누수 감지가 조기 결정승을 못 잡아서,
> 조기에 끝난 경기의 점수가 다음 에피소드의 0-0으로 덮어써졌다. 근거:
> - Run 4의 `eval/mean_steps`는 408~514결정으로 **게임 시간 약 16~21초**다. 버그 수정 후의 Run 5도 같은 길이
>   (395~462)인데, 점수차는 −56~−92로 전패였다.
> - 즉 Run 4도 휴리스틱에게 약 18초 만에 지고 있었다.
>
> 버그는 조기 승리도 무승부로 바꾸므로 Run 4의 승률 0이 승리를 가렸을 가능성이 이론상 있지만, 같은 경기 길이
> 패턴을 보인 Run 5 보정 데이터가 전패라 가능성은 낮다. 아래의 41.66% blocked 분석은 `agent_states`만 쓰므로
> 이 버그와 무관하다. 학습 리워드도 버그와 무관하다 (유닛 terminal reward 채널을 씀).

**추가 진단 1: idle/blocked 지표의 맹점 발견** — `eval/*_blocked_per_1000_ticks`는 **인시던트
개수**(12틱 이상 연속 블록만 카운트)이지 지속시간이 아님. `examples/evaluate_checkpoint_vs_heuristic.py`로
final.pt vs 휴리스틱 3매치를 직접 틱 단위로 재분석([movement_monitor.py](../blackout_env/train/movement_monitor.py)의
`MovementMonitor`를 그대로 재사용)한 결과:
- idle(정지) 비율 0.00% — `MyPolicy`는 항상 8방향 중 하나를 단위벡터로 명령하는 구조라 "정지" 액션이
  아예 없음([my_policy.py](../blackout_env/model/my_policy.py) `DIRECTION_VECTORS`), 구조적으로 idle
  판정이 절대 안 걸림.
- 방향 반전(thrashing) 비율 0.2~1.2% — 왔다갔다 하는 문제는 아님.
- **blocked(명령했지만 실제로 안 움직임) 비율이 전체 유닛-틱의 41.66%** (6515틱 중 2714틱). 12틱
  이상 지속된 사건은 21건뿐이라 "인시던트 개수" 기준으로는 적어 보였지만, 각각이 평균적으로 매우
  길게(30틱 이상 지속 15건) 이어지면서 전체 시간의 42%를 차지 — 인시던트 카운트가 심각성을 크게
  과소평가하고 있었음.

**추가 진단 2: reward_proposal.md / 실제 C# 코드 대조** — `/Users/mac/project/26rl/reward_proposal.md`
(1253줄, 확률 기반 potential shaping 설계안)와 `blackout/Assets/Project/Runtime/Scripts/ML/`의
`PotentialRewardCalculator.cs`/`IndividualNavPotentialCalculator.cs`/`BlackOutEpisodeCoordinator.cs`를
직접 대조. **결론: `reward_config.json`의 killReward/deathPenalty/teamScoreReward/itemRewards=0은
버그가 아니라 의도된 설계** (문서 §1: "중간 사건에 고정된 임의 보상을 직접 부여하지 않고, 확률
변화를 보상으로 사용"). 실제 승패 신호는 potential shaping(`Ψ_k`, tanh(확정점수차+배터리기댓값)/
potentialScale, η=0.25/γ=0.99995 — 문서 §2 권장값과 정확히 일치)과 terminal ±1
(`BlackOutEpisodeCoordinator.cs`에 하드코딩)로 옴. 단, **벽 충돌 자체를 직접 벌하는 항은 어디에도
없음** — `IndividualNavPotentialCalculator`의 nav-potential은 그리드 경로 거리 기반이라 유닛이
막혀서 진전이 없으면 보상이 늘지도 줄지도 않음(중립). 이게 42% blocked 문제와 직결되는 지점.

**추가 진단 3: Q값 분포 분석** (blocked vs 정상 틱 비교, 3시드 6515틱) —

| | blocked | 정상 |
|---|---|---|
| top1-top2 마진 | 0.7975 | 1.4534 |
| top1 Q값 | **3.0009** | 2.3576 |
| 8방향 표준편차 | 1.0494 | 1.2332 |

blocked일 때 확신도(마진)는 낮아지지만 완전히 tie는 아니고, 오히려 top1 Q값 자체는 **더 큼** —
전형적인 오프라인 Q값 과대추정 신호. 결정적으로 **12틱 이상 지속된 blocked 구간 19건 전부(100%)가
처음부터 끝까지 같은 방향만 반복 선택** — `MyPolicy`가 순수 greedy라 탐색/재시도 메커니즘이 없고,
벽에 막혀 상태가 거의 안 바뀌면 Q값도 안 바뀌어서 같은 실수를 무한 반복. 휴리스틱 데이터에는 애초에
"벽에 막혔다 회복" 상황이 거의 없어(휴리스틱은 이렇게 안 막힘) Q함수가 이 OOD 상태에 대한 학습
신호를 못 받은 것으로 추정 — Run 4 설계 당시의 causal-confusion 가설과 일맥상통.

**Run 5용 코드 변경 3건 (구현 + 스모크 테스트 완료, 2026-09-15):**

1. **벽 충돌 페널티** ([reward_shaping.py](../blackout_env/train/reward_shaping.py) 신규) —
   `blocked_penalty_adjustment()`: 연속 `agent_states`로 movement≤2e-4 판정된 유닛마다
   `-penalty_per_unit`을 그 틱의 (팀 합산) reward에서 차감. action_norm 체크는 생략 —
   `MyPolicy`/휴리스틱 모두 항상 8방향 중 하나(norm=1)를 고르므로 항상 참이라 무의미.
   `offline_dataset.load_dataset_into()`에 `team_indices`/`penalty_per_unit` 파라미터 추가해서
   기존 정적 데이터셋에도 로드 시점에 소급 적용 (휴리스틱은 거의 안 막히므로 영향 미미), 신규
   수집 데이터에도 동일 적용. `--blocked-penalty` CLI 플래그 (기본 0.0=off).
2. **온라인 데이터 혼합** ([onpolicy_collect.py](../blackout_env/train/onpolicy_collect.py) 신규) —
   기존 eval 훅(10k스텝마다)을 재활용해서 self(후보)-vs-휴리스틱 매치는 그대로 두고, 틱 데이터를
   `buffer_a`/`buffer_b`에 push. 추가로 self-play(후보 vs 자기자신) 매치도 실행해서 push.
   `SequentialReplayBuffer`가 이미 진짜 FIFO ring buffer라 새 로직 없이 그냥 push만 하면 오래된
   순수 휴리스틱 데이터부터 자연스럽게 밀려남. 목표: 전체 런에 걸쳐 buffer_capacity의 30%를
   self-vs-heuristic, 10%를 self-play로 주입(윈도우당 균등 분배) → 최종적으로 대략 60/30/10 구성에
   수렴. 매치는 고정 판수가 아니라 **목표 틱 수 도달할 때까지 반복**(매치 끝나고 나서 체크, 중간에
   안 끊음) — safety cap(`--onpolicy-max-matches`, 기본 50)으로 무한루프만 방지.
   self-play 매치는 헤드-투-헤드라 초반엔 시간제한까지 채우는 경우가 많아(스모크 테스트에서 목표
   500틱인데 실제 1판이 10,501틱) self-play 비중이 의도한 10%보다 커질 수 있음 — 사용자 확인:
   중간에 끊지 않고 그대로 두는 것으로 결정(self-play는 eval 지표에 안 쓰이므로 오염 없음).
   `--onpolicy-self-vs-heuristic-frac`/`--onpolicy-self-play-frac` CLI 플래그 (기본 0.0=off).
3. **reset 직후 encoder LR 웜업** ([qmix_trainer.py](../blackout_env/train/qmix_trainer.py)) —
   `QMIXConfig.reset_warmup_steps` 추가 (기본 0=off). `_reset_submodule()`이 리셋 시 해당 서브모듈의
   Adam momentum/variance state도 지우므로, 리셋 직후 몇 스텝은 2차 모멘트 추정치가 없는 상태로
   업데이트가 들어감 — Adam warmup이 원래 완화하려는 바로 그 상황. graphic_encoder 전용 param
   group의 lr만 `_anneal_cycle_start_step`(리셋마다 갱신되는 사이클 시작점) 기준으로 0→목표값
   선형 램프. **주의: 이건 논문 근거(BBF/IQL 등)로 도입한 게 아니라 우리 자체 진단(reset 직후
   graphic_encoder 그래디언트 불안정)에 국한된 대증 조치** — 아래 "학습률 스케줄 조사" 참고.
   attention도 리셋되지만 grad_norm이 계속 건강했으므로 warmup 대상에서 제외.
   `--reset-warmup-steps` CLI 플래그.

**학습률 스케줄 조사 (전체 런 스케줄은 도입 안 하기로 결정)**: BBF 공식 gin config
(`BBF.gin`)를 직접 확인한 결과 `learning_rate=0.0001` **고정, 스케줄 전혀 없음** — reset_every=20k/
cycle_steps=10k로 리셋해도 lr은 안 바뀜. 즉 이 코드베이스가 그대로 복제한 reset 메커니즘의 원전
자체가 "리셋해도 스케줄 불필요"를 실측으로 보여줌. IQL 논문(Kostrikov et al. 2021, Appendix B)도
직접 확인 — "We use cosine schedule for the actor learning rate"라는 한 줄이 전부(근거 설명 없음),
게다가 **actor(정책망) 전용**이고 critic/value는 고정 3e-4. 우리 구조(QMIX+IQN)는 별도 정책망 없이
Q값 argmax가 곧 정책이라 IQL의 "정책망 스케줄" 논리가 적용될 대상 자체가 없음 → 전체 런
cosine/linear decay는 근거 부족으로 기각, 위 3번(reset 직후 encoder만 국소 웜업)으로 대체.

**검증**: `--steps 40 --eval-interval 20` 스모크 테스트 2회(Unity 실제 빌드, 헤드리스) — 1회는
블록 페널티+온폴리시 수집만, 2회는 `--reset-warmup-steps 5`까지 포함 — 둘 다 에러 없이 끝까지
완료 확인. `_apply_encoder_lr_warmup()`은 별도 유닛테스트로 리셋마다 0→목표lr 선형 램프, 다른
param group은 안 건드리는 것도 확인.

**다음 실행 예정 (Run 5, 아직 시작 안 함)**:
```bash
python -m blackout_env.train.offline_pretrain \
    --dataset-dir datasets/heuristic_mixv2_20260913 \
    --steps 200000 \
    --device mps \
    --spr-loss-weight 5.0 \
    --encoder-weight-decay 1e-4 \
    --bc-loss-alpha 1.0 \
    --blocked-penalty 0.02 \
    --onpolicy-self-vs-heuristic-frac 0.3 \
    --onpolicy-self-play-frac 0.1 \
    --reset-warmup-steps 2000 \
    --eval-interval 10000
```

---

## Run 4 (완료, PID 81173, 시작 2026-09-14 22:2x, 종료 2026-09-15) — 인코더 축소 + BC loss + 온라인/오프라인 비교

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
