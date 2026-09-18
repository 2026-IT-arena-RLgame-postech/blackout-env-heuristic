# 모델 용량(capacity) 포화 판단 기준

## 배경

현재 `MyModel`은 학습 대상 파라미터 기준 약 207만 개(`net` 170.4만 + `dist_mixer` 11.6만 +
`spr_predictor` 24.9만)로, LLM 스케일링 법칙(Chinchilla의 params:tokens 비율 등)을 그대로 갖다
쓰기엔 도메인이 너무 다르다 — grid 관측은 텍스트 토큰만큼 정보 밀도가 조밀하지 않고, offline RL은
같은 데이터를 여러 epoch 도는 게 정상이라 "unique 토큰 대비 파라미터" 식의 비유는 결론이 기준 잡기에
따라 정반대로 뒤집힌다. 그래서 graphic encoder / attention trunk / SPR head 등 특정 부분이 실제로
용량이 부족해 병목인지는 비유가 아니라 아래 지표로 직접 판단한다.

이 문서는 [`qplex_migration_criteria.md`](blackout_env/train/qplex_migration_criteria.md)와 같은
원칙을 따른다: **필요조건 → 정황 증거 → 확정적 ablation** 순서로, 모든 단계가 같은 방향을 가리킬 때만
"용량을 늘려야 한다"고 결론 내린다. 필요조건 없이 정황 증거만으로, 혹은 ablation 없이 1~3단계만으로는
모델 크기를 키우지 않는다.

## 현재 아키텍처 규모 (기준점, Run 11 체크포인트로 확인 2026-09-18)

- **`hidden_size=128`**, `N_ATTENTION_HEADS=8` (head_dim=16), `ATTENTION_DEPTH=4`
  — `MyModel.__init__`의 기본값은 256이지만 실제 학습은 `QMIXConfig.hidden_size=128`이 덮어쓴다
  ([`my_model.py`](blackout_env/model/my_model.py), [`qmix_trainer.py`](blackout_env/train/qmix_trainer.py)).
  **이 문서의 이전 판은 코드 기본값 256 / 301만을 적어 두었으나, 학습된 적 없는 설정이다.**
- 학습 대상 파라미터: net **1,703,832** / dist_mixer 116,294 / spr_predictor 249,472
  (합계 **2,069,598**) — `checkpoints/offline/20260917-130437_59475/final.pt`(Run 11)의
  `policy_state`/`dist_mixer_state`/`spr_state` 실측값이고 현재 코드로 만든 모델과 일치한다.
- net 내부 배분: attention 754,688(44.3%) / graphic_encoder 344,448(20.2%) /
  vector_encoder 298,128(17.5%) / q_head 156,936(9.2%) / spr_head 147,584(8.7%) / 임베딩 2,048.
- 연산량(`torch.utils.flop_counter`, batch 1): forward 당 **146 MFLOPs**(n_quantiles=8) /
  218 MFLOPs(32). 이 중 matmul 113.6M, conv 31.9M. CPU 1.97 ms/forward.
  환경 스텝 하나에 두 팀이 같은 net을 한 번씩 쓰므로 292 MFLOPs, 학습 스텝(batch 64, fwd+bwd, net 하나)
  26.8 GFLOPs.

`graphic_encoder`(Run 4에서 1,292,288 → 405,184)와 trunk 폭·깊이는 과거에 **의도적으로 줄인** 것이다
(입력 13채널 중 매 틱 변하는 것은 4채널뿐). 줄인 뒤 성능 저하가 관측되지 않았다는 점을 용량 논의의
출발점으로 둔다.

## 로깅된 진단 신호

`train_step()`의 TB 로깅 블록([`qmix_trainer.py`](blackout_env/train/qmix_trainer.py))에 이미
찍히고 있는 것들:

- `grad_norm/{part}`, `weight_norm/{part}` — part는 `graphic_encoder`, `vector_encoder`,
  `attention_proj`, `attention_ffn`, `token_type_emb`, `spr_head`, `q_head`, `dist_mixer`,
  `spr_predictor` (`_tb_net_parts`, [qmix_trainer.py:365](blackout_env/train/qmix_trainer.py:365)).
- `attention_logit_rms/layer_{i}` — attention trunk 각 레이어의 pre-softmax QK^T 로짓 RMS.
- `loss/{total,iqn,spr}`, `td_error/mean`, `q_value/{mean,std}`.
- `mixer_clamp_pressure/*` — 이건 mixer의 monotonicity 제약 전용 진단이라 QPLEX 판단 기준 문서
  쪽에서 다룬다. 여기서는 참고만.

값 자체보다 **학습 진행에 따른 추세**로 판단한다. 초기화 직후 값은 대부분 의미가 없다.

### 반드시 리셋 사이클에 정렬해서 볼 것 (2026-09-18 추가)

BBF shrink-and-perturb 리셋이 `steps//5`(200k 런에서 40k)마다 걸리고 `weight_norm/*`이 그때마다
급락한다. 그래서 **"마지막 20%"(160k–200k)를 그대로 보면 리셋 직후 회복분을 학습 진전으로 오독한다** —
실제로 Run 11의 raw 마지막 20% `loss/iqn`은 −44.8%로 크게 개선된 것처럼 보이지만, 160k 리셋 스파이크에서
회복한 값일 뿐이다.

주 지표는 **각 리셋 사이클 끝 10k 구간 평균**(30–40k, 70–80k, 110–120k, 150–160k, 190–200k)의 추이로,
보조로 회복이 끝난 꼬리 구간(181k–199k) 기울기를 본다. `schedule/gamma`(0.97→0.997)와 `n_step`(10→3)
어닐이 164k에 끝나고 `per_beta`는 끝까지 오르므로, 마지막 사이클은 런끼리 비교할 때만 쓴다.

grad_norm은 weight norm이 커지면 같이 커지므로 **상대 그래디언트 `grad_norm/{part} ÷ weight_norm/{part}`**로
비교한다. 절대값만 보면 weight norm이 +50% 자란 part가 전부 "그래디언트 증가"로 보인다.

## 판단 기준 (아래를 순서대로 확인)

### 1단계 — grad_norm/weight_norm 추세 (필요조건, 비용 거의 없음)

특정 `part`에 대해 학습 후반에:

- **`grad_norm/{part}`가 줄지 않거나 오히려 커지는데 그 part가 관여하는 loss
  (`loss/iqn`은 attention/graphic_encoder/q_head 전반, `loss/spr`은 spr_head/spr_predictor)가
  plateau** → 계속 밀어붙이는데 더 줄일 표현력이 없다는 신호. 용량 부족 후보로 2단계 진행.
- **`grad_norm/{part}`는 0 근처로 죽었는데 `weight_norm/{part}`는 계속 커짐** → saturating
  nonlinearity 의심. 이건 "더 키워야 함"이 아니라 정규화/활성함수 문제일 수 있으니 용량 부족과
  섞어서 판단하지 않는다.
- **`attention_logit_rms/layer_{i}`가 특정 레이어에서 계속 우상향** → 그 레이어가 점점 더 sharp한
  attention을 원하는데 `head_dim=32`가 부족해 억지로 짜내는 신호일 수 있음. 반대로 학습 내내
  낮게 유지되면 그 레이어는 별로 안 쓰이고 있다는 뜻(용량 이미 남음, 후보에서 제외).

1단계 신호 없이 아래 단계로 넘어가지 않는다.

### 2단계 — loss/지표 디커플링 (정황 증거, 비용 거의 없음)

- `loss/iqn`은 완만히 계속 개선되는데 `loss/spr`만 일찍 plateau → spr_head/spr_predictor 또는
  그 입력을 만드는 graphic_encoder가 다음 latent 예측에 필요한 정보를 못 만든다는 신호.
  `loss/spr`을 trivial baseline(직전 latent를 그대로 다음 latent 예측으로 쓰는 수준)과 비교하면
  더 명확해진다.
- `q_value/std`가 후반에 거의 0으로 수렴하는데 `episode/win_rate*`는 아직 plateau가 아님 → q_head
  또는 그 앞단 encoder가 state를 충분히 구분하지 못하고 있다는 신호.
- 1단계 신호 없이 이 디커플링만 있으면 다른 원인(탐험 부족, reward 스케일, self-play
  비정상성 등)일 가능성이 더 크므로 용량 부족 근거로 쓰지 않는다.

### 3단계 — 미구현이지만 있으면 훨씬 명확해지는 지표 (추가 구현 필요)

- **held-out validation split**: 지금 offline pretrain은 train loss만 본다. 같은 `loss/iqn`·
  `loss/spr`을 학습에 안 쓴 held-out transition에 대해서도 계산해서, train과 held-out이 같이
  높은 곳에서 plateau면 진짜 capacity 부족(bias), train만 계속 내려가고 held-out만 plateau/
  증가하면 그 부분은 이미 충분하거나 과함(variance) — 훨씬 깔끔하게 원인을 가른다.
- **attention head redundancy**: head 간 출력 코사인 유사도/CKA. 여러 head가 거의 동일한 패턴이면
  head 수를 늘려도 의미 없고(이미 redundant), head마다 뚜렷이 다른 패턴인데 다들 sharpness
  한계까지 갔다면 head_dim을 늘리는 쪽이 맞다. 구현하려면 `_forward_and_loss`에서 attention 출력
  텐서를 뽑아 pairwise cosine sim을 추가로 로깅해야 한다.

### 4단계 — 통제된 ablation (확정적 증거, 비용 큼 — 1~2단계 신호가 뚜렷할 때만)

후보로 지목된 부분만 2배로 키운 변형을 같은 데이터/같은 step 수로 별도 학습시켜
`loss/iqn`, `loss/spr`, `td_error/mean` 하한과 (가능하면) win-rate를 비교한다. 예:

- `graphic_encoder` 채널 2배
- `hidden_size` 128 → 192 (attention 전체 폭, `QMIXConfig.hidden_size`)
- `N_ATTENTION_HEADS` 8 → 12(head_dim 16 → 24), 또는 `ATTENTION_DEPTH` 4 → 6
- `spr_predictor` hidden dim 2배

2026-09-18 측정 기준으로 attention 계열은 후보에서 빠졌다(아래 "측정 결과") — 지금 ablation을 한다면
`graphic_encoder`와 `vector_encoder`뿐이고, 그나마 held-out 지표 없이는 Run 13의 신호가 용량인지
데이터 잡음인지 못 가른다.

눈에 띄게 낮은 loss/더 나은 win-rate를 달성하면 그 부분이 실제 병목이었다는 확정적 증거,
거의 차이가 없으면 지금 크기가 이미 충분하다는 뜻이다.

## 측정 결과 (2026-09-18, Run 11 / 12 / 13 TB 로그)

리셋 사이클 정렬 기준으로 1·2단계를 실제로 확인했다. **결론: 지금 용량을 키울 근거 없음.**

| 런 | TB 디렉터리 | 스텝 |
|---|---|---|
| Run 11 | `runs/offline/20260917-130437_59475` | 200k 완주 |
| Run 12 | `runs/offline/20260917-180602_70863` | 85k 중단 |
| Run 13 | `runs/offline/20260917-212224_79536` | 200k 완주 |

### 1단계 — 조건부 성립, 그러나 런 간에 재현되지 않음

사이클 끝(40k → 200k) 기준 `loss/iqn`: **Run 11 .0275 → .0169(−39%, 계속 개선)**,
**Run 13 .0309 → .0306(0%, 완전 plateau)**. `td_error/mean`은 Run 11 .179 → .139, Run 13 .175 → .180.

상대 그래디언트(gn/wn) 40k → 200k 배율:

| part | Run 11 | Run 13 |
|---|---:|---:|
| graphic_encoder | ×0.48 ↓ | **×2.29 ↑** |
| vector_encoder | ×0.87 | **×1.55 ↑** |
| attention_proj | ×0.97 | **×1.85 ↑** |
| attention_ffn | ×1.65 ↑ | **×2.22 ↑** |
| token_type_emb | ×2.53 ↑ | ×4.76 ↑ |
| spr_head | ×0.50 ↓ | ×1.05 |
| q_head | ×0.44 ↓ | ×0.75 ↓ |

"그래디언트는 유지·상승하는데 해당 loss는 plateau"라는 1단계 첫 조건은 **Run 13에서만** 성립한다
(graphic_encoder, attention_proj/ffn, vector_encoder). 그런데 같은 아키텍처·같은 스케줄의 Run 11은
같은 part들의 상대 그래디언트가 오히려 줄면서 loss가 계속 내려간다. 모델이 동일하고 데이터만 다르므로
(Run 13 = 가우시안 BC + 균등 무작위 Q 데이터) **표현력 한계보다 Run 13 데이터가 만든 타깃 잡음**으로
보는 편이 자연스럽다. 결론 규칙의 "모든 단계가 같은 컴포넌트를 가리킬 때"에 해당하지 않는다.

`attention_logit_rms/layer_{0..3}`은 세 런 모두 0.75–1.3 범위에서 후반 평평~하향이다. 우상향 레이어가
없으므로 **attention trunk과 `head_dim`은 용량 후보에서 제외**하고, 0 근처로 죽은 레이어도 없어 놀고 있는
레이어도 없다(전 레이어 균등 사용). net 파라미터의 44%가 attention에 있다는 점과 함께 보면, 이쪽은 이미
여유가 있다.

1단계 두 번째 조건(grad → 0인데 weight_norm 상승)은 **`q_head`**가 해당한다 — 절대 grad가 다른 part의
1/20~1/100(1e-2 수준)이고 Run 11에서 −36% 줄어드는 동안 weight_norm은 +47%. 문서 지시대로 이는 용량이
아니라 정규화/포화 문제로 따로 다룬다(`q_value/std`·`action_range`는 죽지 않아 완전 붕괴는 아님).

### 2단계 — 성립하지 않음

- `loss/iqn`만 개선되고 `loss/spr`만 일찍 plateau하는 디커플링이 없다(Run 11 둘 다 개선, Run 13 둘 다 plateau).
  `probe/spr_latent_cos`도 0.01–0.20으로 latent 붕괴가 없다.
- `q_value/std`는 어느 런에서도 0으로 수렴하지 않는다(1.8–2.1).
- `eval/win_rate`·`onpolicy win_rate`가 세 런 모두 **전 구간 정확히 0**이라 "승률은 아직 plateau가 아닌데
  q가 상태를 못 가른다"는 판정 자체가 불가능하다. eval 점수차는 후반 −67~−81에서 진동.

### 다음에 할 것

4단계 ablation이 아니라 **3단계 held-out validation split 구현**이 먼저다. Run 13의 "그래디언트는 커지는데
train loss가 안 내려간다"가 bias(용량)인지 데이터 잡음인지 현재 로그로는 가를 수 없고, held-out
`loss/iqn`·`loss/spr`이 그것을 바로 가른다. 같은 설정 반복 런의 Elo 분산 측정
(`docs/run11_research_baseline.md` §5-7)과 함께 하면 측정 인프라가 한 번에 정리된다.

## 결론 규칙

모델 크기 확대는 **1단계(grad_norm/weight_norm/attention_logit_rms 추세) + 2단계(loss
디커플링, 가능하면 3단계 held-out 지표까지) + 4단계 ablation**이 모두 같은 컴포넌트를 가리킬 때만
진행한다. 1단계 신호 없이 2~3단계만으로는 키우지 않는다 — 다른 원인일 가능성이 더 크다. 4단계
없이 1~3단계만으로도 키우지 않는다 — 정황 증거일 뿐 확정적이지 않다.

키우기로 결정했다면 병목으로 지목된 부분만 좁게 키운다 — 전체를 다 키우면 어느 변경이 실제
효과였는지 다음번에 또 알 수 없게 된다.
