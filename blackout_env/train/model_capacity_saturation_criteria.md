# 모델 용량(capacity) 포화 판단 기준

## 배경

현재 `MyModel`은 학습 대상 파라미터 기준 약 301만 개(`net` 264.8만 + `dist_mixer` 11.6만 +
`spr_predictor` 24.9만)로, LLM 스케일링 법칙(Chinchilla의 params:tokens 비율 등)을 그대로 갖다
쓰기엔 도메인이 너무 다르다 — grid 관측은 텍스트 토큰만큼 정보 밀도가 조밀하지 않고, offline RL은
같은 데이터를 여러 epoch 도는 게 정상이라 "unique 토큰 대비 파라미터" 식의 비유는 결론이 기준 잡기에
따라 정반대로 뒤집힌다. 그래서 graphic encoder / attention trunk / SPR head 등 특정 부분이 실제로
용량이 부족해 병목인지는 비유가 아니라 아래 지표로 직접 판단한다.

이 문서는 [`qplex_migration_criteria.md`](blackout_env/train/qplex_migration_criteria.md)와 같은
원칙을 따른다: **필요조건 → 정황 증거 → 확정적 ablation** 순서로, 모든 단계가 같은 방향을 가리킬 때만
"용량을 늘려야 한다"고 결론 내린다. 필요조건 없이 정황 증거만으로, 혹은 ablation 없이 1~3단계만으로는
모델 크기를 키우지 않는다.

## 현재 아키텍처 규모 (기준점)

- `hidden_size=256`, `N_ATTENTION_HEADS=8` (head_dim=32), `ATTENTION_DEPTH=4`
  ([`my_model.py`](blackout_env/model/my_model.py))
- 학습 대상 파라미터: net 2,648,344 / dist_mixer 116,294 / spr_predictor 249,472 (합계 3,014,110)

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

값 자체보다 **학습 진행(특히 후반 20% 구간)에 따른 추세**로 판단한다. 초기화 직후 값은 대부분
의미가 없다.

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
- `hidden_size` 256 → 384 (attention 전체 폭)
- `N_ATTENTION_HEADS` 8 → 12, 또는 `ATTENTION_DEPTH` 4 → 6
- `spr_predictor` hidden dim 2배

눈에 띄게 낮은 loss/더 나은 win-rate를 달성하면 그 부분이 실제 병목이었다는 확정적 증거,
거의 차이가 없으면 지금 크기가 이미 충분하다는 뜻이다.

## 결론 규칙

모델 크기 확대는 **1단계(grad_norm/weight_norm/attention_logit_rms 추세) + 2단계(loss
디커플링, 가능하면 3단계 held-out 지표까지) + 4단계 ablation**이 모두 같은 컴포넌트를 가리킬 때만
진행한다. 1단계 신호 없이 2~3단계만으로는 키우지 않는다 — 다른 원인일 가능성이 더 크다. 4단계
없이 1~3단계만으로도 키우지 않는다 — 정황 증거일 뿐 확정적이지 않다.

키우기로 결정했다면 병목으로 지목된 부분만 좁게 키운다 — 전체를 다 키우면 어느 변경이 실제
효과였는지 다음번에 또 알 수 없게 된다.
