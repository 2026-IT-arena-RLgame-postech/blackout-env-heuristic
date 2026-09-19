# QMIX → QPLEX 전환 판단 기준

## 배경

현재 `QMIXTrainer`는 [`QMixer`](../../blackout_env/model/modules/qmix_mixer.py)의 monotonicity 제약
(`dQ_tot/dQ_i >= 0`, 하이퍼넷 출력에 `.abs()`를 걸어 강제)으로 IGM을 만족시킨다. 이 제약은 QMIX가
표현 가능한 팀 가치함수 집합을 "진짜 IGM을 만족하는 함수 전체"의 부분집합으로 좁힌다 — 한 유닛이
손해를 감수해야 팀 전체가 이득인 비단조적 협응(예: 미끼 역할)을 값으로 표현하지 못할 수 있다.

QPLEX(duplex dueling)는 이 제약 없이 IGM을 만족시켜 표현력이 더 크지만, 지금 쓰고 있는 DFAC 스타일
distributional 확장([`DistributionalQMixer`](../../blackout_env/model/modules/qmix_mixer.py))은 QMIX의
monotonic mixer를 전제로 설계된 트릭이라 QPLEX로 바꾸면 이 부분을 통째로 다시 설계해야 한다. SPR/PER/
self-play까지 이미 복잡하게 얽힌 상태에서 근거 없이 감행할 변경은 아니다 — 아래 기준으로 실제
병목인지 먼저 확인한다.

## 로깅된 진단 신호

`train_step()`의 TB 로깅 블록([`qmix_trainer.py`](../../blackout_env/train/qmix_trainer.py))에
[`clamp_pressure_stats`](../../blackout_env/model/modules/qmix_mixer.py)로 아래 6개 스칼라가 매
`tb_log_interval` 스텝마다 기록된다.

- `mixer_clamp_pressure/frac_negative/{hyper_w1,hyper_w2,shape_weight}` — 각 하이퍼넷의 raw
  출력(=.abs() 적용 전) 중 몇 %가 음수인지. 즉 "몇 %가 지금 clamp에 걸려 있는지".
- `mixer_clamp_pressure/neg_magnitude/{hyper_w1,hyper_w2,shape_weight}` — 음수인 항들의 평균
  절댓값. "얼마나 세게 음수 쪽으로 밀리고 있는지".

값이 크다/작다만으로는 판단하지 않는다 — **학습 진행에 따른 추세**가 중요하다. 초기화 직후에는
`frac_negative`가 0.5 근처인 게 자연스럽다(랜덤 초기화라 절반은 음수). 판단은 이 값이 학습이
수렴 근처로 갈 때 어떻게 움직이는지로 한다.

## 전환 기준 (아래를 순서대로 확인)

### 1단계 — clamp pressure 추세 (필요조건, 비용 거의 없음)

학습 후반(예: 마지막 20% 구간)에 `frac_negative` 또는 `neg_magnitude`가:

- **감소하거나 낮은 값에서 안정** → monotonicity가 거의 공짜다. QPLEX로 넘어갈 이유 없음. 여기서 멈춘다.
- **줄어들지 않고 유지되거나 오히려 증가** → 네트워크가 계속 음의 mixing weight를 원하는데
  막히고 있다는 신호. 2단계로 진행.

셋 중 하나(`hyper_w1`, `hyper_w2`, `shape_weight`) 중 하나라도 뚜렷하게 이 패턴을 보이면 2단계로
넘어갈 후보로 본다. 세 신호가 항상 같이 움직일 필요는 없다 — `shape_weight`는 distributional shape
믹싱 전용이라 스칼라 mixer(`hyper_w1`/`hyper_w2`)와 별개로 병목일 수 있다.

### 2단계 — TD loss/win-rate 디커플링 (정황 증거, 비용 거의 없음)

기존에 로깅 중인 `loss/iqn`, `td_error/mean`, `episode/win_rate*`를 같이 본다.

- TD loss(`loss/iqn`, `td_error/mean`)는 계속 완만히 개선되는데 `episode/win_rate_selfplay/online`이
  일찍 plateau에 걸려 더 안 오른다 → capacity가 아니라 표현력 한계일 가능성.
- `mixer_embed_dim`/`hyper_hidden`을 키워도(`QMIXConfig`) TD loss/win-rate가 거의 안 움직인다 →
  일반적인 capacity 부족이 아니라 monotonic 함수 클래스 자체의 상한에 막혔을 가능성이 커진다.

1단계 신호 없이 이 디커플링만 있는 경우는 다른 원인(탐험 부족, 보상 스케일, self-play 비정상성 등)일
수 있으므로 QPLEX 전환 근거로 쓰지 않는다 — 반드시 1단계와 함께 봐야 한다.

### 3단계 — heuristic 상대 국소 win-rate (도메인 특화 정황 증거)

`heuristics/`에 비단조적 협응(미끼/유인 등)을 쓰는 전략이 있다면(`v4_family.py`, `intercept.py` 등
후보) 그 heuristic만 상대로 한 win-rate를 별도로 추적한다. 전체 평균 win-rate는 오르는데 이 특정
상대에게만 계속 지거나 정체되어 있다면, "특정 유형의 협응을 학습하지 못하고 있다"는 국소적이지만
구체적인 증거가 된다.

### 4단계 — 통제된 ablation (확정적 증거, 비용 큼 — 1~3단계 신호가 뚜렷할 때만)

`QMixer.forward`에서 `.abs()`를 뺀(=IGM 깨짐, 진단 전용, 실제 배포 금지) 버전을 별도 프로세스로
같은 replay 데이터 분포에 대해 학습시켜 TD loss 하한과 (가능하면) win-rate를 비교한다.

- 눈에 띄게 낮은 TD loss 또는 높은 win-rate를 달성한다 → monotonicity가 실제 병목이었다는 확정적
  증거. QPLEX(또는 우선 QTRAN — 구조 변경이 더 작음) 전환을 진행한다.
- 거의 차이가 없다 → monotonicity는 병목이 아니다. 다른 곳(보상 스케일, 탐험, 모델 capacity 등)을
  본다.

## 결론 규칙

QPLEX 전환은 **1단계(clamp pressure 추세) + (2단계 또는 3단계 중 하나) + 4단계 ablation**이 모두
같은 방향을 가리킬 때만 진행한다. 1단계 신호 없이 2~3단계만으로는 전환하지 않는다 — 다른 원인일
가능성이 더 크다. 4단계 없이 1~3단계만으로도 전환하지 않는다 — 정황 증거일 뿐 확정적이지 않다.

QPLEX 대신 **QTRAN**을 먼저 고려할 것: 구조 변경 폭이 더 작고(mixer만 교체, dueling 분해 불필요),
distributional 확장 설계도 QPLEX보다 덜 침습적이다. 1~4단계 기준을 통과했을 때 QTRAN으로 먼저
검증한 뒤에도 개선이 부족하면 그때 QPLEX를 고려한다.
