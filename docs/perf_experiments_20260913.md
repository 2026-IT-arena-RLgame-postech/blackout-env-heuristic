# 학습 속도 실험 — 2026-09-13

QMIX 학습(`blackout_env/train/qmix_trainer.py`)의 병목을 줄이려고 시도한 두 가지 실험 기록.
측정은 모두 M-시리즈 Mac, MPS 디바이스, 실제 모델 shape(hidden=128, seq=47 토큰, train batch=64) 기준.

## 배경: 실측 병목 분석

`20260913-014025_53890` 런(993k/1,000k step, 27.2 steps/s)의 TensorBoard `perf/*` 스칼라 기준:

| 구간 | 비중 |
|---|---:|
| train_step (전체) | ~65% |
| ㄴ forward (net+target+ema+SPR) | ~59% of train_step |
| ㄴ backward+optim | ~35% of train_step |
| env.step + select_actions (Unity 통신) | ~35% |

결론: 병목은 Unity 시뮬레이션이 아니라 **forward pass의 커널 launch 오버헤드**. batch=64, hidden=128짜리
작은 모델을 net/target_net/ema_net 총 4번(SPR 포함) 따로 forward하는 구조라, 실제 연산량보다
디스패치 오버헤드 비중이 큼.

## 실험 1: `torch.compile` — 채택, `main`에 병합

`--compile` 플래그로 `net`/`target_net`/`ema_net`을 `torch.compile`로 감싸서 attention/RoPE/IQN head의
자잘한 op들을 그래프 단위로 퓨징, 커널 launch 수 자체를 줄임.

**정확성**: 동일 tau(quantile 샘플)를 고정하고 eager와 compiled 출력을 비교 — 차이 `~1e-7`(float32 노이즈
수준). tau를 고정하지 않으면 forward마다 새로 샘플링되는 랜덤성 때문에 큰 차이가 나는 것처럼 보이니
착각하지 말 것.

**속도**: 학습에서 실제로 쓰이는 3개 shape 전부에서 일관되게 1.5~1.6배 개선.

| shape | eager | compiled | speedup |
|---|---:|---:|---:|
| train (B=64, q=8) | 7.67ms | 4.91ms | 1.56x |
| select_actions (B=2, q=8) | 2.34ms | 1.43ms | 1.64x |
| SPR/ema (B=320, q=1) | 34.47ms | 23.15ms | 1.49x |

**주의**: `select_actions`(B=2)와 `train_step`(B=64/320, n_quantiles 8 또는 1)이 shape가 달라서
distinct shape마다 최초 1회 재컴파일 비용 발생 — 매 스텝이 아니라 3가지 고정 shape에 대해 한 번씩만.
학습 시작 직후 몇 초는 컴파일 대기로 오히려 느려 보일 수 있음.

커밋: `5a86c9a` (`blackout_env/train/qmix_trainer.py`에 `QMIXConfig.compile` + `--compile` CLI 플래그).

## 실험 2: Reversible Net (RevNet 스타일) — 폐기

트랜스포머 트렁크(`AttentionLayers`, depth=4)를 RevNet 스타일 additive coupling(`y1=x1+F(x2)`,
`y2=x2+G(y1)`, custom autograd Function으로 backward 시 activation을 저장 대신 재계산)으로 바꾸면
활성화 메모리를 O(depth)에서 O(1)로 줄일 수 있다는 아이디어. 별도 파일
(`reversible_attention_block.py` + `tests/test_reversible_block.py` + `examples/benchmark_reversible_block.py`,
현재 삭제됨 — 필요하면 이 문서의 기록으로 재구현)로 구현해서 검증.

**정확성**: naive(non-reversible) autograd 참조 구현과 float64로 비교, output/gradient 차이 `~1e-9~1e-10`
수준으로 backward 수식(부호, 파라미터 gradient 귀속) 정확함을 확인. closed-form inverse도 원본 입력을
정확히 복원.

**속도 (함정 주의)**: 순수 full-width plain(d=128) 대비로만 재면 reversible이 2배 가까이 "빠른 것처럼"
보이는데, 이는 reversible의 F/G가 채널을 반으로 쪼갠 64-width에서 돌기 때문 — Linear 비용이
width^2에 비례하므로 폭을 반으로 줄인 것만으로 이미 4배 가까이 싸짐. reversible 메커니즘 자체와는
무관한 착시. **같은 64-width plain(non-reversible) 스택을 대조군으로 넣고 다시 재면**:

| depth | plain (d=128) | plain (d=64, 폭 맞춘 대조군) | reversible (d=128 total) | reversible vs 대조군 |
|---:|---:|---:|---:|---:|
| 4 | 14.26ms | 6.76ms | 7.70ms | +14% |
| 8 | 26.90ms | 12.45ms | 14.42ms | +16% |
| 16 | 53.42ms | 24.60ms | 27.75ms | +13% |
| 32 | 105.67ms | 47.03ms | 54.40ms | +16% |

즉 폭을 맞추고 비교하면 reversible이 일관되게 **13~16% 더 느림** — RevNet 문헌 그대로, backward에서
F/G를 한 번 더 계산(recompute)하는 대가.

**메모리도 이득 확인 못함**: MPS엔 CUDA의 `max_memory_allocated` 같은 정확한 backward-peak 메모리
API가 없어서 `torch.mps.driver_allocated_memory()`로 대신 쟀는데, reversible이 오히려 대조군보다
더 많이 나옴(depth=8: 754MB vs 586MB). RevNet이 실제로 이기는 지표(활성화 메모리)를 이 플랫폼에서
신뢰성 있게 측정할 방법이 마땅치 않았고, 측정된 결과만 보면 이득이 없음.

**결론**: 지금 병목은 메모리 부족(OOM)이 아니라 커널 launch 오버헤드(위 배경 섹션)이고, 현재 모델은
batch=64/depth=4로 작아서 OOM 근처에도 안 감. Reversible net은 "메모리를 아끼는 대신 연산을 더 쓰는"
트레이드지 속도 개선 기법이 아니라서, 이 사이즈에선 얻는 것 없이 연산량만 +13~16% 느는 셈 — 폐기.
트렁크를 depth 32+ 수준으로 훨씬 깊게 갈 계획이 생기면(OOM이 실제 걸릴 정도) 재검토 가치 있음.
