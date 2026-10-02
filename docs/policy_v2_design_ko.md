# Policy v2 — 기억 있는 정책 학생, 임의 휴리스틱 교사, 세밀한 행동공간

> 브랜치 `feat/policy-v2` (2026-09-30 분기, `feat/v17-distill`의 `1c70ea4`에서).
> 실험 실행은 안정 브랜치 작업트리 `../blackout-env-runs`(`feat/v17-distill`)에서 한다 — 아래 §7.
> 관련 문서: `docs/kda_temporal_design_ko.md`(시간축 설계 검토), `blackout_env/model/kda_architecture_plan.md`
> (원안), `docs/distill_plan_ko.md`(D1–D6 증류 결과).

## 1. 왜 바꾸나

| 전제 | 결과 |
|---|---|
| RL 미세조정을 하지 않는다(당분간 가능성 낮음) | Q 학습 장치(IQN 분위수 헤드, SPR 보조 손실, target net)가 증류에선 비용만 든다. IQN 헤드는 순전파의 절반이었다(`distill_step_bench.py`) |
| 목표: **아무 휴리스틱이나 교사**로 — 내부 상태가 있는 것 포함 | V4/V17/V19/새 전략은 행동의 18–32%가 이력에 따라 달라진다(같은 관측을 이력 없는 새 인스턴스에 준 것과 비교; V21-teach는 0%). 기억 없는 학생은 이만큼 못 따라 한다 → **기억(KDA)** |
| 상대 조건부 교사도 만든다 | 상대를 식별하려면 여러 틱의 관측이 필요 → 역시 기억 |
| Unity는 행동의 **방향만** 쓴다(`Unit.Move`: `input.normalized`) | 행동공간 = 원 위의 임의 방향 ∪ {정지}. 8방향은 모델의 제약일 뿐 → 세밀한 방향 + 임의 각도 이동 휴리스틱 |

사용자 결정(2026-09-30): 리셋 단위 = **경기**, 추론이 조금 느려져도 OK(틱당 +0.5–1 ms 이상 허용).

## 2. 아키텍처

```
관측 o_t ─► trunk (지금 MyModel: CNN 36 vision + 10 unit + 1 team_state 토큰, GQA×4, RoPE, XSA)
          ─► 시간 모듈: 슬롯별 KDA (유닛 10 + team_state 1 토큰, 잔차, 출력 투영 0-초기화)   ← 새로
          ─► 정책 헤드: 유닛별 K개 방향 구간 + STAND 로짓                                   ← 교체
```

- **시간 모듈 (b2)**: `docs/kda_temporal_design_ko.md`의 권장안. trunk와 헤드 사이에만 들어가고 헤드와 무관하다.
  출력 투영이 0-초기화라 기존(무상태) 체크포인트에서 **출력이 정확히 같은 상태로** 웜스타트된다.
  원안(인코더→KDA→디코더, trunk 2회)은 b2가 이득을 보이고 포화된 뒤에만.
- **정책 헤드**: IQN 대신 평범한 로짓. 방향 K개 구간(32 또는 64, 11.25°/5.6°) + STAND.
  집합값 교차엔트로피를 그대로 쓴다 — 라벨 = 허용되는 방향들의 **호(arc)**(+ 교사가 서면 STAND).
  추론: 최대 확률 구간의 중심 방향, 또는 상위 호의 원형 평균(흔들림 완화).
  von Mises 혼합 같은 연속 분포보다 다봉(동점 방향)이 자연스럽고 학습이 안정적이라 1순위.
- **호환성**: 헤드 교체로 헤드는 새로 배우지만 trunk는 D4/D5에서 가져온다. 8방향+STAND(현재 `--stand-action`)는
  K=8 특수 경우로 남긴다.
- **구현(P4, §9)**: 가장 싼 변경을 택했다 — IQN 헤드를 그대로 두고 마지막 선형층 출력만 K(+STAND)개로.
  8방향 체크포인트의 방향 행을 각도로 보간해 K구간 행을 만들므로 웜스타트 직후 **같은 방향으로 움직인다**.
  IQN 제거는 별도 실험 P5(§10, `--policy-head logits`).

## 3. 교사

- **`act:<서술자>` 교사**: 어떤 휴리스틱이든 그 실제 행동을 라벨로(방향 → 해당 구간, 0 → STAND).
  상태 있는 교사는 **매 틱 한 번**, 학습자 관측으로 호출돼야 한다(학생이 조종한 틱에도 — 교사 상태가 실제
  궤적을 따라가야 함). β=1 경기는 휴리스틱을 그대로 재현.
- **teach 교사**(v17/v20/v21-teach): 집합 라벨. K구간에선 8방향 집합을 호로 넓히거나, 임의 각도 teach로 다시 만든다.
- **상대 조건부 교사**: 상대 정체/행동 이력에 따라 대응을 바꾸는 교사 — 기억이 필요한 라벨의 원천. 새 상대들
  (`strategic_raider/cargo_hunter/early_rush/fortress/duelist`, `perturb:` 래퍼)이 그 대상 풀.
- **임의 각도 휴리스틱**: A* 경로 직선화(시야·충돌 여유 확인)로 임의 방향 이동. 이득(이동 효율, V21 대비 점수차)을
  먼저 잰다 — 이득이 작으면 K=8+STAND로 두고 헤드만 단순화.

## 4. 데이터와 학습

- 샤드에 **모든 full-window 틱을 순서대로**(라벨 없는 틱 포함) + `tick` 필드. 기존 샤드는 `team_state[2]`로 tick 복원.
- 링은 행마다 경기 식별자와 tick을 갖고, `sample_windows(B, length, burn_in)`이 경기 경계·링 덮어쓰기를
  넘지 않는 윈도우 + `valid`/`reset`/`loss_mask`를 준다.
- 학습: 윈도우 [burn-in + length]를 trunk에 B·T 프레임으로 한 번에, KDA는 chunk 병렬, 손실은 burn-in 뒤만.
  burn-in 프레임의 trunk는 no_grad 선택.
- DAgger: 틱별 β 섞기는 그대로(학생 기억은 관측 이력의 함수). 워커의 학생은 경기 동안 기억을 들고 다니고
  `reset()`에서 0으로.
- 행동 라벨 저장: K비트 → uint32/uint64(지금 uint8).

## 5. 비용

- KDA 코어: MPS fwd+bwd (B=64, T=64, H=4, d=64) 8 ms(chunk) vs 64 ms(틱 루프). 레이어 전체는 trunk의 약 2%.
- 워커 추론(배치 1, CPU 1스레드): 5–11 슬롯 step +0.5–0.8 ms/틱(현재 학생 약 2 ms/틱).
- 학습: 윈도우 T 프레임의 trunk가 비용을 결정 — 배치를 윈도우로 나눠 같은 프레임 수로 맞춘다.

## 6. 실험 계획 (작고 결정적인 것부터)

| # | 실험 | 결정 |
|---|---|---|
| P0 | 임의 각도 V21 vs 8방향 V21 (이동 효율, 맞대결) | 이득 있으면 K구간 헤드, 없으면 K=8+STAND |
| P1 | `act:strategic_v19` 교사로 무상태 학생 vs KDA 학생(같은 예산) — 교사 행동 일치율, 점수차 | KDA가 상태 있는 교사를 더 잘 따라 하는가(18%의 이력 의존분을 얼마나 회수하나) |
| P2 | 상대 식별 프로브: 학생 trunk 동결 + KDA/GRU/무상태 프로브로 상대 정책 id를 경기 2/5/10/20초에 예측 | 식별이 늦거나 격차가 없으면 상대 적응용 기억은 접는다 |
| P3 | 상대 조건부 교사 → KDA 학생 vs 무상태 학생 | 기억의 실제 가치(점수차) |
| P4 | 세밀한 방향(K구간) 헤드: D6에서 시작, K=32로 v21-teach(넓힌 집합) vs `act:strategic_v21_anyangle` — 속도·일치율·점수차 (§9) | K구간이 8방향보다 나은가, 임의 각도 교사를 따라 하는가 |
| P5 | IQN 분위수 헤드 제거 → 평범한 로짓 헤드(`--policy-head logits`): p4a와 같은 설정, D6에서 접어(fold+fit) 시작 (§10) | 분위수 장치 없이 같은 성능을 더 싸게 내는가 |

판단은 항상 보류 시드 평가(상대당 24경기, 짝 차이)로.

## 7. 브랜치와 실행 환경

- `feat/v17-distill` — 안정. D5/D6 같은 실험은 여기서 실행: 작업트리 `/Users/mac/project/26rl/blackout-env-runs`
  (checkpoints, build, datasets, .venv, reports의 결과 디렉터리는 원래 체크아웃으로 심볼릭 링크). 실행 스크립트는
  `cd` 후 `PYTHONPATH=<작업트리>`를 설정 — `.venv`의 editable 설치가 원래 체크아웃을 가리키기 때문(`examples/*.py`는
  sys.path[0]가 examples/라 PYTHONPATH가 없으면 원래 체크아웃 코드를 부른다).
- `feat/policy-v2` — 이 문서의 개발. 원래 체크아웃(`/Users/mac/project/26rl/blackout-env`).
- 교훈(2026-09-30): 한 체크아웃에서 코드를 고치는 동안 그 코드로 평가를 새로 띄우면, 워커가 편집 중인 파일을
  불러온다 — D5 보류 평가 94경기를 버리고 다시 돌렸다.
- Unity: `blackout` 저장소 `perf/mapobs-on-decision`(맵 방송을 결정 틱에만). `build/mac`이 그 빌드, 이전 빌드는
  `build/mac_pre_mapobs_20260930`.

## 8. 결과

### P1 (2026-10-01, `reports/p1_heldout`)
`act:strategic_v19` 교사, 20k 스텝, D5n + STAND에서 시작, β 0.5→. 상대당 24경기.

| 상대 | 무상태 | KDA | KDA − 무상태 (짝) | V19 자신 |
|---|---|---|---|---|
| V4 | −7.0 | +16.9 | +23.8 ± 10.4 | +36.2 |
| V17 | −23.4 | −24.2 | −0.8 ± 12.5 | +43.2 |
| V21 | −42.6 | −10.8 | +31.8 ± 10.9 | −3.4 |

- 전체 짝 차이 +18.3 ± 6.6 (72쌍): 기억이 상태 있는 교사 모방에 도움. 일치율 0.55–0.71 → 0.65–0.73, 방향 반전 절반 이하.
- 그러나 둘 다 교사 V19보다 크게 약하다(V17 상대 −67). act 라벨(한 방향)만으로 20k 스텝은 부족 — 집합 라벨 교사(D6)보다 정보가 적다.

## 9. P4 — K구간 방향 헤드 (`--action-bins K`, 2026-10-01 구현)

무엇이 바뀌었나 (K=8이면 이전과 비트 단위로 같다 — 같은 시드의 수집 경기 18개에서 행·라벨·카운트를 이전 커밋과 비교):

- **행동공간** (`blackout_env/model/direction_bins.py`): 구간 k = 각도 2πk/K, K ∈ {8,16,32,64}. 구간 0, K/8, 2K/8, …이
  지금의 8방향과 일치. 출력 폭 n에서 K와 STAND 여부를 읽는다(n이 홀수면 마지막이 STAND). 팀 B 반사는 k → (K/4 − k) mod K.
- **헤드**: `MyModel(n_actions=K [+1])` — IQN 헤드 유지, 마지막 `Linear`만 K행. 순전파 비용 차이는 무시할 만하다.
- **웜스타트** (`run.load_weights`): 원본의 방향 행(가중치·편향)을 각도로 선형 보간 — 나침반 구간은 정확한 복사, 그 사이는
  양옆 행의 보간. 로짓도 똑같이 보간되므로 K구간 argmax는 늘 원본 방향 중 하나다 → D6를 `--action-bins 32`로 불러도 0스텝에서
  같은 행동(`tests/test_direction_bins.py`). STAND 규칙은 그대로(없던 STAND는 최저 편향 − 4).
- **라벨**: K비트 집합, K>8이면 uint64로 저장(샤드 meta의 `action_bins`). 교사는 여전히 8방향으로 라벨하고
  `collect.bin_labels`가 K구간으로 옮긴다.
  - 집합 교사(v21-teach 등): 허용 방향마다 ±`--label-width-deg`(기본 22.5°, 경계 포함) 안의 구간. K=32면 방향 하나 = 5구간,
    이웃한 두 방향은 22.5° 구간을 공유한다.
  - act 교사: 8방향으로 스냅한 라벨이 아니라 **실제 연속 이동**(env처럼 [−1,1] 클립 후)의 가장 가까운 구간 ±`--act-arc-bins`.
    0 이동 → 라벨 없음(STAND 대상), 이전과 같다.
  - 교사가 조종한 틱은 교사의 정확한 이동을 실행(act 교사는 원래 연속 이동, 집합 교사는 `label_action`) — 바뀐 것 없음.
  - 옛 uint8 샤드도 읽힌다: `DistillDataset(action_bins=K)`/`new_shards(..., action_bins)`가 불러올 때 ±22.5°로 넓힌다
    (`convert_bits`; K→8 축소도 된다).
- **실행 행동**: 기본은 최대 로짓 구간의 단위벡터. `--inference arc_mean`(그리고 held-out의 `--inference arc_mean`)은
  최대 구간에서 양옆으로 로짓이 최대−1 이상인 동안 이어지는 **연속 호**의 확률가중 원형 평균 — 이웃 구간 사이 흔들림 완화.
  `iqn/fixed_tau/tie_mean`도 K구간에서 동작. `_FastNet`(trace)과 `MyPolicy`는 출력 폭에서 K를 읽으므로 그대로,
  `load_net`도 state dict에서 크기를 읽는다. 체크포인트 `distill` meta에 `action_bins`, `stand_action` 기록.
- **지표**: `agree` = 학생 이동이 K구간 집합 안(v21-teach를 넓힌 집합이면 8방향 때와 같은 ±22.5° 의미, act 교사 ±1구간이면 더
  엄격). K>8이면 `agree8/*`도 기록 — 학생 이동을 8방향으로 스냅해 교사의 원래 8방향 라벨에 넣은 것, 즉 K=8의 `agree`와 같은
  정의라 D6와 직접 비교된다. 반전(`reversals_per_1000`)은 원래 이동 벡터의 각도(>135°)라 K와 무관. 새로
  `mean_turn_deg/{student,teacher}`(연속 이동 사이 평균 회전각) — 이웃 구간 흔들림은 반전이 아니라 여기서 보인다.
  `distill_report.py`의 반전 표도 샤드의 K를 읽는다. held-out 평가(`distill_heldout_eval.py`)는 기본 K=8 라벨로 기록하므로
  K구간 학생도 D6와 같은 8방향 일치율로 비교된다.

실행:

```
python -m blackout_env.train.distill.run --action-bins 32 --teacher v21-teach --init checkpoints/distill/d6/latest.pt ...
python -m blackout_env.train.distill.run --action-bins 32 --teacher act:strategic_v21_anyangle --act-arc-bins 1 \
    --init checkpoints/distill/d6/latest.pt ...      # --inference arc_mean: 수집·평가 학생이 원형 평균으로
```

실험 스크립트 `checkpoints/distill/launch_p4.sh`: 작업트리 `/Users/mac/project/26rl/blackout-env-p4`(P4 커밋에 고정,
p1처럼 checkpoints·build·datasets·.venv·runs와 reports의 결과 디렉터리를 심볼릭 링크)에서, D7 체인이 끝난 뒤
(`reports/distill_heldout/stdout_d7.log`의 `D7 CHAIN DONE`) 시작. 두 팔 `p4s`(v21-teach 넓힘), `p4a`
(`act:strategic_v21_anyangle`, ±1구간), D6 초기화·D6 상대 풀·설정, 20k 스텝. 이어서 보류 시드 평가(D6 기록과 짝,
iqn·arc_mean), p4a vs 임의 각도 V21 자신(`reports/p4_heldout`), OOD.

스모크(2026-10-01, `act:strategic_v21_anyangle`, D6 초기화, 2 워커, 200 스텝, `--train-quantiles 8 --amp fp16 --compile`,
D7s 학습·p3 평가와 기계 공유 중이라 절대값은 낮다):

| | 학습 속도 (step 150–200) | eval 일치율 agree / agree8 (v2–v4, 2경기씩) |
|---|---|---|
| K=8 | 6.0–6.2 step/s | 0.68–0.69 / – |
| K=32, ±1구간 | 5.6–6.0 step/s | 0.77–0.80 / 0.73–0.76 |

- 속도 차이는 측정 잡음 수준(헤드 마지막 층만 커짐). 체크포인트 meta `action_bins: 32`, 샤드 라벨 uint64 확인.
- D6 가중치를 K=32로 불러 D6 eval 샤드 256행에 적용: 유닛별 argmax 100% 일치, 나침반 구간 로짓 차 최대 4e-6.
- 점수차는 2경기라 판단 근거가 아니다 — 판단은 `launch_p4.sh`의 보류 시드 평가로.

## 10. P5 — IQN 제거, 평범한 로짓 헤드 (`--policy-head logits`, 2026-10-01 구현)

RL 미세조정 가능성이 낮으니(§1) 분위수 표본·tau 입력은 증류에서 비용일 뿐이다. 기본은 여전히 `iqn`이고
D6 가중치·고정 tau 출력·새 모델 초기화까지 이전 커밋과 비트 단위로 같다(SHA-1 비교).

- **헤드** (`blackout_env/model/modules/logits_head.py`): `MyModel(policy_head="logits")`의 `q_head = LogitsHead` —
  IQN의 `value_head`(RMSNorm-SwiGLU → Linear)는 그대로, tau 임베딩 `e(τ) = ReLU(tau_proj(cos(πiτ)))` 대신 학습되는
  채널별 입력 배율 `in_scale` 하나. 분위수마다 한 번이 아니라 **한 번** 평가. 키 = IQN 키 − `tau_proj.*` + `in_scale`,
  그래서 `config_from_state_dict`/`load_net`이 키로 헤드 종류를 알아낸다(체크포인트 `distill` meta에도 `policy_head`).
- **반환 형태**: `q_values` = 로짓 [B, N, A], `quantile_values` = 크기 1 분위수 축 [B, N, 1, A], `tau` = None —
  `q_values, *_ = net(...)` 호출부는 그대로. `n_quantiles`/`tau`/`--train-quantiles`는 무시된다.
  QMIX 학습기는 IQN 전용으로 남는다(기본값이라 영향 없음).
- **웜스타트 정확성**: 단일 τ면 `in_scale = e(τ)`로 **정확히** 같은 출력(테스트, 오차 <1e-5). 그러나 추론·학습이 읽는 건
  고정 τ 32개에 대한 **비선형 평가 32번의 평균**이라 배율 하나로는 정확히 접히지 않는다(정확하려면 SwiGLU를 32배 넓혀야 하고,
  그건 IQN 비용 그대로). 그래서 두 단계:
  1. **fold** (`MyModel.iqn_to_logits_state`, `--init`/`load_weights`가 자동 적용): `in_scale` = 32개 고정 τ 임베딩의 평균.
  2. **fit** (`python -m blackout_env.train.distill.head_fit`): trunk 고정, 캐시한 trunk 특징(D6 샤드 16k행 × 5유닛)에서
     로짓 헤드를 IQN 고정 τ 평균 로짓에 맞춘다(softmax KL + 작은 MSE, 1500 스텝, CPU ~1분). 결과는 체크포인트로.

  D6 eval 샤드 4096행 × 5유닛(학습에 안 쓴 행), 기준 = IQN 고정 τ 32개 평균:

  | | 최대 로짓 차 | 평균 로짓 차 | argmax 일치 | 1·2위 차 >0.5인 유닛(95%)에서 |
  |---|---|---|---|---|
  | fold | 5.51 | 1.65 (대부분 행 공통 이동, 중심화 최대 4.36) | 99.63% | 100% |
  | fit | 0.58 | 0.056 | **99.80%** | 100% |
  | 참고: IQN 자신의 `iqn` 추론(무작위 τ 32개) | 1.10 | 0.17 | 99.42% | 100% |

  즉 fold만으로도 IQN이 자기 무작위 τ 추론과 일치하는 것보다 더 잘 일치하고, 불일치는 전부 1·2위가 거의 동점인 유닛이다.
  K=32(P4 보간 후)도 같은 숫자(보간은 선형이라 argmax 보존).
- **추론** (`inference.py`): 로짓 헤드는 tau가 없으니 `iqn` = `fixed_tau` = 평범한 argmax(같은 이동), `tie_mean`/`arc_mean`은
  로짓을 그대로 읽는다. `_FastNet`은 tau를 뽑지도, trace 입력으로 넣지도 않는다(무상태·`--temporal` 모두).
- **학습** (`run.py --policy-head logits`): 손실은 같은 집합값 CE + z-loss. `--init` IQN 체크포인트 → fold → (K가 다르면) P4 보간
  → (temporal이면) 기억 블록 항등 초기화. 로짓 체크포인트를 IQN 학생에 넣으면 거부.

속도(기계를 학습 런과 공유 중 — 같은 프로세스에서 번갈아 3회씩 잰 상대값):

| | IQN q8 | IQN q32 | logits |
|---|---|---|---|
| 학습 스텝 (MPS, B=256, fp16, compile; `distill_step_bench.py --policy-heads iqn logits --repeats 3`) | 144 ms | 164 ms | 133 ms (q8 대비 −7%, q32 대비 −19%) |
| 워커 틱 (CPU 1스레드, 배치 1, trace, K=32, D6 eval 관측 400틱 × 12회, 최소/중앙) 무상태 | 2.67 / 3.42 ms | – | 2.23 / 2.78 ms (−0.5 ms) |
| 같은, temporal | 3.14 / 3.72 ms | – | 2.83 / 3.49 ms (−0.3 ms) |

절대값은 부하 때문에 P4 기록(84 ms/스텝)보다 크다. 이득은 IQN 헤드 몫만큼 — 학습 q8에선 작고, 학습 비용의 대부분은 역전파의 trunk다.

스모크(2026-10-01, `--policy-head logits --action-bins 32 --act-arc-bins 1 --teacher act:strategic_v21_anyangle`, D6 초기화(fold),
2 워커, 200 스텝, `--amp fp16 --compile`, P1 학습과 기계 공유): 6.2–6.6 step/s(throttle 없는 구간), eval 일치율 agree 0.77–0.80 /
agree8 0.74–0.76(2경기씩) — P4 K=32 IQN 스모크(5.6–6.0 step/s, 0.77–0.80 / 0.73–0.76)와 같은 수준. meta `policy_head: logits` 확인.

실험 스크립트 `checkpoints/distill/launch_p5.sh`: 작업트리 `/Users/mac/project/26rl/blackout-env-p5`(P5 커밋에 고정, p4와 같은 링크)에서
P4 체인이 끝난 뒤(`reports/distill_heldout/stdout_p4.log`의 `P4 CHAIN DONE`) 시작. 한 팔 `p5a`: `head_fit`으로 D6 → 로짓 헤드
(`checkpoints/distill/p5a/init_logits.pt`), p4a와 같은 설정(D6 상대 풀, `--action-bins 32`, `act:strategic_v21_anyangle` ±1구간,
20k 스텝)에 `--policy-head logits`. 이어서 보류 시드 평가(`reports/distill_heldout`, v21-teach, iqn·arc_mean — D6/p4와 짝),
p4a와 짝이 되는 자기 교사 평가(`reports/p4_heldout`), OOD. 판단: p5a − p4a 짝 차이가 0 근처면 IQN을 기본에서 뺀다.

### 학습률 일정 (`--lr-schedule`, 2026-10-02 구현 — D8 후속)

D8에서 수렴한 D6를 3e-4로 이어 학습하면 ~10점 잃고 1e-4면 수준 유지 → 이어 학습·처음부터 학습 모두 끝에서 lr을 낮추도록.
- `--lr-schedule {constant,cosine,linear}` (기본 `constant` = 이전과 비트 단위로 같음: 옵티마이저 lr을 아예 건드리지 않음),
  `--lr-warmup W` (기본 0; 0에서 `--lr`까지 선형, 스텝 s<W에서 `lr·(s+1)/W`), `--lr-final F` (기본 0.1; 마지막 스텝 `--steps − 1`의 lr = `lr·F`).
  cosine/linear는 스텝 W의 `--lr`에서 마지막 스텝의 `lr·F`까지 감쇠, 그 뒤로는 `lr·F` 유지.
- 전역 스텝과 `--steps`만의 함수(`run.lr_at`)라 저장할 스케줄러 상태가 없다: 매 스텝 param_group lr을 덮어쓰고 `train/lr`로 기록.
  재개하면 같은 곡선을 이어간다. **재개 때 `--steps`를 늘리면 현재 스텝에서 곡선을 다시 계산**하므로 감쇠가 늘어나 lr이 다시 올라갈 수 있다.
- 인자는 `run_info.json`(모든 인자)과 체크포인트 `distill` meta(`lr`, `lr_schedule`, `lr_warmup`, `lr_final`, `steps`)에 남는다.
- 스모크(cosine, W=20, F=0.1, 120 스텝, 2 워커): `train/lr` 3e-4(20) → 1.25e-4(80) → 3e-5(120), 테스트 `tests/test_lr_schedule.py`.

## 11. S2 시뮬레이터 (`blackout_env/sim`, 2026-10-02, 브랜치 `feat/sim`)

탐색 교사(S3/S4)가 미래를 굴려 볼 수 있도록 Unity 빌드(`build/mac/BlackOut.app`)의 게임 규칙을 Python/numba로 옮겼다.
C# 코드(`blackout/Assets/Project/Runtime/Scripts`)를 메서드 단위로 옮기고(함수마다 대응 C# 메서드를 docstring에 적음),
기록한 Unity 경기를 그대로 재생해 맞췄다. **결과: 기록한 60경기 전부가 첫 관측부터 끝까지 관측 비트 단위로 같다**
(위치·들고 있는 아이템·클래스·점수·타이머·의미 지도 전부, 경기 끝의 결정적 적재까지).

### 무엇을 모델링하나

- **틱 구조**: 결정 1회 = FixedUpdate 2틱(decisionPeriod 2, 두 번째 틱은 같은 행동 반복). 틱마다 10유닛의 `Unit.Move`를
  ML-Agents `AgentAct` 순서로, 그다음 세계 갱신(`GameScenario.EpisodeUpdate`: 특수 아이템 생성기 → 경기 타이머 → 흡수 타이머).
- **이동**: `input.normalized * dt * MoveSpeed`, 축 분리(X 다음 Y) `ClampAxis`, 이동 방향 쪽 8이웃의 막힌 칸(정사각형, 사이트 0.9)에 대한
  원-정사각형 밀어내기(벽은 **크기 버프 없는** 클래스 반지름), 팀별 통행(벽 전부 막힘, 기지 바닥·사이트·스폰은 적만 막힘, 창고 통과).
  정지 입력이어도 `ProcessInteractions`는 돈다.
- **상호작용**: 이동 직후 이동한 유닛 기준 3×3 통행 가능 칸의 `MapObjects`(리스트 순서까지 재현: 추가는 뒤, 제거는 앞으로 당김)를
  스냅샷 순서로 훑고, **UnitSize가 곱해진** 원끼리 겹침 판정. 전투(Beats: 헌터는 모두, 헌터-헌터·운반자-운반자 동귀어진, 수집자는 운반자를 이김),
  사망 = 아이템 파괴 → 스폰 텔레포트 → 수집자로 리셋 → 잠금 해제. 죽은 뒤에도 같은 루프가 옛 위치·새 클래스로 계속 도는 C# 동작까지 그대로.
- **아이템**: 줍기(수집자·운반자, 하나만, 자기 땅의 아이템은 못 줍는 IgnoreFriend → 적 창고에서 훔치면 적 점수 감소),
  창고 **영역 진입 때만** 적재(원자적 계획: 먼 칸부터 같은 종류 합치기·빈 칸, 모자라면 취소), 수량 갱신(Refresh 퇴장/재진입),
  창고 안에서만 효과(점수, 속도·크기 수정자: 합연산 계수, `max(0, ·)`), 20초 흡수(창고 아이템 소멸·점수 고정),
  헌터 변신 때 아이템 파괴, 특수 아이템 생성(이전 것이 사라진 뒤 10초, 중립 바닥 무작위 칸, 종류 무작위).
- **성소**: 2×2 중앙(수집자 → 헌터), 팀별 잠금 성소(수집자 → 운반자, 운반자가 죽을 때까지 잠금). 영역 진입 때만.
- **종료**: 100점 도달(점수 변경 알림 시점, 틱 중간이면 거기서 끝) 또는 420초.
- **관측 렌더링**: `MapObsAgent`의 44-float 방송과 `SemanticMapRenderer`의 packed 지도를 같은 float32 식으로 만들고,
  env의 전처리기(`MyObsPreprocessor`)를 그대로 통과시켜 `BlackOutEnv`와 같은 관측 dict(`sim.obs()`)를 낸다.

### 조사 요약과 달랐던 점 (재생으로 확정)

| 항목 | 실제 | 근거 |
|---|---|---|
| dt | `Time.fixedDeltaTime` = **0.019999992f** (0.02f보다 4 ulp 작음), time_scale 1·20 같음 | 0.02f로는 흡수 타이머가 20초 안에 한 틱 어긋남; 자유 이동 일치율 93%→100% |
| 부동소수 | Mono가 float 식을 double로 계산하고 저장 때 반올림: 벡터 크기 `x*x+y*y`·겹침 거리·스탯 합이 double | 이걸 안 하면 스텝의 ~6%에서 1–2 ulp 차이 |
| AgentAct 순서 | 유닛 **[3, 6, 2, 7, 5, 8, 4, 9, 1, 0]** (BlackOutAgent OnEnable 순서) — env 관측 dict의 에이전트 순서와 같다 | 같은 성소·같은 적에 동시에 닿는 경합에서 결정; 쌍 교환 탐색(`sim_validate.py --fit-order`)으로 40경기를 끝까지 맞추는 유일한 국소 최적. RandomWalker가 dict 순서대로 난수를 뽑는 것까지 재현 |
| 첫 관측 | 보통 경기는 리셋 뒤 **2틱 지난** 시점(유닛 이동 없이 타이머만), 플레이어 시작 직후 첫 경기만 0틱이고 첫 결정에 이동이 한 번 | 시간 관측값 |
| 맵 | 선택 안 된 창고 후보 칸은 **Floor**(배터리·특수 아이템이 생긴다), 장식용 창고 바닥 2×2 두 곳((1–2,6–7), (6–7,1–2))은 아님 — 씬 파일과 빌드가 다르다 | 40경기 초기 배터리 배치 + 시드 RNG 재현 |
| 난수 | `UnityEngine.Random` = MT 초기화 + xorshift128, `value = (u & 0x7FFFFF)/8388607`, `Range(a,b) = a + u % (b−a)` | 시드로 창고 선택·배터리 위치/수량·특수 아이템 종류/칸까지 정확 (플레이어 시작 직후 첫 경기만 재현 안 됨) |
| 텔레포트 | `Teleport`의 `ProcessInteractions`는 생략 — 두 스폰의 3×3은 자기 기지 바닥과 벽이라 적도 아이템도 있을 수 없다 | 재생 일치 |

### API

```python
from blackout_env.sim import SimState
sim = SimState.from_seed(seed, elapsed_ticks=2)      # env.reset(seed) 직후와 같은 상태 + Unity RNG 상태
sim = SimState.from_raw(raw_state, raw_map)          # MapObsAgent 방송 하나에서 (관측 dict는 from_obs)
sim = SimState.from_history(raw_states, raw_maps, moves)   # 직전 방송들로 위치를 float32 정확히
sim.step(moves)            # float32 [10, 2] (유닛 번호 순), 결정 1회 = 2틱
obs = sim.obs()            # BlackOutEnv 형식 {unit_i: {graphic, team_state, agent_states}}
fork = sim.copy(); sim.load(fork)   # 롤아웃용 복사(~4 µs) / 제자리 복원(~3 µs)
sim.reseed(k)              # 특수 아이템을 Unity 분포에서 표본; force_spawns / force_timed_spawns 로 강제
sim.events(), sim.scores, sim.over, sim.winner
```

상태는 numba 배열 묶음(`core.py` 머리말): 유닛 위치·클래스·든 아이템·소속 칸, 아이템 슬롯(종류·수량·상태·칸·적용 효과 팀·수정자),
칸별 `MapObjects` 리스트, 팀별 수정자 리스트, 점수, 세 타이머, dt, RNG 상태, 성소 잠금, 강제 생성 목록, 이벤트 기록.

**관측 하나로 복원 안 되는 것** (`from_raw`): ① 특수 아이템 생성 타이머 — 특수 아이템이 있으면 0으로 정확, 없으면 [0,10)초 미지
(`spawn_timer=` 지정 또는 표본), ② RNG 상태 — 시드를 알면 정확(`from_seed`), 아니면 무작위 상태 = Unity 분포에서 표본,
③ 한 칸 안 `MapObjects` 순서(아이템 먼저, 유닛 번호 순으로 가정 — 한 유닛이 같은 틱에 같은 칸의 여러 물체와 겹칠 때만 영향),
④ 수정자 삽입 순서(합의 반올림만), ⑤ **위치의 마지막 1–2 ulp**: `pos/24*2−1`이 float32 위치 여럿을 같은 값으로 보낸다.
⑤는 `from_history`가 직전 방송·행동으로 후보를 굴려 거의 다 없앤다(정지해 있던 유닛은 못 가림).
나머지(클래스, 아이템·수량, 점수, 타이머 = float32 누적 표에서 경과 틱을 찾아 정확, 성소 잠금 = 살아 있는 운반자, 적용 효과)는 정확.

### S0 기록

`examples/sim_record_traces.py`: BlackOutEnv를 헤드리스로 돌리며 `MyObsPreprocessor` 입력(원시 44-float 방송, packed 지도)과
`UnityEnvironment.set_actions`로 실제 보낸 float32 행동(클리핑 뒤)을 결정마다 저장. 증류 샤드 형식·`collect.py`는 건드리지 않았다.
정책 풀: 임의 각도 V21, P5a 학생(`ckpt:checkpoints/distill/frozen/p5a_v22.pt`), V21, V17, mixture, random (보류 세트: V19, V4 추가).
`reports/sim/traces` 40경기(4 워커), `reports/sim/traces_holdout` 20경기(다른 시드, 순서·dt 맞춤에 안 씀), `reports/sim/traces_ts1` time_scale 1 두 경기(30초).
고정 데이터 `tests/data/sim_trace_short.npz` = trace_007(V21 대 P5a, 595결정, 42-100).

### 검증 (`examples/sim_validate.py`, `reports/sim/validation*/validation.json`)

기록된 양 팀 행동을 열린 고리로 재생, 특수 아이템은 기록에서 강제, 같은 관측 역변환으로 위치 비교(세계 단위).

| | 맞춤 세트 40경기 | 보류 20경기 |
|---|---|---|
| 첫 관측부터 경기 끝까지 관측 비트 일치 (시드 초기화 / 관측 초기화 둘 다) | **40/40** | **20/20** |
| 같은 재생, 특수 아이템을 기록 대신 **Unity RNG**로 (시드 재현 가능한 경기) | 36/36 | 17/17 |
| 닫힌 고리: 휴리스틱 정책을 시뮬레이터 관측 위에서 다시 돌림 → 기록 행동과 매 스텝 동일 | 29/29 | 14/14 |
| 점수 시계열·최종 점수·승자 | 40/40 | 20/20 |
| 이벤트(줍기·적재·탈취·사망·변신·흡수·점수 변화) 일치 | 10643/10643 | 5026/5026 |

경기 중간(10초마다, 맞춤 세트 540개 / 보류 180개 시작점)에서 Unity 관측 하나로 초기화:

| 구간 | 틱별 위치 오차 중앙/p99/최대 | 시작점 중 최대 오차 ≤1e-3 | 같은, `from_history`(직전 25방송) |
|---|---|---|---|
| 1초 | 0 / 1.4e-6 / 1.9e-4 | 100% | 100% (오차 최대 2.9e-6, 비트 일치 387/540) |
| 5초 | 0 / 1.4e-6 / 7.5e-3 | 99.3% | 99.8% |
| 20초 | 0 / 2.9e-6 / 13.7 | 97.8% | 99.1% |

5초 창의 이벤트 일치는 두 방식 모두 4221/4221(보류 2040/2040). 보류 세트는 20초까지 전부 ≤3.4e-4.

**남은 차이**: 경기 중간 초기화에서만. ulp 단위 위치 모호성이 벽 모서리에 눌린 유닛(모서리 원-점 밀어내기는 방향이 위치에 민감)에서
커지고, 드물게 몇 초 뒤 겹침 판정(사망·줍기)을 뒤집는다(20초 창의 최대 13.7 = 한쪽만 죽어 스폰으로 감). `from_history`가 대부분 없앤다.
디버깅 순서(첫 어긋남 분석): AgentAct 순서(성소 경합, 헌터 동시 접촉) → 0.02f 타이머가 20초에 한 틱 밀림 → dt 4 ulp → double 크기 →
성소 위에서 초기화하면 진입이 다시 발동(초기화는 '배치'로) → 후반 float32 타이머 누적 오차로 경과 틱 추정 실패(누적 표로 해결) →
특수 생성 틱 모호성(검증은 기록 틱에 강제, `force_timed_spawns`).

### 속도 (`examples/sim_bench.py`, CPU 1코어, D9 학습과 기계 공유 — 상대값으로)

| | µs/결정 | 2초(50결정) 롤아웃/초 |
|---|---|---|
| 시뮬레이터만, `run`(행동열 한 번에) | 4.6 | 4300 |
| 시뮬레이터만, `step` 한 번씩(Python 호출 포함) | 6.8 | 2900 |
| 관측 dict 렌더링(전처리기) | +62 | – |
| 양 팀 random 정책 | 107 | 190 |
| 양 팀 V17 | 740 | 27 |
| 양 팀 임의 각도 V21 | 1760 | 11 |
| 양 팀 P5a 학생(배치 1, CPU) | 9960 | 2 |
| P5a 대 임의 각도 V21 | 5770 | 3.5 |

시뮬레이터 자체는 정책 비용의 1% 미만이다. S3에서 롤아웃을 많이 하려면 정책 쪽이 병목: 거시 행동(목표별 이동)을
시뮬레이터 안에서 싸게 실행하는 롤아웃 정책, 학생은 롤아웃 여러 개를 배치로 묶어 한 번에 추론, 또는 결정 몇 번마다만 정책 호출.
numba 첫 컴파일 ~40초(디스크 캐시 뒤 ~0.1초).
