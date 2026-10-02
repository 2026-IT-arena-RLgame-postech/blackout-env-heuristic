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

## 11. P9 — teacher_v2: 더 강한 교사 `strategic_v22` (2026-10-02, 브랜치 `feat/teacher-v2`)

동기: 증류 학생이 교사(임의 각도 V21)를 이미 넘었고(교사 대 P5a −29.5, 대 D6 +17.4), 강한 학생을 상대로 넣어도 교사 상한에 막힌다(D9).
→ 교사 자체를 강하게. 코드 `blackout_env/heuristics/v22_teacher.py` (`strategic_v22`, 기본 혼합 가중치에는 없음), 결과 `reports/teacher_v2/`.

### 진단 (교사 vs P5a/D6 48경기, 샤드 저장, `examples/teacher_loss_analysis.py`)
- 경기는 **5초 안에** 정해진다: 중앙 성소에서 먼저 Hunter가 된 쪽(양쪽 다 ~4.0초 도착)이 상대 보행자를 그 자리에서 잡고,
  상대는 V21 규칙(적 Hunter를 피하는 경로가 없으면 변신 안 함) 때문에 Hunter를 영영 못 얻고 기지 출구를 진치기당한다.
  첫 Hunter를 먼저 얻은 쪽이 48경기 중 41승. P5a가 24경기 중 13번, D6가 7번 먼저였다(교사가 진 경기: 0–1 Hunter, 수집기 사망 27/경기, 기지 근처 24).
- 원인(교사 라벨 재생): 보행자 셋이 가는 길에 필드 배터리를 밟고, V17-teach는 짐을 든 Collector를 변신 후보에서 빼서 **되돌아가 납품**한다.
  진 경기 대부분에서 보행자 전원이 그랬다.
- 경주를 이긴 뒤에도: 성소를 비워 두어 상대가 10–20초에 Hunter 3–4를 얻고 역전(성소 감시 없음); 성소 0.3칸 앞에서 적 Hunter가
  나타나면 보행자가 돌아서다 잡힘(V21 안전 경로 규칙); 반대 방향의 동점 목표 둘 사이에서 라벨이 무효 → 유닛이 **경기 끝까지 정지**
  (V19/V17 상대 경기의 0.7에서 5초 이상 정지, 경기당 ~200 유닛·초. V17-teach 계보 공통 버그).

### 바꾼 것 (모두 관측만의 함수 — 새 인스턴스와 행동 차이 0/3240, 스위치로 끌 수 있음)
| 스위치 | 내용 |
|---|---|
| `race_cargo` | 경주 중(우리 Hunter 0 또는 10초 이내) 배터리를 든 Collector도 Hunter 후보, 변신 목표가 납품보다 우선(배터리는 변신 때 소멸) |
| `item_free_walkers` | 보행자는 필드 아이템을 피하는 경로가 더 길지 않으면 그 길로(V18/V19 규칙) |
| `region_approach` | 변신·납품은 유닛 중심이 그 지역 타일에 **들어갈 때** 일어남(`MapRegion.OnUnitEnter`) → 보이면 타일 중심이 아니라 지역의 가장 가까운 점(0.1칸 안)으로 |
| `sentry` | Hunter 2+면 하나(초소에 가장 가까운 것)가 V19식 성소 초소(성소 중심에서 적 스폰 쪽 2.5칸): 성소 7칸 안 적 Collector를 친다 |
| `commit` | 안전 경로가 없어도, 적 Hunter가 정면으로 달려와도 닿기 전에 성소에 들어갈 수 있는 보행자는 계속 간다(위협 회피 끔) |
| `break_ties` | 동점 목표가 서로 상쇄되어 라벨이 무효면 경로상 가장 가까운 목표 하나로(가까이 갈수록 더 가까워져 안정) |
| `size_aware` | 크기·속도 아이템 반영(아래 A) |

### A. 유닛 크기 아이템 (Unity 소스 확인)
- `Unit.Move`는 이동 시스템에 `UnitData.CollisionBound`(클래스 값)를 넘긴다 → 벽·모서리 여유는 크기와 무관(`UnitMovementSystem`은
  비보행 타일하고만 충돌, 유닛끼리는 통과). `UnitInteractionSystem`(46–47행)은 `Unit.CollisionBound`(폭 × `UnitSize`)로 유닛–유닛·유닛–아이템
  겹침을 판정 → 잡기·줍기 반경만 크기를 따른다. Hunter–Collector 접촉 거리 = 0.325·s_H + 0.275·s_C (크기 1에서 0.6칸).
- 효과는 창고에 있는 아이템마다 하나씩, 같은 스탯은 합산(`TeamContext.CalculateStat`: 기본 × (1 + Σ)): BuffSize 그 팀 +50%,
  DebuffSize 상대 팀 전 유닛 −30%, BuffSpeed 그 팀 +50%, DebuffSpeed 상대 Collector −90%(둘이면 속도 0). 관측은 모든 창고 칸의 아이템을
  보여 주므로 개수를 센다(V17은 "있다/없다" 한 비트).
- `size_aware`: 위협 회피 반경과 Hunter 위험 구역(V20 진치기 회피, V21 안전 변신 경로)을 현재 접촉 거리만큼 늘리거나 줄이고, 이동 시간에
  속도 아이템 개수를 쓴다. **효과는 없다시피 하다**: 크기 아이템이 창고에 있는 틱은 1–2.5%(특수 아이템은 한 번에 하나, 경기는 대부분 ~50초에 끝남).
  V22에서 끄면 −0.9±0.5 (168쌍), V21-임의 각도에 이것만 켜면 +0.5±1.5.

### 결과 (보류 시드 900000–011 × 양 진영, 상대당 24경기, 짝 = 같은 상대·시드·진영, 점수차 = 우리 − 상대)
기준 = `strategic_v21_anyangle` (휴리스틱 상대: `reports/anyangle_targets`, 학생 상대: `reports/d9_vs_students` + 재실행 `base_students` 평균).
휴리스틱끼리 경기는 결정적(같은 시드 재실행 V17 4/4, V21 3/4 동일), 학생 상대는 아니다(교사 재실행 차이 +6.8±5.7).

| 상대 | V21-임의 각도 | **V22** | 짝 차이 |
|---|---|---|---|
| V4 | +59.5 | +69.6 | +10.0±4.7 |
| V17 | +36.7 | +68.7 | +32.0±10.3 |
| V19 | +40.8 | +68.7 | +27.9±8.8 |
| V21 | +25.5 | +67.5 | +42.0±11.8 |
| V21-임의 각도 | +0.0 | +60.6 | +60.6±13.1 |
| P5a (frozen p5a_v22) | −24.8 | **+54.0** | +78.8±10.2 |
| D6 (frozen d6_v42) | +19.4 | +67.5 | +48.0±11.0 |
| 전체 (168쌍) | | | **+42.8±4.1** |

- 첫 Hunter: P5a 상대 21/24 먼저(2 동시, 1 짐), D6 24/24. 학생 상대 우리 수집기 사망 27→4(P5a), 18→0.2(D6), 진치기당함 0.48→0.10.
- 5초 이상 정지(`break_ties` 효과): V19 상대 경기 비율 0.67→0.04, V17 0.71→0.12.
- 단계별(학생 48경기, 앞 단계 대비): 경주 3종(race_cargo+item_free+region, size 포함) +39.9±9.8 → +sentry +13.0±6.1 → +commit +4.3±2.1(96쌍, V19/V21-임의 각도 포함)
  → +break_ties +8.4±2.0(168쌍, V19 +22.7).
- 속도: 틱당(한 팀, 같은 관측 재생, 부하 중) V21-임의 각도 1.04 ms vs V22 1.00 ms — 차이 없음.

하나씩 끈 절제(V22 − 끈 것 = 그 스위치의 기여; 학생 48쌍, `*`는 7상대 168쌍; `reports/teacher_v2/abl_*`):

| 끈 것 | P5a | D6 | 전체 | 판단 |
|---|---|---|---|---|
| `race_cargo` | +51.4±12.3 | +33.9±11.1 | **+42.6±8.3** | 핵심 |
| `sentry` | +31.5±9.6 | +6.6±5.6 | **+19.1±5.8** | 핵심 |
| `break_ties` | +7.6±5.8 | +8.1±5.3 | +7.8±3.9 (168쌍 단계 비교 +8.4±2.0) | 유지 |
| `item_free_walkers` | +1.4±5.9 | +8.1±4.9 | +4.8±3.8 | 약함, 유지 |
| `commit` | +3.8±4.9 | +3.4±2.9 | +3.6±2.8 | 약함, 유지 |
| `region_approach`* | −6.0±3.2 | +3.8±4.0 | +1.5±1.5 (V19 +15.5, 나머지 ±4) | 유의하지 않음, 유지 |
| `size_aware`* (A) | +0.1 | +3.2±2.9 | +0.9±0.5 | 효과 없음 |

- A만 따로: V21-임의 각도 + `size_aware`만 − V21-임의 각도 = +0.5±1.5 (168쌍). B가 거의 전부다.
- 남은 약점: 경주를 같은 틱에 맞붙으면(양쪽 보행자가 겹쳐 서로 Hunter가 되며 상쇄) 지는 시드가 있다(P5a 900002_B, 900007_B).
  같은 팀 보행자가 한 점에 겹쳐 걷기 때문 — 성소 타일을 나눠 들어가기는 미시도. 경주를 진 뒤의 복구(진치기 3명 돌파)는 여전히 없다.
- 증류에 대해: 모든 규칙이 관측의 함수라 `act:strategic_v22`는 V21처럼 무상태 라벨이다. 다만 이득의 대부분이 첫 4초의 틱 단위 타이밍이라
  학생이 그 정밀도를 따라 할지가 다음 질문(P9 후속: `act:strategic_v22` 교사로 P5a 설정 증류).
## 12. P6 — 임의 각도 목표 선택 — `strategic_v21_anyangle_ed` (2026-10-02)

**질문.** `strategic_v21_anyangle`은 이동만 임의 각도이고, 목표·역할·위협을 고르는 모든 거리는 여전히 8방향(옥타일) 격자 거리다
(실제 최단 경로보다 평균 5–7%, 방향에 따라 0–8% 길다). 목표 선택도 같은 거리로 바꾸면 교사가 더 강해지나?

**V21이 거리로 고르는 곳 (전부 두 함수를 지난다).**
- `_distance_map`(한 출발점, `_cached_distance_map`·`_store_field`·`_dist`): 유닛별 경제 작업 도착 시간(`_economy_pairs`: t_pick,
  집까지 t_home, 팀원 배제 1초 마진), 저장소 비용(`_deposit_costs`)과 적의 저장소 도달 시간, 유휴 저장소 감시 점수, Hunter 추격 점수
  (`_hunt_scores`)·야영 진입(`_camp_plan`)·추격 대상 배제(도착 시간 비교), 야영 Hunter 선정.
- `TeachLabeller._distance_to`(목표 묶음): 변신 후보 순위(V21의 Hunter 없는 지도 포함), 저장소 묶음 선택, 8방향 라벨, 그리고 임의 각도
  제어기의 D(c).
- 직선 거리(`math.dist`)로 쓰는 위험·경합·야영 반경은 격자 거리가 아니어서 그대로 둠.

**거리장.** `any_angle.anyangle_distance_field`: 반지름 c(= Hunter 0.325 + 여유 0.05 + 표본 간격 절반 0.025 = **0.4**)인 점의 최단 경로.
(1) 막힌 칸의 볼록 모서리마다 c만큼 바깥에 모서리 노드(실제 지도 41개), 노드 사이 가시성은 마스크별로 캐시, (2) 질의마다 노드 위
가시성 그래프 Dijkstra(정확), (3) 칸 중심을 Dijkstra 순서로 훑는 Lazy-Theta*식 전파 — 칸은 이웃의 부모 *점*(출발점·노드·칸 중심)을
물려받고, 노드는 주변 5×5 칸에 직접 심고(검증된 씨앗은 보관), 확장할 때 시선을 검사해 실패하면 씨앗/이웃 중 최선으로 되돌린다. 격자
간선은 늘 0.5를 지키므로 도달 가능 영역은 옥타일과 같고 값은 옥타일 이하. 여러 타일 목표는 한 번의 다중 출발 전파.
c를 처음 Collector 기준 0.325로 잡았더니(v1) Hunter가 통과 못 하는 모서리 길이 D에 들어가 자기 칸 중심이 앞쪽 칸보다 늘 싸져
제자리에서 떨었다(재현 보행 49/200 실패). 가장 큰 요구치 0.4로 올리자 실패 0.

제어기도 같은 장을 쓴다: 정확한 장의 경로는 칸 중심이 아니라 모서리 노드에서 꺾이므로 `field_waypoint`가 칸 중심과 **모서리 노드**
중 |p−q|+D(q) 최소인 보이는 점으로 향한다(칸 중심만 쓰면 팽팽한 선 근처를 지그재그 — 보행 +0.45% 손해였음).

| 정확도·속도 (실제 지도 24×24, 392칸) | 임의 각도 장 | 옥타일 |
|---|---|---|
| 경로 길이 / 가시성 그래프 최적(c=0.4, 386쌍) | 평균 1.0000, 범위 0.9994–1.0000 | 평균 1.053, p90 1.079, 최대 1.104 |
| 지도 한 장 | 0.40 ms | 0.02 ms |
| 휴리스틱 틱 (기록 관측 2,537틱 재생, 5유닛) | 1.40 ms (p99 3.9) | 1.01 ms (p99 1.8) — `strategic_v21_anyangle` |
| 틱당 장 계산 | 0.72 (다중 출발로 합쳐짐) | 1.65 |
| 재현 물리 보행 시간 × 속력 / 최적 (200쌍, 짝 차이) | Collector −0.22%±0.06, Hunter −0.21%±0.08, Carrier −0.08%±0.08 (조금 빠름, 실패 0) | |

**평가** (`examples/distill_heldout_eval.py --teacher act:<id>`, 체크포인트 없음, 상대 5종, 시드 900000 12개 + 900500 12개 = 상대당
48경기 양쪽, 4 워커; `reports/anyangle_targets/`, 짝 차이 = (learner − opponent) 새 − 옛, (상대, 시드, swap)으로 짝, `paired_gap.py`):

| 상대 | ED − anyangle (48쌍) | 순위만(24쌍) | v1 c=0.325·칸 중심(24쌍) |
|---|---|---|---|
| strategic_v4 | −1.3 ± 3.8 | −2.5 ± 6.5 | −1.8 ± 4.3 |
| strategic_v17 | **+9.8 ± 6.8** | −8.3 ± 5.2 | −16.8 ± 9.1 |
| strategic_v19 | +1.0 ± 6.5 | −16.9 ± 7.9 | −26.2 ± 13.0 |
| strategic_v21 | +4.9 ± 6.6 | +0.1 ± 7.4 | +0.9 ± 11.8 |
| strategic_v21_anyangle (거울) | +2.4 ± 8.4 | −10.4 ± 8.7 | −5.8 ± 11.2 |
| **전체** | **+3.4 ± 2.9** (240쌍) | **−7.6 ± 3.2** | **−9.9 ± 4.6** |

"순위만" = `strategic_v21_anyangle_ed:field_goals=false`: 유닛별 ETA(`_distance_map`)만 임의 각도, 목표 묶음·라벨·제어기는 옥타일.

**결론.**
- 일관된 임의 각도 장(목표 선택 + 라벨 + 제어기)은 **작은 개선, 유의하지 않음**: +3.4±2.9, V17 상대 +9.8±6.8이 가장 크다. 질 곳은
  없다(최저 V4 −1.3±3.8). 교사 교체의 근거로는 약하다 — P5a 계열 증류에 쓰려면 시드를 더 늘려 확인할 것.
- 거리장을 **섞으면 해롭다**: ETA만 바꾸면 −7.6±3.2(V19 −16.9). V17 계열 가중치·임계값(점수 대역 1.12, 배제 1초, 추격 0.35 등)은
  한 거리 척도 안에서 서로 맞춰져 있어, 한쪽만 짧아지면 비교가 어긋난다.
- 제어기의 장은 가장 큰 유닛이 따라갈 수 있어야 한다(v1의 Hunter 떨림 −9.9).
- 옥타일 과대평가가 방향마다 다르긴 해도 V21의 선택은 대부분 큰 차이로 갈린다 — 같은 척도로 5% 줄여도 순위가 거의 안 바뀐다.

구현: `blackout_env/heuristics/any_angle.py`(`anyangle_distance_field`, `field_waypoint`, `AnyAngleFieldMixin(field_ranking, field_goals)`,
`StrategicHeuristicV21AnyAngleED`), 등록 id `strategic_v21_anyangle_ed`(혼합 기본 가중치 밖), 테스트 `tests/test_any_angle_field.py`
(가시성 그래프 최적과 비교, 도달·옥타일 이하·진행 이동 존재, 다중 출발, 재현 물리 보행 도착, 절제 모두 끄면 기존과 동일).
`strategic_v21_anyangle`은 동작 그대로(`_path_heading`으로 분리만).
