# V17 증류 학습·실험 계획 (2026-09-29)

파이프라인: `blackout_env/train/distill/` (구조·옵션은 `docs/offline_pretrain_runs.md` "증류 파이프라인").
교사: `StrategicHeuristicV17Teach` (`heuristics/v17_teach.py`).

## 0. 목표와 판단 기준

**질문**: MyModel이 V17의 편성(H3/Ca1/C1)과 역할별 운용을 상태만 보고 재현할 수 있는가. QMIX 경로(Run 11–16)는 V4 상대
−30 ~ −45에서 멈췄고, 원인 분해에서 편성·이동 모두 "가르치지 않은 것은 못 한다"가 나왔다. 증류는 보상 없이 교사를
복제하므로, 여기서 V17 수준에 못 닿으면 병목은 보상이 아니라 표현·용량·라벨 쪽이다.

| 기준 | 값 | 출처 |
|---|---|---|
| 교사 자신 (직접 조종) | V4 상대 +32.5, V17 상대 +24.8 (각 6경기) | `offline_pretrain_runs.md` V17-teach |
| QMIX 최고 | V4 상대 −27.7 (Run 14 80k eval), Run 16 최종 −42.8 | Run 14/16 로그 |
| **성공** | 학생 단독 V17 상대 평균 점수차 ≥ 0, V4 상대 > 0 (12시드 × 양 진영) | |
| **부분 성공** | V4 상대 > −27.7 (QMIX 최고 경신) | |
| **실패 신호** | 20k 스텝에 신규 상태 정확도 < 0.7, 또는 평가가 두 번 연속 개선 없음 | |

## 1. 기록 (재실행 없이 판단하기 위해)

모든 판단은 한 런의 기록만으로 내린다. `python examples/distill_report.py <out-dir>`가 아래 표를 뽑는다.

| 판단 | 지표 (`metrics.jsonl` kind / 키) |
|---|---|
| 암기 대 일반화 | `train`: `fresh_collect/accuracy`(새 샤드를 링에 넣기 *전* 점수), `fresh_eval/accuracy`(학생 단독 상태), `train/accuracy`와의 격차. `probe`: 고정 교사 상태 2048행, 배포마다 |
| 어느 역할·행동을 못 배우나 | 위 정확도의 `_role_{collector,hunter,carrier}`, `_kind_{battery,deposit,hunt,transform,patrol_defend,…}`, 라벨 점유율 `share_*` |
| 경기에서의 일치 | `eval`: `agree/all`, `agree/role_*`, `agree/kind_*` (워커가 실제 플레이한 모든 상태에서 학생 argmax ∈ 교사 집합) |
| 편성 | `units_{hunter,carrier,collector}`, `first_hunter{1,2,3}_s`·`first_carrier1_s`(+ `_reached_frac`) |
| 점수를 잃는 곳 | `learner/{pickups,deliveries,cargo_lost}_per_1000`, `opponent/cargo_lost_per_1000`(= 우리 처치), `approach_battery/storage`, `margin_{30,60,120}s` |
| 진동·막힘 | `reversals_per_1000/{student,teacher}`, `movement/{blocked,idle}_per_1000`, `stall/*`(코너 막힘, 가득 찬 창고 대기) |
| 로짓 건강 | `train/z`, `entropy`, `set_prob`, `logit_std`, `logit_abs_max`, `param_norm`, `param_update_norm`, `grad_norm` |
| 파이프라인 | `pipeline/{steps_per_s, matches_per_min, worker_utilization, queued_per_worker, throttled_sync_frac}`, `replay_ratio`, `ring/{mean_version_age, teacher_only_frac, kind_share_*}`, `student_version_lag`, `system/peak_rss_gb` |
| 목표치 | `eval_teacher`: 런 시작 시 교사가 같은 평가 시드로 치른 경기 (위 지표 전부) |
| 예상 못 한 질문 | 샤드에 틱별 실행 행동·학생 선택·조종 주체와 경기 요약(`meta`), `matches.jsonl`에 원시 카운트, `run_info.json`에 커밋·dirty 파일·빌드 시각 |

평가 점수차에는 `margin_se`(경기 간 표준오차)가 붙는다 — 6경기 평가의 차이를 읽을 때 기준.

**남은 사전 작업**: P3 처리량 실측(워커 4개, 10분) — `pipeline/*`가 자동으로 기록하므로 D1 첫 10분으로 대신해도 된다.

## 2. 본 학습 D1

```
caffeinate -i -m -s python -m blackout_env.train.distill.run --build build/mac/BlackOut.app \
  --out-dir checkpoints/distill/d1 --workers 4 --steps 60000 \
  --publish-every 1000 --eval-every 5 --eval-seeds 3 --replay-ratio 2 --beta-decay 0.9
```

- 기간: 60k 스텝 ≈ 7.5시간(0.45 s/스텝) + 평가. **AC 연결, 별도 세션**(배터리 수면으로 Run 15가 3시간 멈춤).
- β: v1 1.0 → v10 0.39 → v20 0.14 → v30 0.05. 30k 이후는 사실상 학생 단독 상태에 라벨.
- 체크포인트: `v_0005.pt`마다 가중치 보존 → 사후 Elo(`examples/elo_checkpoints.py`)에 쓸 수 있다.

**중간 점검** (보는 순서):

| 시점 | 볼 것 | 계속 조건 |
|---|---|---|
| 5k | `fresh_collect/accuracy_role_*`, `train/z`·`logit_abs_max` | fresh ≥ 0.6, z가 계속 커지지 않음 |
| 10k (v10 평가) | `units_hunter`, `first_hunter3_s`, `movement/blocked_per_1000`, `reversals_per_1000/student` | Hunter ≥ 2 (편성 라벨이 먹는지), 반전이 교사 기준선의 2배 이내 |
| 20k | 평가 `mean_margin`(± `margin_se`) 추세, `train/accuracy − fresh_eval/accuracy` | 격차 < 0.15 (암기 아님) |
| 60k | 최종 평가 (§4) | — |

## 3. 실험 (D1 결과로 필요한 것만, 각 20k 스텝 ≈ 2.5시간)

우선순위 순. 한 번에 한 변수만 바꾸고 D1의 20k 체크포인트와 비교한다.

| # | 변경 | 가설 | 언제 돌리나 |
|---|---|---|---|
| E1 | `--class-weighting balanced` | V17 편성에서 라벨의 ~60%가 Hunter라 Collector·Carrier 정확도가 뒤처진다 | D1에서 `fresh_eval/accuracy_role_collector` 또는 `_carrier`가 hunter보다 0.1 이상 낮을 때 |
| E2 | `--beta-decay 1.0` (교사 상태만 = 순수 BC) | DAgger가 공변량 이동을 줄인다 — Run 11 진단(학생이 만든 상태에서 못 줍는다)의 직접 검증 | 항상 (DAgger 효과의 대조군, 결과 해석의 기준) |
| E3 | `--replay-ratio 1` / `4` | 1: 더 신선한 상태, 4: 같은 데이터로 더 많은 학습 | `fresh_*`가 `train/accuracy`보다 크게 낮으면 1, 둘이 같이 낮으면(과소적합) 4. `pipeline/throttled_sync_frac`가 0에 가까우면 비율이 아니라 수집이 병목 |
| E4 | `--opponents strategic_v17 mixture` | 강한 상대 앞에서의 상태(쫓기는 짐꾼, 진치기 당함)가 mixture에 적다 | V17 평가만 유독 나쁠 때 |
| E5 | `--teacher v4-teach` | 학생이 V17의 복잡한 역할을 못 배우면 더 단순한 교사가 실제로 더 강한 학생을 만든다 | D1이 부분 성공 이하일 때 |

E2는 결과와 관계없이 돌린다 — DAgger가 효과 없다면 파이프라인의 절반(학생 수집)이 불필요하다는 뜻이기 때문이다.

## 4. 최종 평가 (D1 60k 및 채택된 실험)

온라인 평가(6경기)는 추세용이다. 최종 판단은 아래 별도 평가로 한다.

1. **점수차**: V4·V17·V19 상대 12시드 × 양 진영, 헤드리스(`examples/evaluate_checkpoint_vs_heuristic.py`).
2. **편성**: 경기당 평균 H/Ca/C를 B0(교사)와 비교. 편성은 같은데 점수가 낮으면 역할 운용, 편성부터 다르면 변신 라벨 문제.
3. **역할별 분해**: 배달·도난·처치(`train/objective_monitor.py` 지표)를 교사와 나란히 — 어느 역할이 점수차를 만드는지.
4. **이동 품질**: 막힘/정지(`movement_monitor`), 벽 옆 반전(`examples/probe_boundary_chatter.py`).
5. 필요하면 GUI 1–2경기로 눈 확인(기본은 헤드리스).

## 5. 이후 (D1 결과에 따라)

- **성공/부분 성공** → RL 미세조정: `qmix_trainer`가 `policy_state`만으로 시작하는 `--init-policy`가 필요하다(지금은
  믹서·SPR 상태를 요구). 증류된 로짓은 가치가 아니므로 Q 헤드 재보정 구간(짧은 TD 워밍업, BC/KL 정규화로 증류 정책
  근처 유지)을 둔다.
- **실패** → 먼저 교사 라벨이 학생 입력만으로 결정 가능한지 확인: 같은 관측 → 같은 라벨인데 틀리면 용량·표현,
  관측에 없는 정보(예: 진치기 Hunter 선정의 경로 거리)에 의존하면 라벨 설계를 고친다.

## 6. 진행 상황과 다음 단계 (2026-09-30)

| 런 | 교사 | 링 | 보류 평가 최종 (V4 / V17 / V19) |
|---|---|---|---|
| D1 | v17-teach | 60만 | +58.5 / +15.4 / −39.5 |
| D2 | v17-teach, 상대 다양화 | 60만 | +54.8 / +1.6 / −37.0 |
| D3b | v20-teach (스폰 진치기 대응) | 120만 | **+60.4 / +34.2 / −21.8** |
| D4 | v21-teach (안전한 변신만) | 120만 | +44.8 / +25.6 / **+16.3** |

교사 자체(직접 조종): v17-teach +49.8/+17.8/−38.9, v20 +48.5/+14.7/−23.3, **v21 +49.7/+12.8/+3.4**.

D4 결과(2026-09-30): D4 − D3b 짝 비교 V4 −15.6±8.7, V17 −8.5±11.5, **V19 +38.1±7.1**. 처음으로 V19를 이긴 학생(승률
0.62)이고 교사 v21보다도 V19에서 +12.9±9.7. 대신 V4에서 D3b보다 약하다(교사 v21 대비 −5.0±9.5라 교사 수준).

**D4** — `checkpoints/distill/launch_d4.sh`(분리 실행, AC 필요). D3b와 교사만 다르다(v21-teach), 링 120만, 워커 6, 60k.
보류 평가가 종료 후 자동으로 이어진다. OOD용으로 Run 16 체크포인트(`20260919-010456_42178`, 41개)를 상대 풀에서 뺀다.

### OOD 테스트 (D4 이후)

학습 분포 밖에서도 교사만큼 하는지. 보류 시드 × 양 진영, 상대당 24경기, 학생과 **교사(v21) 모두** 같은 상대로 —
판단은 "OOD에서 학생 − 교사 격차가 분포 안(V4/V17/V19)의 격차보다 벌어지는가".

| OOD 축 | 상대 | 왜 OOD인가 |
|---|---|---|
| 처음 보는 학습 정책 | Run 16 체크포인트(final, 중간 몇 개) | D4 상대 풀에서 제외 |
| 처음 보는 휴리스틱 | `strategic_v20`, `strategic_v21` | mixture 기본 가중치에 없음 |
| 섭동 범위 밖 | V17/V19에 Hunter 4기, `hunter_mode="camp"`, `steal_weight` 3.0, `exposure_cost` 0.3 | mixture 섭동은 Hunter 2–4, mixed/hunt, 약탈 1.3–2.3, 노출 0.6–1.2 |
| 비정상 상대 | 무작위 행동, 제자리 정지 | 어떤 휴리스틱도 이렇게 두지 않음 |
| 거울전 | 학생 대 학생, 학생 대 교사 | 학생이 만든 상태가 양쪽에 |

볼 것: 점수차(학생 − 교사, 짝), 교사 일치율(`agree/*`, 역할·라벨 종류별), 트랩(`distill_report` 반전 표 — `--save-shards`
필요), 사망·출구 봉쇄.

구현: `distill_heldout_eval --opponents ood`가 위 표를 펼친다(`OOD_OPPONENTS` — Run 16 final/100k/50k, V20, V21,
`strategic_v19:hunter_quota=4,hunter_mode=camp`, `strategic_v17:steal_weight=3.0,exposure_cost=0.3`, `random`, `idle`,
`self`, `teacher:<교사>`). 서술자는 `opponents.py`(`key=value` 인자, `random`은 틱당 10% 확률로 방향 전환, `self`는
학습자 자신). `summary.txt` 하단에 짝 격차(후보 − 교사, 같은 시드·진영, 시드 단위 ±se)가 붙는다. `self` 거울전은 점수차가
대칭 잡음이라 격차에서 빼고 일치율·사망·트랩만 본다. `teacher:<교사>` 행의 격차가 곧 학생 대 교사 직접 대결 점수차.
실행: `checkpoints/distill/launch_ood_d4.sh`(gitignore 경로) — D4 보류 평가가 끝나면 D4·D3b latest를 `reports/distill_ood/`에.

**OOD 결과 (2026-09-30, `reports/distill_ood/`, 상대당 24경기, 짝 격차 = 후보 − 교사 v21, ±se)**

| 상대 | 교사 v21 | D4 | D4 격차 | D3b 격차 |
|---|---|---|---|---|
| Run 16 final / 100k / 50k | +58.7 / +56.4 / +70.5 | +65.2 / +67.6 / +75.9 | +6.5±6.6 / +11.2±7.7 / +5.3±4.0 | +10.8 / +15.2 / +2.7 |
| V20 / V21 | −10.7 / 0.0 | +17.8 / −3.8 | +28.5±10.4 / −3.8±9.0 | +29.2 / +22.4 |
| V19 Hunter 4·camp | +3.4 | +16.5 | +13.1±10.5 | **−19.3±10.0** |
| V17 steal 3.0·exposure 0.3 | +6.2 | +1.4 | −4.8±8.9 | +25.0±9.6 |
| random / idle | +79.6 / +75.0 | +92.1 / +96.5 | +12.5±4.1 / +21.5±3.3 | +13.2 / +21.8 |
| 교사 v21 직접 대결 | 0 | **+16.0±9.6** | | +33.2±7.3 |

분포 안 격차(D4: V4 −5.0, V17 +12.8, V19 +12.9)와 비교해 **OOD에서 벌어진 곳이 없다** — 전 축에서 학생 ≥ 교사 또는 잡음 안.
합성 상태 보강은 지금은 불필요. D3b는 V19 camp 4기에서 −19.3(교사 v20로 배운 탓, v21 교사가 막는 상황), D4는 +13.1.
남은 약점은 점수가 아니라 **흔들림**: 학생 반전/1k가 교사의 3–30배(idle 상대 455 vs 1, 체크포인트 상대 260–420 vs 80–130),
체크포인트 상대 사망/1k 1.3–2.2 vs 교사 0.7–1.1. 일치율은 0.84–0.92로 분포 안과 같다.

### OOD 대응이 부족하면 — 합성 상태에 교사 라벨

교사는 기억 없는 관측→라벨 함수라 실제로 방문하지 않은 상태에도 라벨을 붙일 수 있다(Unity를 그 상태로 되돌릴 필요 없이
한 스텝 라벨만으로 BC가 된다). 두 단계로:

1. **구조적 섭동** (먼저): 실제 샤드 상태를 규칙으로 흔든다 — 유닛을 다른 걸을 수 있는 칸으로, 클래스·짐 변경, 배터리
   추가·제거(15분의 1 단위), 적 위치·편성 변경, 시간·점수·흡수 타이머 변경. 맵은 실제 것이라 관측이 항상 유효하다
   (벽·본진 진입 금지·창고 용량·원-핫 제약을 규칙이 지킨다).
2. **생성 모델** (1로 부족할 때): 샤드 상태로 VAE/조건부 생성 모델을 학습해 맵은 실제로 두고 유닛·아이템 배치를 생성,
   유효성 검사를 통과한 것만 라벨. 이산 채널·원-핫·15분의 1 단위라 연속 출력을 투영해야 하고 불가능한 상태가 섞이기 쉬워
   2순위.

공통: 합성 행은 배치의 10–20%로 제한(실제로 안 일어나는 상태에 용량을 쓰지 않게), 합성 상태에서 교사 라벨의 무효 비율과
흔들림을 따로 기록, 효과는 OOD 테스트의 학생 − 교사 격차가 줄어드는지로 판단.

## 위험

| 위험 | 대응 |
|---|---|
| Q 헤드를 로짓으로 써서 크기 드리프트 | z-loss 1e-4, `train/z` 감시. 커지면 1e-3 |
| 교사의 "정지"(상호작용 중 무효 라벨)를 학생은 표현 못 함 | 라벨에서 제외돼 학습에 무영향. 평가에서 픽업 실패율로 확인 |
| IQN τ 샘플링 잡음이 추론 행동을 흔듦 | Run 11에서 argmax 3.2% 변경으로 부차적. 필요하면 평가 시 고정 τ |
| 60만 행 링 ≈ 5GB RAM | 메모리 압박 시 `--max-rows` 축소 |
