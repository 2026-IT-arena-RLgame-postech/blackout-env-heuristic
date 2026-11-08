# 밸런스 패치 휴리스틱 (2026-10-31~)

패치: 팀당 살아 있는 헌터 최대 2, 벽 (5,18)(6,18) + 대칭 (18,5)(18,6) 추가, (1,16) + 대칭 (16,1) 개방.
시뮬은 패치된 Unity와 이벤트 단위로 일치 (`reports/sim/validate_bp.log`, 16경기).

## 패치 후 경기 양상

- 경기는 첫 20 s에 결정된다 (20 s 점수와 최종 점수차 상관 0.91). 20 s 내 헌터 킬 +0.81, 콜렉터 사망 −0.81.
- 오프닝: 양 팀 경주 워커 2명이 같은 틱(~4.0 s)에 성소에 들어가 헌터 2:2 맞교환. 다음 헌터를 먼저 만든 쪽이
  성소를 차지하고 20–30 s 동안 독점한다 (상대 기지 출구 (4,17)–(4,18) 봉쇄 + 성소 보초).
- 독점당한 쪽 콜렉터는 기지 출구에서 죽는다. 새 벽 (5,18)(6,18)이 남동쪽 출구를 막아 출구가 한 곳으로 몰렸다.

## 진영 쏠림과 cap_sync (V34)

- 패치 후 같은 휴리스틱끼리도 팀 B가 크게 유리했다 (V34+예비2 자가대전: A −13 / B +13; 패치 전에는 쏠림 없음).
- 원인: 예비 워커가 경주 워커와 같은 틱에 성소에 들어가면 헌터 상한(경주 워커가 아직 살아 있음)으로 변신이
  거부되고, 변신은 *진입할 때만* 일어나므로 성소 안에 콜렉터로 서 있다가 죽는다. 팀 A 유닛이 먼저 처리되어
  A에서만 주로 일어났다.
- `cap_sync`: 결정마다 성소에 들어갈 수 있는 우리 콜렉터를 (2 − 우리 헌터 수)명으로 제한하고 나머지는 대기,
  안에서 거부된 콜렉터는 밖으로 나갔다가 재진입. 겹친 유닛(같은 좌표)을 좌표별 개수로 처리해야 한다.

## 실험 (V33bp 또는 V34 기준, 32 seeds × 2면, paired)

| 변경 | 결과 |
|---|---|
| reserve_count 1 → 2 | +9.9 / +11.4 (재현) |
| reserve_count 3 | = 2 (세 번째 후보 없음) |
| carrier 없음 | −20 |
| carrier_delay 6–20 s | 0 ~ −6 |
| hunter_quota 1 | −34 |
| **cap_sync (cap_enter 0.2)** | **+10.2** (A −13→−2, B +13→+22) → V34a |
| storm (독점 시 콜렉터 커밋 돌진) 1–3명 | −4.5 ~ −8 (기지 출구에서 죽음) |
| breakout_units 2–4 | ±1 |
| dom_threat_radius 0.25 / 0.35 | −5 / −12 |
| CEM g0 (V33bp 주변 40개) | 24 seeds 상위 +7 → 64 seeds 확인 0 (승자의 저주) |

## 현재 챔피언

`V34a` = V33bp + `reserve_count=2,cap_sync=true,cap_enter=0.2` (`strategic_v34`). 루프(`scripts/bp_loop.py`)가 이를
시드로 개선/파훼 반복 중.

## V34b (2026-10-31 03:00)

- 원인 1: 예비 워커가 오프닝에 배터리를 줍고 입금하느라 두 번째 레이스에 늦는다 (시드의 배터리 배치가 레이스를 결정).
- 원인 2: 예비가 성소 경계 0.2–0.55 칸 밖에서 대기해, 맞교환 후 진입까지 2–4 결정이 걸린다. 상대 예비가 한 결정 먼저 들어오면 진다.
- `early_reserve=5`: 5 s 전까지 예비 2명이 처음부터 성소로 직행 (짐을 들어도 유지). `reserve_stage=0.17`, `cap_enter=0.17`
  (한 결정 이동량 0.16 칸): 경계 바로 앞에서 대기 → 맞교환 다음 결정에 진입. 셋이 함께여야 효과 (early_reserve만: −45).
- 확인 (새 64 seeds × 2면): V34b − V34a +22.6±3.6 (승률 .71), V34b − V33bp +34.2±2.7 (.87). V34b 자가대전 A −3.2 / B +3.2
  (V34a는 −15.8 / +15.8) → 진영 쏠림 거의 해소.
- 효과 없음: 커밋 범위 확대(commit_margin −0.6/−1.0, commit_reach), race_second(상대 먼저 변신시키기, −26).
- 구현 주의: 겹친 유닛은 좌표가 같으므로 좌표로 유닛을 구분하면 안 된다 (cap_sync는 `teach()`에서 유닛 이름으로 덮어쓴다).

## 04:10 추가 실험 (V34b 기준, 48 seeds × 2면)

| 변경 | 결과 |
|---|---|
| carrier_retreat 3 / 5 / 8 tiles (헌터 접근 시 캐리어 귀가) | −8.9 / −24.0 / −24.8 (수집 시간 손실이 더 큼) |
| w_dspd, w_bspd ×2 / ×4 / ×0.5 | +0.3 / −0.4 / −1.7 (무효) |
| CEM g1 (V34b 주변 40개, vs V34a+V34b) | 64 seeds 확인 최고 +0.9±2.2 (무효) |

- V34b 자가대전: 헌터 전쟁 영향 감소 (w20 hun_s 상관 0.58), 20 s 첫 흡수량 상관 0.82, 캐리어 사망 −0.42.
  캐리어는 새 통로 (1,14)–(1,18)에서 주로 헌터에게 죽는다 (팀·경기당 1.2회).
- 첫 특수 아이템(~10 s)을 먼저 가진 쪽 평균 +21 (가중치로는 바뀌지 않음 → 위치/유닛 배정 문제).
- 세 번째 레이스(~9 s, 부활 유닛 재경주)는 6–7 s 살아 있는 상대 헌터 회피 경로 차이로 1.5칸 앞뒤가 갈린다.

## Elo (scripts/bp_league.py, reports/bp/league)

V34b +171±16, V34a +18±14, V33bp −189±19.

## 보류 가설 (폐기 아님: 큰 줄기 변경은 기존 노브가 옛 국소 최적에 맞춰져 있어 단순 비교로 손해로 보일 수 있음 → 주변 재튜닝 후 재실험)

| 가설 | 단순 비교 결과 | 재시도 조건 / 아이디어 |
|---|---|---|
| carrier_delay (캐리어를 늦게) | 0 ~ −6 | 헌터 전쟁 인원 +1이 효과를 낼 구조(예: 예비 3명)와 함께 |
| storm (독점 시 커밋 돌진) | −4 ~ −8 | 출구 선택(camp_path, 동쪽 출구)과 결합, 동시 출발 |
| race_second (상대 먼저 변신시키기) | −26 | cap_sync 세부 타이밍 재설계 후 |
| carrier_retreat | −9 ~ −25 | 반경을 작게(1.5–2) + 기지 대신 가까운 안전 지점 |
| camp_path (경로 거리 위험 지역) | +1.4±1.5 | 출구별 목표 재선택과 함께 |
| hunter_quota 1 / reserve_count 1 (경제 우선) | −34 | 습격형 헌터 배치와 결합 |

## 헌터 역할 (05:00, 사용자 아이디어) → V34c

`hunter_roles` (V34): default(상속 동작: 성소 보초, 스폰 봉쇄, 추격) / raid(상대 콜렉터·캐리어 추격, 짐 든 유닛 우선) /
south·east(상대 기지 남쪽 y=17·동쪽 통로 출구 지키기) / storage(상대 저장소 매복) / field(배터리 밀집 지역).
`roles_when`: free(상대 헌터가 기지 밖에 없을 때만) / always.

x18 (48 seeds × 2면, paired):

| 역할 | vs V34b | vs V34a |
|---|---|---|
| **raid/raid, always** | **+27.0±3.6 (승률 .78)** | +26.5±4.3 |
| south/raid | +9.5 | +8.4 |
| raid/raid (free) | +8.0 | +14.3 |
| default/raid | +6.8 | +10.6 |
| south/south | +4.5 | +3.0 |
| south/east | −0.1 | −15.2 |
| east/raid | −3.2 | −26.2 |
| east_post (단독, 앞 실험) | −1 ~ +2 | −6 ~ −11 |

→ 봉쇄는 약하고, 배터리 수급 습격이 강하다. 상대 헌터가 있어도 습격을 유지(always)하는 것이 결정적.
**V34c** = V34b + `hunter_roles=raid/raid,roles_when=always`. 루프 풀 {V34a, V34b, V34c}, 챔피언 V34c (g003부터).

## V34c 주변 (05:10–06:05)

- x19 (48 seeds): raid_value 1 / 4, raid/field, raid/storage, raid/default → V34c 대비 −0 ~ −11, storage/storage −55. V34c 유지.
- x20 보류 가설 재시험 (V34c 기준): storm +4.1±5 (64 seeds 재확인 x21: −1.8±2.2), camp_path +1.3, h_evade 2 −1.7,
  carrier_retreat 2 −8.6, carrier_delay 8 −14.9, early_reserve 0 −31, reserve_count 1 −65, hunter_quota 1 −72.
  → 습격형에서도 레이스 구조(예비 2 + early_reserve)는 필수. 보류 가설은 여전히 무효.
- V34c vs V34b 경기 분석: 헌터-초는 V34c가 적지만(1291 vs 1858) 상대 캐리어 킬이 2.3배 (165 vs 73).
- x21 raid_carrier 0.5 / 2 / 3: +5.3±2.9 / −3.7 / −3.5 (vs V34c), vs V34b −0.5 / −2.8 / −3.8 → 무효.
- 속도: 서브에이전트 #1 (perf/bp-heur-speed 병합 71f480e) 결정당 10–17 % 단축, 9 조합 해시 동일
  (`examples/heur_perf_hash.py --check examples/heur_perf_hash_base.json`).

## 06:25–12:40

- x22 raid_lead 0.5 / 1.0 (가로채기 추격): −4.1 / −2.9 (vs V34c) → 무효. x23 escort(캐리어 호위) 조합: −15 ~ −73 → 보류.
- 루프 (V34c 기준 g003–g013): improve 확인 대부분 0~+3, **g6 NEW CHAMPION B3** (+4.9±2.1 over pool; V34c + CEM 노브 다수,
  `h_trade_up=true`, `h_evade_radius=1`, `dom_threat_radius=0.1`, `breakout_units=2`, `opening_seconds≈15`). g5/g9 파훼 실패,
  **g12 COUNTER C0** (B3 상대 +14.5; `reserve_stage≈0`, `cap_enter 0.142`, `solo_sentry`, `hunt_threshold 0.15`, `h_trade_up`) → 풀 합류.
- 서브에이전트 속도 최적화 #2 시작 (12:40, branch perf/bp-heur-speed2).

## 14:15 (세션 압축 전)

- 풀 리그 (league2, 32 seeds × 2면): B3 +115, C0 +94, V34c +70, V34b −78, V34a −201 (Elo). B3 − C0 +2.4 → C0의 +14.5는 승자의 저주.
- f1race (B3 주변 레이스 노브 24개, 96 seeds 확인): 최고 +0.7±1.3 → B3가 국소 최적.
- `race_search` (V34, d9dca78): 관측 → SimState.from_obs → 성소 근처 우리 콜렉터(≤4)의 진입 지연 0..k 조합을 10결정 굴림
  (상대 모델 2: 직진 / 헌터 상한 존중), (우리 헌터 − 상대 헌터)×10 − 잃은 콜렉터 최대 조합 실행. 탐색 1회 ~50 ms.
  f2rs 평가 중 (B3 + race_search 1/2/2,h15/3 vs B3, C0, V34c).

## 15:00 race_search 결과 → 보류

- f2rs (48 seeds × 3 상대): race_search=1 −1.9±3.5, =2 −31.8, =2,h15 −38.1, =3 −42.2. 진입 지연 탐색은 손해 (지연이 곧 레이스 패배).
  보류 목록에 추가: 상대 모델이 실제 휴리스틱과 다름 → 상대를 실제 B3 정책으로 굴리는 버전이면 재시도 가능.
- 루프 g15 파훼 단계 진행 중 (screen 최고 +1.3).
- `examples/gui_match.py`가 `이름=descriptor`(별칭 포함)를 받도록 수정 → 사용자 GUI 관전용.

## 16:00–16:50 사용자 GUI 관전 아이디어 4개 (순서대로 검증)

1. **오프닝 멈춤 → 먼저 변신**: 원인 = V27 예비 대기가 좌표로 유닛을 구분 → 같은 칸에 겹친 경주 워커까지 성소 가장자리에서
   멈춤 (4.0 s 도착, early_reserve 끝나는 5.0 s까지 양 팀 모두 대기 후 동시 진입). `stage_by_name` (이름으로 대기) → 경주 워커는 4.0 s에
   바로 진입. f3sn 평가 중 (race_search=1 재확인 포함; f2rs 확인 단계에서 +6.5±2.5였음).
2. **스폰 회피** (`gate_split`): 기지 출구(남쪽 y=18 줄 / 동쪽 x=5 통로)별로 (상대 헌터 도착 시간 − 우리 도착 시간) 여유를 계산해
   가장 안전한 출구로, 같이 나가는 유닛은 출구 분산, 모두 위험하면 기지 안 대기(헌터 진입 불가), 출구는 기지 밖 4칸까지 유지.
   - 사실: B3 자가대전 90 s 동안 팀당 콜렉터·캐리어 41.9회 사망, 대부분 기지 나온 지 1 s 내, 헌터 1마리가 남/서 출구를 막음.
     부활은 즉시(빈손 사망 손해 작음) → 손해는 수집 시간.
   - gate_split 궤적(64경기): 기지 근처 사망 41.9 → 26.0, 90 s 점수차 +2.3. f4gate 평가 중.
3. **헌터 목표 진동** (`raid_idle=gate`: 먹잇감 없으면 상대 기지 출구 하나씩 맡아 대기, `raid_sticky`: 목표 유지 이력). f5rb 평가 중.
4. **캐리어 미끼** (`bait`: 헌터가 캐리어 N칸 안이면 가까운 빈 콜렉터가 헌터에게 돌진). f5rb 평가 중.

- 자원 사고: bp_focus/bp_loop가 동시에 돌 때 각자 호스트 한도를 다 써서 4배 과다 (gpu1 64 프로세스). `host_budget`로 수정
  (우리 다른 작업의 워커 수를 빼고 배치 시작). 루프는 g016에서 수동 정지 (최근 세대 이득 ≈0).

## 17:00–18:00 확인 단계 + 증류 시작

- 96시드 확인 (B3 대비): gate_breakout=1 −0.2±2.5, raid_sticky=0.7 −3.5±1.4, stage_by_name+race_search=1 +1.7±2.3,
  G(gate_split,breakout3,tol0.8) 부분 24시드 −0.5±5.6 → 1차 화면의 +3~+7은 기준 B3의 공통 잡음(같은 시드)으로 보임.
  다만 궤적상 gate_split은 기지 근처 사망 42→26, 캐리어 사망 11.1→5.0 → 새 시드 조합 평가 f6comb (64→128 seeds)로 판정.
- bait −6~−9 (보류), raid_idle ≈0, stage_by_name ≈0 (상대가 같은 결정에 반응 진입: 마지막 걸음 진입 규칙).
- `gate_return` 추가 (기지로 돌아올 때도 안전한 출구로).
- **증류 D1 시작 (18:00, wsl_gpu)**: 교사 act:B3, init p24c_v321, P24c 레시피 20k steps, 상대 = 패치 후 풀 + B3 섭동 + 옛 학생 + self
  (`reports/bp/distill/launch.sh`, 출력 wsl ~/bpruns/d1_b3). wsl은 평가 호스트에서 제외.

## 18:00–19:10

- f6comb (새 시드 64): G(gate_split) −2.8, +gate_return −5.9/−6.3, G+race_search +0.6 → gate_split·gate_return 보류.
- f7rs: B3+race_search=1 192 seeds +0.4±1.6, 96 seeds +0.7±2.3 → 무효, 보류. **B3 유지.**
- 증류: Unity D1 (B3 교사) 4800 steps에서 중단 (워커 CPU 병목: 학생 CPU forward + 교사, 20k ≈ 3 h).
  `run.py --sim` (sim_collect.py: BatchSim 32경기 묶음, 같은 shard 형식) 추가 → D2 = D1 s4800에서 이어서 sim 수집.
  GPU 학생 forward는 학습과 GPU 경합 (190 rows/s = Unity와 같음) → 워커 CPU 묶음 forward로 재시작.
- 시뮬 패치 반영 재확인: MAX_HUNTERS=2, 맵 6칸, V33(헌터 4 목표)도 시뮬에서 최대 2마리.
- 루프 재시작 (19:10, g016부터).

## 23:00–00:30 대회 형식 후속 (공식 env b5743f7: frame_stack + state_in/state_out 지원)

- 공식 = 50 Hz (DecisionPeriod 1), 포크/시뮬 = 25 Hz. sim `ticks_per_decision=1` 추가. 혼합 속도 시험 (sim, paired 아님):
  B3@50 vs B3@25 −3.4±7.0 (64), D3@50 vs D3@25 −4.1±8.3 (48), D3@50 vs B3@50 −4.5±10 (32), D3@25 vs B3@25 −1.5±6.8 (64)
  → 50 Hz 추론의 뚜렷한 손해 없음 (성소 레이스 첫 헌터 틱도 동일) → 학습은 25 Hz 유지 (사용자 지시).
- 적 클래스 추적기 (official/obs_mask.py track_classes, ONNX state용): 시뮬 규칙 (성소 진입 타일 변경, 중간 틱, 헌터 상한,
  캐리어 잠금, 순간이동=사망, 에이전트 순서 3,6,2,7,5,8,4,9,1,0) → 10만 건 오류 0.
- 배터리 양 가림 (B3, 모든 양=5): −54.7±4.6 (48 seeds) — 휴리스틱 계산이 깨지는 영향 포함. 부분별 (바닥/저장소/들고 있는 양) 측정 중.
- 시드 크랙 (official/seed_crack.py): 첫 프레임 (빠진 저장소 후보 + 배터리 위치 집합)으로 후보 시드 검사, 배터리 12개 대조면 유일.
  torch GPU(gpu2) 단계별 걸러내기 ~0.2 G seeds/s. **ORT 1.22 CUDA EP에 BitShift/BitwiseXor/BitwiseAnd 없음 (CPU로 넘어감)** →
  GPU에서는 곱/나눗셈 + 바이트 XOR 조회표로 대체 필요 (추정 ~0.15 G seeds/s, 2^32 ≈ 30 s를 스텝당 ~20 ms로 분할).

## 01:00–02:30 정보 복원 (ONNX 상태)

- 배터리 양 부분별 가치 (B3, 32 seeds): 바닥 −0.7±7.8 (무관), 저장소 −55.4±5.4, 들고 있는 양 −13.1±5.5 → **시드 크랙 불필요**, 저장소 추적이 핵심.
- 점수 규칙 확인: 저장소에 들어가는 즉시 점수 +, 도난 시 −, 흡수는 점수 불변 (확정) + 저장소 비움.
- 저장소 추적기 `track_storage`: 점수 변화 + 도난(배터리 픽셀 소실) + 내려놓은 유닛의 저장소 + 고정 채움 순서(스폰에서 먼 칸부터, 칸당 10, 특수 아이템 칸 건너뜀)
  + 흡수(시간으로 계산) + 팀 총량(점수 − 마지막 흡수 때 점수) 보정. 50 Hz 시뮬 11경기 약 7만 프레임 칸별 오류 0.
  동시 입금(다른 저장소)은 16경기 484 입금 중 0회.
- 마스크 모드 3 = 바닥·들고 있는 양만 가림 (적 클래스·저장소 양은 추적). ONNX 상태 1231 floats, ORT GPU(2080 Ti) 스텝 15.6 ms, CPU 1.9 ms.
  D3 기반 모드 3 그래프 vs 시뮬 학생 50 Hz 일치 99.8 %.
- 증류: D5 (모드 2) → **D6 (모드 3, gpu2, ~/bpruns/d6_store)**.
- 첫 스텝 30 s 허용은 시리즈 전체의 첫 호출에만 (remote_model stats.steps == 0).

## 02:30–03:30 공식 v1.0.2 + 대회 실행기 검증

- v1.0.2 (23:11 배포): 헌터 팀당 2 (MaxUnitsPerTeam, 변신한 유닛이 클래스가 바뀌면 자리 반환), 벽/개방 위치 = 우리 구현과 동일.
- 대회 실행기 (Docker + MPS) gpu2 GPU1에서 실행 (`scripts/comp/run_official.py`: GPU 번호만 바꿈, COMP_NO_RENDER로 리플레이만 저장).
  `--no-gpu`는 공식 정책 서버가 CUDA 세션에서 세그폴트 (gpu1) → GPU 경로 사용.
  D3 모드3 vs D2 모드1: 8경기 전승 (101~104 : 15~20), 시간 초과 0, 최대 지연 59 ms.
- **시뮬 ≠ 공식 원인: 벽 충돌**. 우리 포크는 x/y 분리 클램프(코너 교착 수정), 공식은 대각 변위 전체를 한 번에 클램프.
  sim `core.OFFICIAL_CLAMP=True`로 공식 방식 채택 → 공식 리플레이 재생 시 위치 전 스텝 일치, 점수 5/6 일치
  (불일치 1개 = 시리즈 첫 경기: 첫 에피소드 맵이 시드를 안 따름).
- **numba 캐시 주의**: batch.py의 캐시가 core.py 변경을 추적 안 함 → core 변경 후 `find blackout_env -name '*.nb[ic]' -delete` (모든 호스트).
- 증류: D6 (모드3, 포크 충돌) 10k → **D7 (모드3, 공식 충돌) gpu2 ~/bpruns/d7_off** 시작.

## 03:30–04:00 (세션 압축 전 현황)

- D7 (모드 3, 공식 충돌) gpu2 GPU0 진행 중 (~/bpruns/d7_off, 30k steps, ~17:05 UTC 완료 예상).
- 교사 개선 서브에이전트: worktree ../blackout-env-teach (branch feat/bp-teacher), mac+gpu1 CPU, 새 별칭(B4) 보고 예정.
  공식 충돌에서 B3 vs V17 +46.8±10.1 (16), 2 s 이상 끼임 0.
- RL 준비: reports/bp/rl/ppo_bp1.sh (INIT=d7_final, GPU=2,0 비동기 2-GPU), 별칭 B3/C0 추가, ppo_league_b.sh에 EVAL_OPPS.
## 2026-11-01 official-collision teacher round (feat/bp-teacher)

- rr0 (48 seeds, official clamp): C0 +40.6 avg, B3 -4.9, V34c -6.0, V34b -29.7; C0 beats B3 by +36 (tie-order exploit below).
- Wall slowdown is a non-issue (0.06 % of command steps in the first 10 s; later only idle units pressing base corners).
- Agent update order (3,6,2,7,5,8,4,9,1,0) decides ties: B's racers 6,7 act before A's 1,2, so a hunter one tick earlier
  wipes the stacked collectors at the entrance. `race_order=true` picks Hunter walkers/reserves by that order (A: units 3,2).
  C0R = C0+race_order: vs B3 +76.7, V34c +73.4, V34b +77.0, C0 -1.3+-4.0 (64) / -4.9+-3.1 (96). B3R: +12.6 vs C0, loses to C0R -28.7.
- race_beam (beam-searched open-loop race walk, arrives 1 tick earlier at 98): ~0 vs C0, much worse vs B3 (no wipe) -> off.
- CEM around C0R (race,hunters,evasion): best confirmed +2.6+-1.4 over pool: noise level.

## 2026-11-01 RL(rl_bp1_final) 우위 출처 분석 (feat/bp-teacher)

- RL vs B3 / B4 / C0 (48 seeds × 2면, seed0 8.1M, v27_eval cap 150 s): B3 상대 A면 +34.5±9.5 (승률 .69), B면 +69.8±3.6 (.98);
  B4 상대 −24.9±9.4 / −22.9±9.3; C0 상대 −28.6±8.9 / −13.8±8.9. 같은 시드에서 B4 vs B3 = +72.9±4.2 (A) / +72.9±2.6 (B), B3R vs B3 −12.6 / +24.8, B3 vs B3 −24.7 / +24.7 (면 비대칭).
- 결론: RL의 B3 대비 우위 = 성소 레이스 동시틱 와이프(race_order와 같은 트릭). RL은 100틱에 헌터 2.0 vs B3 0.0, 입구 와이프로 B3 콜렉터 3.3마리 vs 0.2 (B4: 4.0 vs 0).
  마진 증가 구간 (궤적 24시드×2면): RL vs B3 110–400틱 +27.7, 400–800 +12.6 (합 ≈ +43) / B4 vs B3 +56.2, +11.9 (합 ≈ +68). RL은 B4의 열화판
  (A면에서 31 %를 놓침, 와이프 후 헌터 1.58 vs 2.0 유지). RL vs B4는 와이프 쌍방(헌터 3.3마리씩 사망) 후 B4가 2차 헌터를 더 빨리 확보
  (300틱 헌터 1.33 vs 0.85), 110–800틱에서 RL이 −23 손해. 따라서 B4에 없는 RL 고유 우위는 못 찾음.
- 후반 (800틱+, 생존 경기만): RL 헌터 킬/헌터초 0.65 vs B4 0.58 (+12 %), RL 헌터는 항상 이동, 그러나 선택편향 + 마진 영향 +1~2에 그침.
- 기존 노브 손시험 (B4 기준 32 seeds × {B4,C0,B3R,V34c}): solo_sentry=false −8/−16/0/0, late_hunter_quota=1 −8/−7/−4/−2, raid_sticky 0.8 ≈0, hunter_quota=1 −70;
  sweep_camper, late_hunter_quota=2, hunt_threshold, carrier_on_monopoly, late_after는 B4에서 무효과(동일 결과). 새 노브/별칭 없음 → B4 유지.
- 도구: reports/bp/rlsrc/{side,an,an2,an3,an4}.py (면별 표, 구간별 마진/킬, 헌터 수 추이). 주의: bp_focus는 원격 코드 동기화를 안 함 (원격에 `git push -f poscat_gpu1 BR:bp-incoming` + `checkout -f -B`).

## 2026-11-01 RL2(rl_bp2_300) 분석 → B5 (feat/bp-teacher)

- RL2 vs B4 / C0 (v27_eval cap 150 s, 48 seeds × 2면 두 블록): seeds 8.1M(mac) +23.4±5.4 / +21.8±5.7, seeds 9.0M(gpu1) +18.6±5.7 / +10.7±6.3
  (부모가 본 −2.8과 다름 — 같은 체크포인트(md5 일치), 오프닝 레이스는 99/99 동시틱 와이프). 2차 레이스 RL 37승/7무/11패 (B4 상대).
- 출처 분해 (mixcls 정책: 클래스별로 RL/B4 수 선택, 48 seeds×2면): RL 헌터만 + B4 나머지 +10.4±6.1, RL 콜렉터·캐리어만 + B4 헌터 +2.8±7.5 → 헌터 운용이 핵심.
- 궤적 (24 seeds × 2면, `reports/bp/rl2src/an*.py`): 오프닝 와이프 후 (RL 헌터수, B4 헌터수) = (1,1)이 흔한데(130틱, 28/48판), RL은 70틱 안에 13/28판에서 2번째 헌터를 확보하고 B4는 거의 못 함
  (1v1 상태 재충원율 RL 0.86 vs B4 0.21 /100틱). 이후 상대가 헌터 2마리 나와 있으면 재충원율 0.05~0.1/100틱 (고착 상태) → 400–800틱 헌터 수 RL 1.66 vs B4 0.89, 콜렉터 킬 13.5 vs 5.9.
  원인: B4는 `hunter_roles=raid/raid,roles_when=always`라 남은 헌터 1마리가 성소를 떠나 레이드(예: 성소에서 13칸) → 상대 예비 콜렉터가 막힘 없이 성소로, 우리 재충원은 hunter-free 경로 규칙에 막힘.
  RL은 남은 헌터를 성소 접근로 ~4.7칸 지점에 세워 두고 (적 예비 콜렉터 킬) 자기 콜렉터를 들여보냄.
- 시도: `refill_reach` (헌터 있는 상태에서 재충원 워커 강제 커밋; safe=true는 거의 발동 안 함, safe=false는 확인 96 seeds +0.5±1.5 → 무효).
- **`guard_fill`** (v34_bp, 기본 off): 팀 헌터 < quota(2)이고 `guard_after` ≤ t < `guard_until`이면 sentry 헌터(`solo_sentry`가 만드는 성소 파수)를 raid 역할에서 제외 → 쿼터가 찰 때까지 성소 post 유지.
  `guard_enemy`(적 헌터 ≥ N일 때만) 노브 포함. post = `sentry_offset`(성소 중심에서 적 스폰 방향 거리, 기본 2.5).
  스크린/확인 (vs B4;C0 합, 확인은 신선 seeds 96×2면): guard_fill(after 6) −3.9→ 무효, **guard_after=3** (오프닝 직후부터) +13.5 스크린/+6.5±2.8 확인,
  +sentry_offset 4.5 +27.7±3.2, offset 6 +31.5±3.2 (w3), offset 6/7 확인 +19.4/+18.7 (w4, 다른 seed 블록), offset 8 +28 스크린, offset 10 −27 (post가 막힘/적 기지 쪽).
  guard_until/guard_enemy/sentry_radius는 차이 없음. 별칭 **B5 = B4 + guard_fill=true,guard_after=3,sentry_offset=6**.
- 확인 (rr5, 96 seeds × 2면, seed0 70M): B5 vs B4 +26.8±3.2, C0 +27.9±3.2, B3R +30.7±2.9, V34c +75.4±1.5 (평균 +40.2); B4 vs C0 +1.1±3.0.
- 도구: `reports/bp/rl2src/` (an2/an3/an5~an16, ts.py, screen.py, tab.py, diag_*.py, sync.sh), `examples/v27_eval.py`에 `mixcls:<클래스숫자>:<A>|<B>` 정책.
- B5 vs RL2 (rl_bp2_300, 48 seeds × 2면 두 블록): seeds 8.1M +45.1±5.6 (승률 .76), seeds 9.0M +36.9±6.0 (.75). (B4 vs RL2: RL2가 +23.4 / +18.6.)

## 2026-11-01 RL3(rl_bp3_750) 분석 → B6 후보 없음, B5 유지 (feat/bp-teacher)

- RL3 vs B5 (v27_eval cap 150 s, seeds 8.1M, 48 × 2면): A면 +26.1±8.2 (승률 .71), B면 +15.1±7.8 (.58), 평균 +20.6±5.8 (.65). B5 vs B5: +1.1/−1.1 (면 대칭 OK).
  RL 첫 헌터 98틱(69/96) vs B5 99틱. 100틱에 양쪽 2.0 → 2차 파도 후 (RL, B5) 헌터 ≈ (1,1) (B5 미러는 0v0 전멸).
- 클래스별 mixcls (96): RL 헌터만 + B5 나머지 −1.4±6.6, RL 콜렉터·캐리어 + B5 헌터 −43.8±5.4 → RL2와 달리 클래스 분해 불가 (콜렉터/캐리어가 RL 헌터와 한 세트).
  switch 정책: RL→B5 (8/16/32/64 s) −12.5/+1.4/+13.1/+20.3, B5→RL (8/16/32/64 s) −4.0/−2.3/+0.1/+1.2 → 우위는 오프닝+중반 전체에 걸쳐 형성(64 s까지 RL이어야 +20).
- 궤적 (24 seeds × 2면, `reports/bp/rl3src/`): 마진 누적 110–400 +5, 400–800 +15/+17 (면별), 800–1200 +11/+9. 헌터 상태별 시간: (2,0) 27 %·(2,1) 15 % (마진 +9/100틱), 이 둘이 이득의 ≈ 80 %.
  1v1 → 2차 헌터는 RL 36 : B5 32로 동률. 차이는 그 뒤: **2v1 전환** RL 34/36이 1대1 교환((2,1)→(1,0), ~110틱)인데 B5(2v1 쪽)는 11/32 (나머지는 RL이 18번 2v2로 재충원),
  교환 후 (1,0)→(2,0) 재충원 38회 vs 반대편 4회. RL 외로운 헌터는 B5 출구 앞 x≈17.5 (적 기지 7칸 이내 31 %, 적 콜렉터가 성소 5칸 안이면 19 %, 9–12칸이면 50 %)에서 출구를 지킴 → B5 콜렉터가 기지 안에 머무는 비율 64 % (미러 43 %),
  RL 콜렉터 수집 중 배터리 보유 19 % (B5 4–9 %), 100–400 입금 빈도 +30 %. RL 워커는 B5 정적 보초(성소 6칸)의 반대편 동쪽으로 돌아 들어가 성공(보초가 sentry_radius 6.15 안에서만 반응).
- 시도 (B5 기준, vs RL3 cap 150, 48 × 2면 paired): refill_reach 6/10/14 (safe=false) −2.1±2.3/+4.3±4.8/−2.9±5.0; guard 끄기(=B4) −27.3±7.2 (RL3는 가드 없으면 훨씬 강함),
  sentry_offset 3 −12.6±6.9, guard_until=60 −0.8; **guard_leave** (보초가 적 콜렉터가 성소 R칸 안에 없으면 출구 습격) R=7/9 −32.7/−34.8 (성소 놓침), 12 −0.1; **guard_deny** (헌터 2마리여도 적이 정원 미만이면 보초 유지) −7.0±4.3 (vs B5 −4.9±2.4);
  **hunt_hh=30/12** (2v1이면 한 마리가 적 외로운 헌터를 추격, 신규 노브) vs RL3 +5.1±4.0 / +3.4±4.0, vs B5 +0.3±2.2.
- CEM (hunters 그룹, B5 기준, vs B5;C0;B3R 32 seeds, 확인 96): 최고 +5.0±1.4 (sentry_offset 9 + raid_value 3.2 → B5 +3.1/C0 +10.3/B3R +1.5), 재확인 (64 → 128 seeds, B5;C0;B3R;B4):
  offset 9 +2.6±1.1 (B5 +0.9±1.4, C0 +6.1, B3R −1.2, B4 +4.6), +raid 3.17 +hunt_hh 6.8 +2.9±1.2 (B5 +1.0±1.5) — 이득은 가드 없는 상대(C0/B4)에서만, offset 7.5/8는 −2~−3, 비단조라 노이즈로 판단.
- **B5H = B5 + hunt_hh=30** 신선 시드 128 × 2면 두 블록: vs B5 −2.3±1.4 / +0.3±1.8, B4 +5.9/+1.9, C0 +2.3/+2.6, B3R −1.4/+0.1, V34c −1.2/−2.9 → B5를 못 이김. **B6 없음, B5 유지.**
- 결론: RL3의 우위는 (a) 2v1 헌터 교환 전환 (b) 외로운 헌터의 출구 앞 대기 + 적 콜렉터 접근 시 복귀 (c) 정적 보초의 반대편으로 도는 워커의 조합이며 개별 규칙은 B5 대비 이득이 없음.
  다음 시도 후보: 보초 위치를 적 워커 접근 방향에 따라 이동시키는 규칙(정적 보초 착취 방지), 출구 대기와 복귀를 시간 계산(헌터 속도 6 vs 콜렉터 4)으로 묶은 `guard_leave`.
- 도구: reports/bp/rl3src/ (cmp.py 쌍별 비교, an*.py 궤적), examples/diag_{refill,walk,trace,hh}.py, scripts/bp_focus.py (--n 0 --extra 로 수작업 후보).

## 2026-11-02 RL4(rl_bp4_final) 분석 → B6 (feat/bp-teacher)

- RL4-final vs B5 (v27_eval cap 150, 48 seeds × 2면, seed0 8.1M): A면 +39.2±6.4 (.75), B면 +28.2±8.3 (.71) (RL 우위 평균 +33.7). 궤적(24×2, reports/bp/rl4src): 오프닝 후 RL·B5 모두 헌터 1 (1v1, 110틱 0.98 vs 0.98),
  마진 110–400 +13, 400–800 +15, 800–1500 +7. 110–400 RL 콜렉터 킬 7.1 vs B5 3.1 (헌터당 0.56 vs 0.25), RL 헌터가 B5 출구 근처 24 % vs 3 %, RL 콜렉터 보유율 .97 vs .63.
- 새 노브 (모두 기본 off, 효과 없음): `sentry_track`(보초 post가 성소 근처 적 워커 방향으로 이동; track_hold), `guard_trip`(외로운 보초가 헌터 6 vs 콜렉터 4 시간 계산으로 돌아올 수 있는 동안만 적 출구 "south" post 캠프; trip_post, trip_engage),
  `refill_free`(적 헌터 없어도 refill 커밋). 스크린 (24 seeds×2, B5;C0;B3R, paired vs B5): track12 −5.0, track12+hold4 −7.5, track9 −5.2, trip0.3 −3.9, trip0.8 −2.2, trip east −44, trip+track −3.5.
  vs RL4-final (48×2): trip0.3 +1.9±6.2, trip0.8 +2.5±5.3, track −6.1±6.1 → RL 상대로도 무효. refill_free reach 20: 24 seeds에서 B5 +13.8였으나 32 seeds 재측정 −3.8 (노이즈).
- 진단 (B5 미러, 24 seeds): 미러는 100틱 주기의 락스텝 파도. 오프닝 직후 양 팀 콜렉터 4마리가 0v0로 같은 틱에 성소 진입 → 동시 변신·상호 사망 (후속 변신 중 life ≤5틱 짧은 헌터 약 23/경기, 같은 틱 782/1096,
  ±3틱 이내 거의 전부). 110–400틱 중 72 %가 헌터 0v0. RL 상대로는 이 루프가 없음(변신 1.3–1.9/경기). → 외로운 헌터(1v1)용 규칙(guard_trip, sentry_track)은 미러에서 거의 발동 안 함. RL 우위의 출처와 B5 미러의 약점은 다른 체제.
- 해결: 노브가 아니라 **CEM** (bp_focus f1: B5 기준, groups hunters,race,economy,hunting,evasion,specials, n=64, scale 1.0, vs B5만, 32 seeds 스크린 → 128 seeds 확인 +17.6/+12.1/+11.7/+7.2).
  상위 3개를 풀(B5;B4;C0;B3R;V34c)에 확인 (신선 seeds 90M, 96×2, paired vs B5): #1 −0.9±1.6, #3 −2.2±1.6 (B5 전용 카운터), **#2 +8.5±1.4** (B5 +13.3, B4 +14.2, C0 +10.4, B3R +3.6, V34c +1.0, 전부 ≥0).
- **B6 = B5 + #2**: guard_radius 9.358, late_after 25.95, exposure_cost 0.6541, w_dspd 4.49, adapt_at 5.732, hunt_camp_radius 2.53, commit_reach 0.9771, idle_band 0.2026, w_bsize 0.9526, dom_danger_hunter_radius 3.756,
  sentry_offset 5.831, hunt_hh 11.49, carrier_hunter_scale 0.8389, trade_floor 0.239, w_dsize 0.0523, raid_carrier 0.5195 (params.py `ALIASES["B6"]`).
- rr (seed0 70M, 64 seeds × 2면): B6 vs B5 +11.7±5.0, B4 +32.7±4.7, C0 +30.3±4.4, B3R +34.7±3.9, V34c +77.4±1.5 (평균 +37.4; B5 평균 +29.6). B5 vs B4 +24.9, C0 +26.7, B3R +30.0.
- RL 상대 (cap 425, --stop-exhausted, 48×2, seed0 8.3M, 헌터 해석 B6/B5): vs RL3-750 B6 −24.0±6.1 / B5 −21.6±6.5 (paired −2.4±6.2); vs RL4-final B6 −20.1±6.7 / B5 −23.0±5.8 (paired +2.9±6.1) → RL 대비 변화 없음 (여전히 RL이 우세 ~−20).
- 한계: B6는 노브 16개 번들(CEM)이라 개별 기여는 미분해. RL 우위(1v1 후 재충원·출구 캠프)는 이식 못 함. 도구: scripts/bp_cand.py (수작업 후보 paired), reports/bp/rl4src/ (pos.py, churn*.py, waves.py).

## 2026-11-02 라운드 5: RL5(rl_bp5_450) → B7 시도 (50 Hz) — B7 없음, B6 유지

- 50 Hz 진단 (rl5 vs B6, 24 seeds×2면, 청크 12개 평균): RL5 마진 ≈ +41 (청크별 +16~+64) → 50 Hz에서도 RL이 B6보다 크게 우세. 궤적 덤프: reports/bp/r5/traj, 스크립트 reports/bp/r5/sides.py.
- CEM g1 (B6 기준 65 후보, 풀 B6;C0;RL5, 24 seeds 스크린 → 96 확인): 스크린 최고 +8.3±3.6, 확인 +1.0±2.7 / +0.2±2.0 / −5.0 / −5.7 → 무효.
- 손 후보 (refill 노브, 24 seeds×2, B6;C0;RL5 paired vs B6): reach14+safe=false +4.7/+3.4, +after=3 +5.2±2.4, reach20 +4.1, reach10 +0.8, reach7 −2.0, margin 계열 ≤+1.6.
- 신선 시드 확인 (seed0 91M, 96×2, 풀 B6;B5;B4;C0;B3R;RL5): reach14+safe=false+after=3 paired +1.0±1.2 (B6 −0.8, B5 −3.0, B4 +2.7, C0 +0.1, B3R 0.0, RL5 +6.7);
  reach14+safe=false +0.4±1.0 (B5 −2.5, B3R −1.5). → B5에 지고 전체 무효. 스크린 이득은 노이즈/RL5 전용. 결론: B7 없음.
- 교훈: 24 seeds 스크린의 +5는 반복해서 확인에서 사라짐; RL 우위(재충원·출구 캠프)는 규칙 하나로 이식 안 됨. 결과: reports/bp/r5/*.out.

## 2026-11-02 OOD-BP: 패치 후 OOD 상대 스위트 (feat/bp-teacher)

- 구성: `blackout_env/train/distill/ood_bp.py` (OODBP = 20개 (이름, 디스크립터, 그룹, 설명)), `v27_eval.py play --hz 50 --opponent oodbp | oodbp:<그룹>[+..] | oodbp=<이름>,..`,
  리포트 `examples/oodbp_report.py --out <dir> --base B6` (상대별 margin±SE / 승·패율 / cat(<-20) / p10 / vs base paired, 후보별 약점 요약), 테스트 `tests/test_ood_bp.py`.
  새 정책 클래스 `strategic_oodbp` (`heuristics/oodbp.py`, V34 + `hunter_plan="0:2/20:0/50:2"`(시간별 헌터 목표수) + `hide=home|stay` + `raid_squad`/`field_weight`(필드 배터리 안 줍는 스틸러 분대)).
- 핵심 사실: 패치 후 경기는 ~40–100 s(stop-exhausted)에 끝나고 오프닝 헌터 레이스에서 지면 뭘 해도 B6에게 −80 (random/idle/V17 스타일/eps≥0.03 노이즈/hunter_quota 0–1 전부).
  따라서 "유능한" 성격은 B6 오프닝을 유지하고 이후만 다르게 한다 (B6 노브, 또는 `switch:25:B6|perturb:..`). 노이즈를 tick 0부터 주면 레이스만 망가짐 → `sloppy`(freeze 0.04) 하나만 남김.
  "늦은 폭발"은 150 s 이후 의미 없음(경기가 이미 끝남) → late_burst는 20–50 s 창. 캐리어는 규칙상 1명뿐이라 carrier-heavy는 불가. mixcls(B6 헌터 + idle/random 나머지)는 레이스 패배 → 제외.
- 스위트 (B6가 이긴 margin, 24매치): exit_camp(양 헌터 적 출구 캠프) +14, raid_south(1 레이드 + 1 남쪽 출구) +5, blockade(성소 봉쇄) +17, hunter_duel +20, fwd_post +3, turtle +23,
  stealer(3명 분대 스틸 전용) +20, pure_stealer +26, eco_boom(12 s 후 헌터 0, 기지 은신) +60, late_burst +12, storm −8, decoy(bait) +4, camp_after_open +16, noisy_eps +16, laggy +21, sloppy +53, v34c +12,
  random +97, idle +102, early_rush(V17) +80. 행동 검증: reports/bp/ood/behav.py + econ/ (헌터-초 timeline, 스틸 비율: stealer 27/89, pure_stealer 24/41 vs B6 20/90, exit_camp 헌터 킬 4 vs B6 92 등).
- 베이스라인 (50 Hz, cap 425, stop-exhausted, 12 seeds×2, seed0 6.0M; reports/bp/ood/base, 확인 36 seeds×2 `confirm/`): 17 비-crushed 평균 margin RL6-1050 +32.6, RL5-450 +37.1, B6 +18.4. crushed는 전원 승률 1.00 (+78~+102).
  약점: RL6 vs raid_south −27.6 (확인 −19.6, 합 96매치 ≈ −21.6, 패율 .64, 대패(<−50) 50 %), RL6 vs blockade −12.9 (확인 +0.4 → 합 ≈ −3); RL5 vs blockade −27.7 (확인 −1.5 → 합 ≈ −8), RL5 vs raid_south +26 → −2 (합 ≈ +4).
  즉 RL은 "헌터 한 마리는 레이드, 한 마리는 적 남쪽 출구"와 "성소 봉쇄" 같은 헌터 운용 변형에 B6보다 약하거나 비슷하고, 나머지(스틸러/노이즈/turtle/storm 등)는 B6보다 훨씬 강함. B6 자체는 storm(−8)·raid_south(−5)에 진다.
  → RL 학습 풀 추가 우선순위: raid_south, blockade(+ storm, stealer).

## 2026-11-02 익스플로이터: RL7(rl_bp7_1200)을 이기는 휴리스틱 (feat/bp-teacher, 50 Hz)

- 방침 변경: RL의 강점을 휴리스틱에 이식하지 않고, RL 단계마다 최신 RL을 이기는 익스플로이터를 만들어 다음 RL 풀에 넣는다. 레지스트리 `blackout_env/train/distill/exploit_bp.py` (`EXPLOITERS` = (이름, 전체 디스크립터, 설명); RL 상대 로더가 그대로 받는 `switch:<s>:<A>|<B>` / `strategic_v34:...` 문자열, `oodbp=` 문법 아님), 테스트 `tests/test_exploit_bp.py`.
- RL7 약점 (cap 425, stop-exhausted, 24 seeds x 2면 = 48매치, seed0 20.0M; RL 기준 margin): raid_south -27.9±8.6 (승률 .33), blockade -13.4±7.7 (.40); 나머지는 RL 우세 (turtle +43, fwd_post +38, hunter_duel +29, exit_camp +29, camp_after_open +28, stealer +27, storm +18).
  raid_south 풀링(5블록, ~290매치) RL7 ≈ -21; blockade ≈ -10.
- 진단 (traj_dump 48매치, `reports/bp/exp/an*.py`): 오프닝 레이스(tick ~200) 직후 raid_south/blockade 상대로는 양 팀 헌터가 전멸(0 v 0, tick 250; B6 상대로는 1.5 v 1.5 유지).
  이후 2차 파도: 익스플로이터의 예비 콜렉터 3.8마리가 tick 350에 성소 6칸 안 (RL 1.8) → tick 450에 헌터 2 (RL 0.1~0.5) 선점 → 400~1000 구간 헌터 1.7 v 0.4.
  RL 경제가 멈춤: 점수 RL 35→43 (400→1600틱) vs 익스플로이터 28→79 (B6 상대는 RL 30→60 / B6 29→49). RL 콜렉터는 자기 남쪽 출구 근처에서 매치당 ~7.9마리 사망(B6 상대 mirror보다 큼), 출구를 바꾸지 않음(S 출구 사용 비율 그대로).
  RL은 상대 헌터가 있으면 성소 재진입을 포기 → 한 번 뺏기면 못 되찾음.
- 핵심 발견: 역할(hunter_roles)을 끝까지 유지하면 약해지고, **오프닝+2차 파도 동안만(6~8 s) 쓰고 B6로 복귀**하면 훨씬 강하다.
  switch 시점 (raid_south|B6, RL7, 96매치): 4 s -25.8 (레이스 깨짐), 5/6/7 s +53.6 (동일), 8 s +44.7~52.6, 9 s +40.7, 10 s +27, 13 s +33, 17 s +27, 영구 유지 +19 (CB). B6 후 raid_south(6 s) -20.
  blockade 첫 단계 (switch:T:BL|B6): 6/7 s +49.4, 8 s +50~52, 10 s +32, 12 s +30, 16 s +26; 첫 단계의 sentry_offset 3 / radius 14 / min_hunters 2 / guard_deny는 결과 동일(+51.4) → 효과는 B6의 `hunter_roles=raid/raid`(기본)를 오프닝 동안 꺼서 헌터가 레이드 대신 성소/기본 행동을 하는 것에서 나옴 (offset 0.5는 -4.3로 깨짐).
  지속형 노브 튜닝(east_engage, raid_value, hunt_hh, storm, refill_reach, steal_weight, reserve_count, guard_after, raid/east, exit_guards, bait ...)은 모두 24~96 시드 노이즈 범위(±6~8) 안, 24 시드 스크린 +39→재측정 +19처럼 사라짐.
- 익스플로이터 (margin = 익스플로이터 기준, 신선 시드; RL7 96매치 seed0 28.0M / RL6-1050 48매치 29.0M / RL5-450 48매치 29.0M / B6 48매치 30.0M):
  | 이름 | RL7 | RL6 | RL5 | B6 |
  | X2_bl8 (blockade 8 s → B6) | +51.8±4.8 (.83) | +48.9±7.1 | +39.1±8.3 | +3.2±8.2 |
  | X3_combo8 (raid_south+blockade 8 s → B6) | +39.1±6.1 (.74) | +18.8±9.6 | +26.8±9.4 | -15.4±7.7 |
  | X4_combo (영구) | +35.8±5.8 (.75) | +9.5±9.1 | +13.7±9.0 | -22.4±6.8 |
  | X1_rs6 (raid_south 6 s → B6) | +29.9±6.2 (.71) (스크린 +45.7/+53.6 → 3블록 ≈ +43, n=288) | +36.2±7.9 | +0.4±9.9 | +6.5±7.8 |
  | X5_exits8 (south/east 8 s → B6) | +27.8±6.3 (.68) | +33.6±8.1 | +7.9±10.0 | -5.1±8.5 |
  X2_bl8이 가장 일반적(세 RL 모두 +39~52, B6와는 동률)이라 풀 우선순위 1순위, X1/X5는 RL7/RL6에 강하나 RL5에는 약함(체크포인트별 편향), X3/X4는 B6에 지는 극단형.
- 도구: `reports/bp/exp/` (screen1.sh = 로컬 스크린, screen_gpu1.sh = gpu1 실행, tab.py/tab2.py/sumry.py 집계, an1~an7.py 궤적 분석, base7.sh), `examples/traj_dump.py` 태그에서 "/" 제거.

## 11-01 ~ 11-03 요약 (세션 압축 전)

- RL 단계 1~9 (league PPO). 50 Hz로 전환 (25 Hz 평가가 이득 과대평가: rl4 vs rl3 +16@25 → +4@50). beta 0.015, clip 0.15.
- ONNX 저장소 추적기 버그 수정 (같은 틱 입금+도난, 13d1f84): 그래프-시뮬 행동 일치 93.8% → 99.6%. 이전 ONNX 전부 무효, `*f`만 사용.
- 휴리스틱: B4 (race_order) → B5 (guard_fill) → B6 (CEM). RL 이득 이식은 수확 체감 → 사용자 지시로 **RL 약점 공략(exploiter)** 방향 전환.
- OOD-BP (20 개성 정책, ood_bp.py) 구축. RL7은 raid_south −32, X2_bl8(8초 블로케이드 후 B6) −52로 취약.
- exploiter 풀 투입 (rl_bp8~9): 교사(B6) 조건부 증류는 실패 (오프닝 붕괴). 교사 없이 풀만으로 rl9 final: X2 +24.9, raid_south +37.1, rl7 +21.8, rl8 +19.0 → 제출 후보.
- 공식 RR (수정 ONNX): rl4f 14-10, rl6 13-11, rl5 12-12, rl3 9-15 (노이즈 수준).
- KDA 시간 메모리 RL 프로토타입: 동작함 (메모리 사용 확인), 아직 stateless 대비 우위 없음. 상태 ~4 MB/호출, 공식 최대 지연 96 ms → 축소 필요.
- 서브에이전트: 11-03 오후부터 403 (조직이 Claude Code 구독 접근 차단) → 직접 진행.
- 진행 중: rl_bp10 (rl9 it1200에서), rl9 final 공식 테스트 (vs rl4f 16판), rl9 final OOD 스캔 (gpu1), KDA 계속 (GPU1).

## 11-04
- rl_bp10 종료: final은 rl9 수준 정체. 50 Hz 확인 (64판): **rl10 it1050이 exploiter에 가장 강건** (X6 -2.4 vs rl9f -30.9, X2 +50.9, blockade +52.5). 공식 rl10_1050f vs rl9f 8-7-1 (-2.6±16.8), 최대 지연 55 ms → 제출 후보 rl10_1050.
- exploiter 2라운드: X6_cap6 (blockade 6 s + cap_enter 0.35) rl9f 상대 +20.7±6, X7_capd6 +15.5. pool11에 추가.
- rl_bp11: rl10 it1050에서 시작, pool11 (49개).
- rl9 final OOD: 평균 +50.7, 최약 blockade +19.8.
- KDA kda_b: stateless 대비 우위 없음 → 보류.
- 예정 (GPU 비면): **오라클 실험** — 바닥/보유 배터리 개수를 보는 모델을 rl10 it1050에서 ~600 it 학습해 마스킹 모델과 비교. +20 이상이면 배터리 개수 추정 트래커 구현, 아니면 exploiter/풀 방향 유지.

## 2026-11-04 정체 진단 + stop_exhausted (feat/bp-stopex)

- **학습 샘플의 ~72 %가 점수 확정 이후.** rl9 final vs B6 / vs rl9 final (50 Hz, 16 seeds × 2면, seed0 77M, gpu2 `~/bpruns/plat/ex_{stop,full}`): 소진 종료 on/off 마진 **64/64 동일**,
  소진 시각 평균 55 / 44 s vs 끝까지 평균 209 / 150 s → 게임 시간의 74 % / 71 %가 점수 불변 구간. PPO trunk는 소진 종료가 없어 평균 게임 ≈ 8k 결정(반복당 종료 ~4판) →
  단계당 ~5k판, 49 상대 → 상대당 ~100판 결과만 학습.
- **승부 결정 시점** (traj 200 s, `plat/traj`): 리드-최종 상관 0–10 s ≈ 0 (부호 일치 .25–.56), 15 s .16/.59, 20 s .71/.88, 30 s .82/.95, 45 s .95/.98, 점수 마지막 변화 중앙값 42–44 s.
  λ=0.975 @50 Hz = 신용 지평 ~0.8 s → 초반(0–20 s) 결정의 신호는 전부 critic 의존.
- **단계 내 개선은 풀 특화**: rl_bp10 상대별 EMA (it30→끝) 휴리스틱/옛 넷 +0.1~+0.3, 최신 넷 rl9_1200 −0.09, rl8 −0.03, rl7 −0.02. 학습 중 평가 32판 SE ±10–14 → 단계 간 차이 판별 불가.
- 구현: `PPOConfig.stop_exhausted` (기본 off = 기존과 동일; on이면 소진된 trunk 게임을 종료로 처리: 종료 보상, 부트스트랩 없음, 새 시드 재시작),
  `rl_train.py --eval-stop-exhausted` (평가도 소진 종료, 마진 동일). 테스트 `tests/test_league.py::test_stop_exhausted_ends_game`. 학습 A/B는 빈 GPU가 없어 미실행.
- 다음 후보: (1) 다음 RL 단계에 `stop_exhausted=true` + `--eval-stop-exhausted` + eval seeds ×3, (2) 초반 신용할당 (λ 0.99 / 오프닝 이후 action repeat / 0–20 s 분기 CRN 롤아웃 이득 = GRPO r3),
  (3) 최신 자기 비중↑ 또는 RL 익스플로이터, (4) 탐색 Q 회귀 증류.
- **A/B it600** (rl_bp11sx = rl_bp11 레시피 + stop_exhausted, gpu2 GPU2,0; 종료 게임/반복 13.6 vs 3.7, 반복 시간 동일): gpu1 `~/bpruns/sx/cmp_600`, 48 seeds × 2면, seed0 62M, stop-exhausted.
  h2h sx vs base **+14.0±6.7** (.62). paired sx−base: X6 **+21.0±9.2**, X2 **+17.9±8.5**, blockade **+23.5±7.1**, rl9f +14.7±9.8, raid_south +3.2, B6 +2.0, rl10_1050 −1.5. final 비교는 학습 종료 후 자동 실행.
- **A/B final** (rl_bp11sx final vs rl_bp11 final, gpu1 `~/bpruns/sx/cmp_final`, 96 games/opp): h2h **+18.8±7.1** (.62). paired sx−base: X6 **+28.5±9.4**, rl9f **+19.7±9.2**, rl10_1050 +8.9, X2 +7.3, raid_south +6.5, blockade +0.8, **B6 −8.1±7.4** (sx still +31.1, win .75). Mean over 7 opponents +9.1.
  → stop_exhausted adopted as the default candidate for later RL phases; rl_bp11sx final is the new candidate (OOD scan and official ONNX check not done yet).
- **OOD scan** (gpu1 `~/bpruns/ood11all`, 20 OOD opponents × 32 games, same seeds as rl9 scan, seed0 34M): non-crushed mean sx_final **+47.7** / base_final +51.7 / rl9f +50.7;
  worst: sx camp_after_open +28.4 / base stealer +27.8 / rl9 blockade +19.8; cat (<−20) .10 / .08 / .07; crushed win 1.00 for all.
  paired sx−base, 20 opps: **−3.5±1.8**. Losses concentrated in opponents that switch mid-game: camp_after_open (switch 30 s) −22, laggy (lag after 25 s) −18, noisy_eps (25 s) −13, hunt_hh30 −11.5, storm −7;
  gains raid_south +5, blockade +6.5, stealer +12 (by margin). → sx: strong opponents/exploiters +9~+28, OOD slightly weaker (mid-game behaviour change). Next phase: stop_exhausted + more weight on switch:25–30 personalities in the pool.
- 11-05 오라클 결론: 배터리 개수를 보여줘도 이득 없음 (it300 vs rl10_1050 -4.8±8.1, 64판; it600 학습 중 평가도 rl11과 동급) → 정보 부족은 병목 아님, 추정 트래커 안 만듦.
- exploiter 3라운드 (X6 계열 12개) vs rl10_1050: 이기는 후보 없음.
- rl10_1050 OOD: 평균 +44.7 (rl9f +50.7). blockade +51.8로 개선, 대신 stealer +19.8 / storm +26.5 등 경제·템포형 약화.
- rl_bp12: rl10_1050에서, pool12 (경제·템포 OOD 가중치 상향) — blockade 대응 유지 + 경제형 회복 목표.

## 11-06 – 11-09 요약
- 크리틱: 분기 롤아웃 평가셋(cb1, cb4, cb6, cb7)으로 측정. HL-Gauss 크리틱을 오프라인 사전학습(critic_bench) → rl14부터 사용. 학습 중 크리틱(rl15/rl16)이 오프라인보다 정확 (rl16 it1050: 네트워크 상대 10초 R² 0.63–0.77, 편향 ~0). 휴리스틱 상대엔 −0.5 비관 편향 (held-out과 풀 내 상대 차이 없음, 분기 시 휴리스틱 재시작의 영향 일부).
- rl14 (HL 크리틱) → rl15 (크리틱 갱신, it1800) → rl16 (it1050). 릴리즈: org `rl15-2026-11-07`, `rl16-2026-11-08` (Latest).
  - rl16_1050 vs rl15_1800: 시뮬 128판 +8.2±5.0, 공식 9–7, OOD-HO +61.2 (최약 +49.0).
- 제출 형식 검토: 공식 check_submission OK, robust_check (빠진 유닛/타임아웃/재시작) 통과.
- 공식 실행기: gpu2 nvidia-persistenced 중단 → `COMP_PASSTHRU=1` (장치·드라이버 직접 마운트) 우회.
- exploiter 6라운드 (서브에이전트): **assassin** (헌터가 적 헌터를 보는 즉시 추격) — rl16_1050 +47, rl15_1800 +44, rl14_1200 +49 (64판). RL 정책 공통의 오프닝 구멍(살아남은 헌터가 출구 캠핑).
- rl17: rl16_1050 + X8/X9 풀 투입, gpu2 GPU 0+2 비동기, 감시 스크립트로 자동 재개.
