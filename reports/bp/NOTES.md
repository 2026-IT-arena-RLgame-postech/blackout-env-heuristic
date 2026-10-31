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
