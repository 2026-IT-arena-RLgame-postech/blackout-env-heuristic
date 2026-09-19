# BlackOut 휴리스틱 정책 카탈로그와 혼합 가이드

이 문서는 BlackOut의 BC(Behavior Cloning), offline RL, 온라인 강화학습 부트스트랩에 사용할 수 있는
휴리스틱 정책들의 차이, 검증 결과, 혼합 방법과 데이터 기록 규약을 한곳에 정리한다. 구현 세부와
이동 복구 원리는 `heuristic_policy_ko.md`(V1 설계 문서)를, 각 버전의 결정 규칙은
`blackout_env/heuristics/*.py`의 모듈·메서드 docstring을 함께 참고한다.

## 처음 보는 사람을 위한 안내

등록된 정책은 20개다: `blackout_env/heuristics/mixture.py`의 `POLICY_REGISTRY`에 있는
`strategic_v1` … `strategic_v19`(19개)와 `strategic_v4_near`(`V4PolicyFamily`, V4 파라미터 근접 변형).
`make_heuristic("strategic_v17")`처럼 ID로 생성한다. Elo는 `blackout_env/train/policy_strength.py`의
`ELO_20260916`(64초 절단 경기, 20정책 전체 적합, 평균 1500 기준)이다.

| 용도 | 정책 | 비고 |
|---|---|---|
| 표준 평가 상대 | `strategic_v4` (`RecommendedStrategicHeuristic`) | `periodic_eval`, `offline_pretrain`의 평가, on-policy 기본 상대. Elo 1500으로 중간 강도 |
| 가장 강한 상대 | `strategic_v19` (2059) > `strategic_v17` (1832) ≈ `strategic_v18` (1831) | 나머지는 모두 1550 이하. V18/V19는 각각 V17/V18 전용 카운터라 상성이 비이행적이며, V1–V16 상대로 가장 확실한 것은 V17(151승 9패) |
| 오프라인 데이터 수집 | `HeuristicPolicyMixture` | 매치마다 policy_id 하나를 기본 가중치로 뽑아 끝까지 유지(아래 "기본 정책 혼합"). `collect_heuristic_dataset(_parallel).py`가 팀마다 독립 seed로 사용 |
| 약한 대조군 | `strategic_v1` (1192), `strategic_v14` (1220), `strategic_v2`, `strategic_v5` | 스모크 테스트나 쉬운 상대 |

빠른 휴리스틱 대 휴리스틱 확인(저장소 루트에서, 기본은 창 없는 headless 실행, Unity 빌드 기본 경로는
`build/mac/BlackOut.app`이며 다르면 `--build`로 지정):

```bash
# 1) 후보 하나 대 여러 상대, 420초 전체 경기. 시드마다 양 진영을 모두 플레이한다.
#    --candidate/--opponents는 policy_id(기본 상대는 strategic_v1–v16), --output은 선택.
./.venv/bin/python examples/gauntlet_heuristics.py --candidate strategic_v19 \
    --opponents strategic_v4 strategic_v17 strategic_v18 --n-seeds 2 --workers 6

# 2) 두 정책 1:1 비교 + 이동 실패(idle/blocked) 진단. 짧은 이름(v4, v9, v4-near)을 쓰며
#    v1–v12와 v4-near만 지원한다(v1은 --baseline 전용). 시드는 5개 이상.
./.venv/bin/python examples/benchmark_heuristics.py --candidate v9 --baseline v4 --n-seeds 5

# 3) 전체(또는 --policies로 고른 부분군) Elo: 64초 절단 경기의 적응형 Bradley-Terry 적합.
./.venv/bin/python examples/elo_active.py --workers 18 --target-se 40 --output reports/elo_active_new
```

설계 비교용 64초 근사 gauntlet은 `examples/race_gauntlet.py`, 전체 쌍 승률 히트맵은
`examples/tournament_heuristics.py`(아래 "정책 상성 히트맵")를 쓴다.

## 빠른 결론

- 실전 기준 정책과 평가 상대는 `strategic_v4`를 권장한다.
- V4 주변의 조밀한 행동 분포가 필요하면 `strategic_v4_near` 또는 `V4PolicyFamily`를 사용한다.
- 역할 배정 다양성은 `strategic_v7`, 의도적인 역할 리셋 궤적은 `strategic_v9`에서 얻는다.
- `strategic_v5`, `strategic_v6`, `strategic_v8`은 성능 최적점이 아니라 각각 공격적 요격,
  저정체 위험 회피, 보수적 역할 리셋이라는 희귀 상태 분포를 제공한다.
- 한 매치(420초) 안에서는 `policy_id`(어떤 버전인지)를 바꾸지 않는다. `HeuristicPolicyMixture`는
  매치 시작 시 `policy_id`를 한 번만 고르고, 자체적으로 action noise를 넣지 않는다(수집기의
  `--noise-frac` action noise는 수집 스크립트가 별도로 적용한다).
- 다만 세부 파라미터(작은 근접 변형)는 기본적으로(`resample_each_absorption=True`) 흡수
  경계(absorption boundary, qmix_trainer가 말하는 이 게임의 실제 "에피소드" 경계, 20초)마다
  다시 샘플링한다 — 같은 `policy_id`를 유지한 채 V4-near 스타일의 좁은 구름 안에서만 값을 바꿔,
  기준 정책은 그대로 두고 BC/offline RL 샘플 다양성을 매치당 여러 번 확보한다. 매치 시작 시
  최초 1회 샘플링도 이 메커니즘의 특수 경우다.
- 흡수 경계의 재샘플링은 실행 중인 정책 인스턴스의 파라미터만 제자리에서 바꾼다(`retune()`).
  경로·탈출 타이머·역할 배정·전략 모드·respec 쿨다운·캐시는 경계를 넘어 유지된다. 역할 체계 자체를
  바꾸는 `use_specialists`는 매치 단위로 고정한다.
- 기존 버전은 삭제하거나 새 의미로 덮어쓰지 않는다. 새 전략은 새 `policy_id`로 추가한다.

## 정책 계보

```text
StrategicHeuristicV1                         strategic.py
└── V2  전역 경로 기반 작업 배정              advanced.py
    └── V3  위험 인지 적재소 선택             safe_storage.py
        └── V4  적재 타일 분산 ← 권장 기준    spread_deposit.py
            ├── V5  예측 요격                 intercept.py
            ├── V6  위험 비용 A*              risk_path.py
            │   └── V17  차단 중심 팀 planner: Hunter 3기(1기 적 본진 출구 진치기 + 2기 추격)   v17_planner.py
            │       └── V18  V17 전용 카운터: 아이템 회피 변신 경로 + 우리 출구 경비 Hunter     v18_counter.py
            │           └── V19  V18 전용 카운터: 성소 소탕 + 성소 보초 + 가까운 성소 칸 경로 v19_counter.py
            ├── V7  동적 역할 배분            dynamic_roles.py
            │   ├── V8  보수적 사망 리셋      lifecycle_roles.py
            │   │   └── V9  적극적 사망 리셋  opportunistic_respec.py
            │   ├── V10 국면 전환: opening economy / pressure raid / closeout defend   phase_strategies.py
            │   │   └── V16 V10 + storage siege / home guard / convoy rush             counterplay_strategies.py
            │   ├── V12 리드 시 fortress, 그 외 catch-up                                 phase_strategies.py
            │   ├── V14 초반 Hunter 1기로 아군 창고 주변만 방어                          counterplay_strategies.py
            │   └── V15 Hunter 없이 Carrier 1기의 고가 필드 배터리 운송                  counterplay_strategies.py
            └── V11 흡수 직전 외부 창고 약탈 창(raid-window)                            phase_strategies.py
                └── V13 외부 창고가 보이면 지속하는 3기 공성 약탈                        counterplay_strategies.py

V4PolicyFamily (strategic_v4_near) ── V4의 작은 근접 파라미터 변형 (기본은 매치 단위,
                                      `resample_each_absorption=True`면 흡수 단위)       v4_family.py
HeuristicPolicyMixture ── 위 20개 policy_id를 함께 샘플링                                 mixture.py
```

하위 버전은 표시된 부모의 경제 목표 선택과 이동 복구를 상속한다. 예를 들어 V9는 V8의 역할 리셋
상태기계를 복제하지 않고 조건 파라미터만 더 적극적으로 바꾼 버전이다.

## 모든 기본 정책에 공통인 행동

- 공개 competition observation만 사용하며 Unity 내부 특권 상태를 읽지 않는다.
- 연속 2D 이동 action만 출력하고 수집, 적재, 변신과 전투는 접촉 규칙으로 실행한다.
- 벽 semantic channel을 이용한 8방향 A*와 대각선 corner-cut 방지를 사용한다.
- 도달 불가능한 목표에서 벽을 계속 미는 대신 현재 연결 영역을 탐색한다.
- 정지, 짧은 진동, 상호 충돌과 접촉 이벤트 누락을 감지해 재계획·이탈·재진입한다.
- 적 보호 스폰과 그 안의 보호 창고를 공격/약탈 목표에서 제외한다.
- 소지품이 있으면 역할 변신보다 안전한 적재를 우선한다.
- 에피소드 종료 시간이 위로 점프하면 경로, 예약, 역할과 캐시를 초기화한다.
- 정적 지도 연결요소, 거리 맵, A* 경로를 에피소드 안에서 캐시하고 동적 아이템은 tick마다 갱신한다.

## 버전별 상세 특성

| ID | 클래스 | 핵심 특성 | 장점 | 약점/사용 목적 |
|---|---|---|---|---|
| `strategic_v1` | `StrategicHeuristicV1` | 고정 1 Carrier, 1 Hunter, 3 Collector; 개별 목표 점수화 | 단순하고 해석 가능하며 초기 BC 기준선으로 좋음 | 전역 작업 중복과 역할 고정의 기회비용이 큼 |
| `strategic_v2` | `StrategicHeuristicV2` | 경로 거리 기반 팀 전역 greedy 작업 배정; 흡수 전에 도착 불가능한 약탈 제외 | 수집 목표 중복 감소 | 안전한 적재 위치와 동시 창고 진입 고려가 부족 |
| `strategic_v3` | `StrategicHeuristicV3` | V2 + 흡수 시각, 보호 창고, 적 위협을 반영한 적재소 선택 | 운반 중 사망과 노출 감소 | 여러 운반자가 같은 진입 타일에 몰릴 수 있음 |
| `strategic_v4` | `StrategicHeuristicV4` | V3 + 동시 적재 진입 타일 예약·분산 | 현재 가장 검증된 균형형 교사 | 행동 분포가 하나의 모드에 집중됨 |
| `strategic_v5` | `StrategicHeuristicV5` | 적 위치 EMA와 가능한 저장소 경로 ensemble로 Hunter 요격점 예측 | 미래 위치를 향하는 공격적 전투 샘플 | V4 대비 직접 대전 성능과 막힘이 나빠 기본 비중 2% |
| `strategic_v6` | `StrategicHeuristicV6` | 운반자 A* 비용에 적 위협장을 통합 | 벽/막힌 길로 회피하는 비율이 낮고 저정체 경로 제공 | 우회가 길어져 점수 성능이 낮을 수 있음 |
| `strategic_v7` | `StrategicHeuristicV7` | Carrier 최대 1기; 성소까지 실제 경로가 가까운 빈 Collector 배정; Hunter 투입 지연 | 역할과 운반 능력을 명시적으로 고려 | V4보다 짧은 막힘이 많아 기준판 대신 역할 변주용 |
| `strategic_v8` | `StrategicHeuristicV8` | 열세·수송 부재·필드 자원·시간·적 Hunter 조건을 모두 만족할 때 상호사망으로 역할 리셋 | 불필요한 자살을 강하게 억제 | 10경기에서 리셋 0회; 보수적 대조군. 기본 3%(V7/V9/V12와 한 몫을 나눔) |
| `strategic_v9` | `StrategicHeuristicV9` | V8 조건을 완화해 실제 Hunter→사망→Collector trajectory 생성 | 리셋 4/4 성공, V7과 동률 성능 및 비슷한 신뢰성 | 승격판은 아니며 희귀 전략 데이터용 기본 3% |
| `strategic_v10` | `StrategicHeuristicV10` | 공개 점수·시간·흡수·상대 외부창고 가치로 `opening_economy`→`pressure_raid`→`closeout_defend` 전환; Hunter의 목표도 국면별로 변경 | 같은 지도에서도 경제 확장, 약탈 압박, 리드 보호 궤적을 모두 제공 | 새 다양성 정책; V4와의 대전·신뢰성 평가는 별도 기록 후 비중 조정 |
| `strategic_v11` | `StrategicHeuristicV11` | 흡수 직전 또는 열세일 때 최대 2기 Collector를 상대 외부 창고의 고가 배터리에 배정 | 약탈 타이밍과 다중-unit 협공 상태를 의도적으로 많이 생성 | 정상 수집보다 약탈에 치우친 policy support; 교사 주력으로는 사용하지 않음 |
| `strategic_v12` | `StrategicHeuristicV12` | 리드 후반에는 새 Hunter 변신을 늦추고, 기존 Hunter가 빈 적을 추격하지 않고 아군 창고 방어 | 수비/호위와 공격 포기라는 명확한 counterfactual 상태를 제공 | 상대가 적극적으로 역전할 때 기회비용이 생길 수 있음 |
| `strategic_v13` | `StrategicHeuristicV13` | 흡수 임박 여부와 무관하게 노출된 상대 외부 창고에 최대 3기 약탈조를 유지 | 수비형·창고 보존형 상대로 반복 약탈과 다중-unit 압박 trajectory를 생성 | Hunter 수비나 빠른 필드 경제에 취약하도록 의도된 고위험 정책 |
| `strategic_v14` | `StrategicHeuristicV14` | 초반부터 Hunter 1기를 만들고, 적 화물이 아군 창고 반경 안에 들어올 때만 요격; 그 외에는 홈 순찰 | V13/V11 같은 약탈형을 상대로 한 home-defense와 반격 관측을 제공 | 공격·운반 인력 하나를 고정 소비하므로 수동적 경제형에게 점수 손해 가능 |
| `strategic_v15` | `StrategicHeuristicV15` | Carrier 1기와 Hunter 0기를 고정하고 Carrier가 고가 필드 배터리를 우선 운반 | 약탈 대신 처리량을 택하는 장거리 convoy trajectory를 제공 | 적 Hunter·약탈을 막지 못하므로 전투형 정책과 뚜렷한 상성 차이를 만들 수 있음 |
| `strategic_v16` | `StrategicHeuristicV16` | V10의 `opening_economy`·`pressure_raid`·`closeout_defend`에, 공개 창고 가치/위협/필드 자원으로 `storage_siege`·`home_guard`·`convoy_rush`를 추가 | 하나의 에피소드 안에서 조건부 대전략 전환을 관측할 수 있어 장기 horizon BC와 offline RL에 유용 | 각 모드의 조건이 드문 지도에서는 V10과 유사하게 보일 수 있으므로 모드 provenance를 반드시 저장 |
| `strategic_v17` | `StrategicHeuristicV17` | V6 이동 위의 팀 planner: Carrier 1기(필드 가치 25 이상일 때), Hunter 3기(1기 적 본진 출구 진치기 + 2기 추격), 흡수·약탈 위험을 따지는 적재와 초당 가치 기반 경제 배정 | V1–V16 상대 151승 9패; 출구 봉쇄·Hunter 회피 부재를 노리는 차단 플레이 | 아래 V17 절 참조. 틱당 연산이 더 큼 |
| `strategic_v18` | `StrategicHeuristicV18` | V17 + Hunter 4기, 아이템을 피하는 변신 경로, 우리 본진 출구 경비 Hunter | V17 상대 78% | V17 전용 카운터로 일반 강도는 목표가 아님 |
| `strategic_v19` | `StrategicHeuristicV19` | V18 + 성소 소탕, 성소 보초 Hunter, 경로상 가장 가까운 성소 칸 | V18 상대 72%, 전체 Elo 1위(2059) | V18 전용 카운터 |
| `strategic_v4_near` | `V4PolicyFamily` | V4 exact와 세 종류의 좁은 파라미터 변형을 매치 단위 샘플링(혼합 안에서는 흡수마다 재샘플링) | V4 주변 decision boundary를 조밀하게 커버 | 완전히 다른 전략 상태는 거의 만들지 않음 |

## 주요 평가 결과

각 비교는 랜덤 5개 seed를 양 진영에서 실행한 10경기 결과다. `idle_6s`는 action과 이동이 모두
6초 이상 없는 구간, `blocked_0.24s`는 유효 action을 냈지만 0.24초 이상 이동하지 못한 구간이다.

| 후보 vs 기준 | 승-패-무 | 평균 점수차 | 후보 idle 6초 | 후보 blocked / 1,000 유닛틱 | 기준 blocked / 1,000 유닛틱 | 판단 |
|---|---:|---:|---:|---:|---:|---|
| V4-near vs V4 | 6-4-0 | +3.1 | 0 | 0.402 | 0.392 | 근접 변형군으로 채택 |
| V7 vs V4 | 6-4-0 | +4.2 | 0 | 0.379 | 0.327 | 역할 변형으로 유지, V4는 계속 권장판 |
| V8 vs V7 | 4-6-0 | -2.4 | 0 | — | — | 리셋 0회; 200배속 모니터 수정 전 수치는 제외 |
| V9 vs V7 | 5-5-0 | +1.6 | 0 | 0.350 | 0.356 | 리셋 4/4; 희귀 역할 전환 데이터용 |

V5와 V6의 기존 결과도 정책의 목적을 이해하는 데 중요하다.

- V5 vs V4: 2-8, 평균 점수차 -2.6, blocked 1.184 vs 0.305. 공격적 저가중치 변형으로만 사용한다.
- V6 vs V4: 4-6, 평균 점수차 -7.8, blocked 0.074 vs 0.284. 승률보다 저정체·보수 경로의 다양성이 목적이다.
- V4 vs V1 장기 확인: 15 seed/30경기에서 20-10, 평균 점수차 +2.3, blocked 0.339 vs 0.365.

평가 표본이 작고 고배속 Unity 물리는 완전 결정론적이지 않으므로 5 seed 결과 하나만으로 정책을
삭제하지 않는다. 승격은 승패·점수차와 두 신뢰성 지표가 함께 좋아지고 더 많은 seed에서도 재현될 때만
고려한다.

### 2026-09-13: counterplay 정책군 통합 평가

`v13`–`v16`을 추가한 뒤, 기존 13정책의 완료 대진과 새 정책의 모든 교차 대진을 결합해 17정책,
136개 비대각 pair, 5개 고정 seed × 양 진영의 총 1,360경기를 실행했다. 추가 교차 대진 40개는
18개 Unity 프로세스와 `timeScale=200`으로 수행했다. Elo는 모든 승·패·무승부를 한 번에 적합한
Bradley–Terry/Elo 값이며 전체 평균은 1,500으로 고정했다.

| 정책 | Elo | 전체 평균 승률 | 평균 점수차 | `idle_6s` | `blocked_0.24s` / 1,000 unit-tick | 평가 결론 |
|---|---:|---:|---:|---:|---:|---|
| V13 공성 약탈 | 1475 | 46.2% | +1.22 | 0 | 0.397 | V1/V2/V5에는 강하지만 V7/V8/V9/V12에는 25~30%로 약하다. 명확한 약탈 편향 정책으로 보존한다. |
| V14 홈 수비 | 1532 | 54.7% | -0.35 | 0 | **0.146** | V13에는 20%로 밀리지만 V4/V4-near/V7/V8/V9/V12에는 55~60%를 기록한다. 가장 저정체인 defensive counterplay 교사다. |
| V15 수송 러시 | 1444 | 41.6% | -2.73 | 0 | 0.573 | V1/V2에는 55~65%지만 V4-near/V7/V8/V9/V11/V12에는 20~30%다. 의도적으로 취약한 throughput/전투회피 상태를 만든다. |
| V16 확장 V10 director | 1498 | 49.7% | +1.14 | 0 | 0.517 | V2/V4/V13에는 60~70%지만 V4-near/V7/V8/V9/V12에는 35~45%다. 단일 강도보다 장기 조건부 모드 전환 evidence가 목적이다. |

당시에는 기본 혼합에서 V14와 V16을 각각 3%, 4%로 유지하고, V13/V15는 약점과 반례를 충분히
수집하기 위한 3% 희귀 정책으로 유지했다. 이 결과는 성능만으로 V13/V15를 제거하지 않는 근거이기도
하다. (이 표의 Elo와 가중치는 당시 17정책 풀·가드 수정 전 코드 기준이다. 현재 값은 아래 "기본 정책 혼합"의
2026-09-16 적합을 따른다. 예: V14는 현재 1220, 가중치 1%.) 완성된 원시 결과와 히트맵은 `reports/heuristic_tournament_all17_workers18_20260913/`에 보관한다.

## V17: 차단 중심 팀 planner (2026-09-16)

V1–V16을 모두 이기는 것을 목표로 새로 설계한 정책이다. V1–V16 코드는 건드리지 않았다. 처음에는
mixture 기본 가중치에 넣지 않았다가, 2026-09-16 Elo 재조정 때 14%로 넣었다(위 "기본 정책 혼합").

### 설계 근거가 된 측정

- **경기는 첫 1분에 결정된다.** 배터리 ~200점은 시작 시 한 번만 생성되고 재생성되지 않는다. 필드는
  ~40초 안에 비고, 세 번째 흡수(60초) 이후 점수는 거의 변하지 않는다.
- **약탈이 매우 크다.** 60초 안에 외부 창고에 적재한 점수의 대부분이 한 번 이상 도둑맞는다(V7 대 V4에서
  228점 적재 중 203점).
- **운반 중 사망이 점수를 없앤다.** 두 팀 점수 합이 보통 120 안팎이라 200점 중 80점가량이 화물 파괴로 사라진다.
- **기존 휴리스틱의 빈손 Collector는 Hunter를 피하지 않는다.** 본진 출구 앞 Hunter 하나가 나오는 유닛을
  연속으로 처치한다(V7 Hunter가 V17을 상대로 15초에 20회 이상).

### 동작

| 구성 요소 | 내용 |
|---|---|
| 역할 | Carrier 1기, Hunter 3기, Collector 1기 (`carrier_quota=1`, `hunter_quota=3`). Carrier는 필드 배터리 가치가 25 이상 남아 있을 때만 새로 만든다 |
| 진치기 Hunter | 적 본진(스폰을 포함한 코너 4×4, 적은 못 들어가는 바닥)의 중앙 쪽 출구 칸에 서서 `camp_engage_radius`(4.5칸) 안의 비-Hunter 적을 처치. 화물·Carrier 우선, 적 Hunter와는 교환하지 않음 |
| 추격 Hunter | 화물 가치/요격 시간으로 목표 선택. 적 Hunter는 우리 스폰 앞이나 노출 창고를 막을 때만 교환 |
| 경제 | 팀 전체를 한 번에 배정. 필드 배터리는 왕복 시간당 가치, 약탈은 흡수 전에 도착 가능할 때만, 속도 아이템은 효과 지속 시간으로 평가. 적재 창고는 이동 시간과 다음 흡수 전 약탈 위험을 함께 비교 |
| 이동 | V6 위협 비용 A*와 V1 막힘 복구를 그대로 사용(위협 위치가 바뀌므로 경로 캐시는 쓰지 않음) |

### 평가

64초에서 끊는 진영 교대 gauntlet(`examples/race_gauntlet.py`)으로 설계를 고르고, 튜닝에 쓰지 않은 시드로
420초 전체 경기(`examples/gauntlet_heuristics.py`)로 확인했다.

| 변형 (64초 근사, 16상대 × 4시드 × 양 진영) | 승률 | 평균 점수차 |
|---|---:|---:|
| Hunter 1기 추격 (초기 설계) | 66% | +14.7 |
| Hunter 0기 | 50% | +1.1 |
| Hunter 2기 추격 | 86% | +28.2 |
| Hunter 3기 전원 진치기 | 53% | −2.8 |
| **Hunter 3기 혼합(1기 진치기)** | **96%** | **+49.2** |
| 빈손 유닛도 Hunter 주변을 벽으로 막고 대기 | 20–27% | −21~−28 |
| 참고: V13 | 50% | −0.4 |

새 시드 6개 확인에서 채택 구성은 98.4%(+47.3)였다. 최종 420초 전체 경기(새 시드 5개 × 양 진영 × 16상대,
160경기) 결과는 다음과 같다. 원시 결과는 `reports/gauntlet_v17_final_full/`에 있다.

| 상대 | V17 승-패 | 평균 점수차 |
|---|---:|---:|
| v1, v3, v4, v5, v6, v10, v13, v14, v16 | 각 10-0 | +40 ~ +60 |
| v7, v8, v9, v11, v12, v15 | 각 9-1 | +39 ~ +49 |
| v2 | 7-3 | +31.8 |
| **합계** | **151-9 (94%)** | |

```bash
# 빠른 설계 비교 (64초 근사, 18 프로세스)
./.venv/bin/python examples/race_gauntlet.py --workers 18 --n-seeds 4 \
    --variants 'v17=strategic_v17' 'hq2=strategic_v17:{"hunter_quota": 2}'

# 전체 경기 검증
./.venv/bin/python examples/gauntlet_heuristics.py --candidate strategic_v17 --workers 18 --n-seeds 5
```

## V18: V17 전용 카운터 (2026-09-16)

V17을 고정한 채 V17만 이기도록 만든 정책이다. 다른 정책 상대 성능은 목표가 아니다. 2026-09-16부터
mixture 기본 가중치에 9%로 들어간다(위 "기본 정책 혼합").

### V17 대 V17에서 관찰한 승부처

V17 거울전은 **먼저 상대 본진 출구에 진치기 Hunter를 세운 쪽이 이긴다.** 진이 쳐진 쪽은 부활한 유닛이
다시 변신하러 나가다 출구에서 1초 간격으로 죽는다. 이 경쟁을 V17의 고정 규칙 두 개가 좌우한다.

- V17의 변신 대기 유닛은 중앙 성소로 가는 최단 경로에서 배터리를 밟으면 자동으로 줍고, 짐이 있으면
  먼저 적재하러 돌아가 몇 초를 잃는다.
- V17 진치기 Hunter는 적 Hunter를 공격하지 않고, 추격 Hunter도 자기 스폰 7칸 밖의 적 Hunter는
  (자기 노출 창고 3칸 안에 있는 경우를 빼면) 무시한다.

### 동작

| 구성 요소 | 내용 |
|---|---|
| 역할 | V17 경제 + Hunter 4기 |
| 변신 경로 | 중앙 성소로 갈 때 필드 배터리·특수 아이템 칸을 피해 A* (경로가 막히면 원래 경로) |
| Hunter 0 | V17과 같은 적 본진 출구 진치기 |
| Hunter 1 (출구 경비) | 우리 본진 출구에 서서, `guard_radius`(7칸) 안에 들어온 적 Hunter와 즉시 맞교환 |
| 나머지 | V17 추격 |

### 평가 (상대: V17)

게임이 초기 조건에 민감해(리셋 시 1틱 어긋남만으로 같은 시드 결과가 뒤집힘) 변형당 96경기로 비교했다.

| 변형 (64초 근사, 48시드 × 양 진영) | 승률 | 평균 점수차 |
|---|---:|---:|
| V17 거울전 | 50% | +0.0 |
| V17 + Hunter 4기 | 62% | +15.3 |
| **V18** | **75%** | **+24.7** |
| V18 − 아이템 회피 변신 경로 | 53% | +2.5 |
| V18 − 출구 경비 | 70% | +20.0 |

본진 대기, 짐 든 채 변신, V17 방어 반경(7칸) 밖 진치기는 효과가 없어 넣지 않았다. 최종 420초 전체 경기
(새 시드 48개 × 양 진영, 96경기)에서 **V18 대 V17은 75승 21패(78%), 평균 점수차 +24.9**였다. 원시 결과는
`reports/gauntlet_v18_vs_v17_full/`에 있다.

```bash
./.venv/bin/python examples/gauntlet_heuristics.py --candidate strategic_v18 \
    --opponents strategic_v17 --workers 18 --n-seeds 48
```

## V19: V18 전용 카운터 (2026-09-16)

V18을 고정한 채 V18만 이기도록 만든 정책이다. 2026-09-16부터 mixture 기본 가중치에 9%로 들어간다.

### V18 대 V18에서 관찰한 승부처

V18의 변신 대기 유닛 4기는 한 덩어리로 중앙 성소에 간다. 상대가 먼저 변신하면 Hunter 하나가 덩어리 전체를
처치하고, 그 팀은 Hunter를 하나도 못 만든 채 출구에 갇힌다. 90경기 거울전의 약 4분의 1이 이렇게 끝났고,
전멸한 쪽은 전부 졌다. 반대로 전멸 없이 몇 틱 먼저 변신하는 것은 승패와 무관했다(15승 15패).

### 동작

| 구성 요소 | 내용 |
|---|---|
| 성소 소탕 | 모든 Hunter가 자기 자리로 가기 전에, 성소 중심과 자기 자신 모두에서 `sweep_radius`(9칸) 안에 있는 적 Collector부터 처치 |
| 성소 보초 | Hunter 하나(진치기·출구 경비 다음 순번)가 성소의 적 스폰 쪽에 상주. V18의 어떤 규칙도 그 위치의 Hunter를 공격하지 않으며, 교환으로 Hunter를 잃은 V18 유닛은 다시 변신하려면 보초 앞을 지나야 함 |
| 변신 경로 | 경로상 가장 가까운 성소 칸으로. 아이템 회피 경로는 길이가 같을 때만 쓰고, 도중에 짐을 주우면 버리고 변신 |
| 나머지 | V18과 동일(Hunter 4기, 진치기 1 + 출구 경비 1) |

### 평가 (상대: V18)

| 변형 (64초 근사, 64시드 × 양 진영) | 승률 | 평균 점수차 |
|---|---:|---:|
| V18 거울전 | 50% | +0.0 |
| 가까운 성소 칸 경로만 | 48% | +5.0 |
| + 성소 소탕 | 59–61% | +9~+14 |
| **+ 성소 보초, 소탕 반경 9칸** | **66%** | **+20.1** |
| 위 구성 − 출구 경비 | 55–57% | +7~+10 |

대기 유닛을 서로 다른 성소 칸으로 분산(36%)하거나 적 Hunter를 보면 물러나게(20%) 하면 오히려 졌다.
최종 420초 전체 경기(새 시드 64개 × 양 진영, 128경기)에서 **V19 대 V18은 92승 36패(72%), 평균 점수차
+25.7**이었다(보초 추가 전 구성은 77승 51패, +5.3). 원시 결과는 `reports/gauntlet_v19_vs_v18_full/`에 있다.

## V4 근접 변형군

`V4PolicyFamily`는 아래 profile 하나를 샘플링한다 (기본은 매치 시작 시 1회, `resample_each_absorption=True`로
생성하면 흡수마다 다시 샘플링). 매 tick noise는 사용하지 않는다.

| profile | 기본 확률 | `replan_interval` | `threat_radius` | `protected_storage_bonus` | 성격 |
|---|---:|---:|---:|---:|---|
| `exact` | 25% | 10 | 0.16 | 7.0 | V4와 동일 |
| `balanced` | 35% | 9–11 | 0.150–0.170 | 6.5–7.5 | V4 중심의 연속형 미세 변형 |
| `responsive` | 20% | 8–9 | 0.145–0.165 | 5.8–6.8 | 더 자주 재계획하고 가까운 창고에 민감 |
| `cautious` | 20% | 10–12 | 0.165–0.185 | 7.3–8.4 | 위협 반경과 보호 창고 선호가 큼 |

단독 사용:

```python
from blackout_env import V4PolicyFamily

teacher = V4PolicyFamily(seed=20260912)
sample = teacher.current_sample
print(sample.profile, sample.policy_seed, sample.parameters)
actions = teacher.act(observations)
```

특정 profile만 강제하려면 해당 profile의 가중치만 1로 설정한다.

```python
teacher = V4PolicyFamily(
    seed=20260912,
    profile_weights={"cautious": 1.0},
)
```

## 기본 정책 혼합

`HeuristicPolicyMixture`의 기본 분포는 다음과 같다. 입력 가중치는 합이 1일 필요가 없으며 내부에서
정규화된다. 0은 허용되지만 음수와 전체 합 0은 허용되지 않는다.

2026-09-16에 `examples/elo_active.py`의 64초 근사 적응형 Bradley-Terry 적합으로 다시 조정했다
(현재 코드, 보호 스폰 창고 가드 수정 포함). 20개 정책 전체는 `reports/elo_active_20260916/`(대진 159/190쌍,
728경기, 표준오차 42–65), 중간층만은 `reports/elo_active_mid_20260916/`(78/78쌍, 1,324경기, 표준오차 23,
두 정책 차이는 ~65 Elo 이상이어야 유의)에 있다. 이전 16정책 Elo(`heuristic_draw_census_20260915_130356`)는
가드 수정 전 코드였다.

조정 원칙:

- **V17–V19에 합계 32%.** 적 본진 출구 진치기·성소 소탕 같은 차단 플레이는 V1–V16 경기에 전혀 나오지 않고,
  V1–V16은 Hunter를 피하지도 않는다. 이들을 넣지 않으면 학습기는 그런 상태를 보지 못한다. V1–V16 상대로
  가장 강한 V17(151승 9패)을 가장 높게 두고, 카운터인 V18/V19는 그다음으로 둔다.
- **중간층은 거의 평평하게.** 중간층 안의 Elo 차이는 대부분 오차 범위다. 실전 기준/평가 정책인 `v4`/`v4_near`만
  단일 항목으로 가장 크게 유지한다.
- **`v7`/`v8`/`v9`/`v12`는 하나의 몫을 나눠 쓴다.** 행동이 거의 같아서(v8/v9의 respec은 드물게 발동) 각각 온전한
  몫을 주면 한 행동이 과대표집된다.
- **`v5`/`v2`/`v14`/`v1`은 대조군 최소치.** 네 정책이 뚜렷한 최하위다.
- V17–V19는 틱당 연산이 더 커서(팀당 약 300–400µs, 다른 정책은 ~200µs) 수집 속도가 조금 떨어진다.

| policy_id | 기본 가중치 | Elo (20정책 전체) | 중간층 단독 Elo | 데이터 내 역할 |
|---|---:|---:|---:|---|
| `strategic_v17` | 14% | 1832 | — | 차단 중심 팀 planner (V1–V16 상대 최강) |
| `strategic_v18` | 9% | 1831 | — | V17 전용 카운터 |
| `strategic_v19` | 9% | 2059 | — | V18 전용 카운터 |
| `strategic_v4` | 11% | 1500 | 1538 | 주 교사·평가 기준 |
| `strategic_v4_near` | 8% | 1462 | 1489 | V4 주변 조밀한 변형 |
| `strategic_v3` | 6% | 1456 | 1499 | 안전 적재 |
| `strategic_v13` | 6% | 1472 | 1504 | 지속 공성 약탈 |
| `strategic_v10` | 4% | 1472 | 1471 | 국면별 역할/목표 전환 |
| `strategic_v11` | 4% | 1443 | 1478 | 흡수 직전 다중-unit 약탈 |
| `strategic_v15` | 4% | 1482 | 1456 | Carrier 처리량 러시 |
| `strategic_v16` | 4% | 1490 | 1461 | V10 기반 조건부 공성·수비·수송 전환 |
| `strategic_v7` | 3% | 1480 | 1527 | 동적 역할 배분 |
| `strategic_v8` | 3% | 1529 | 1512 | 보수적 역할 리셋 (v7과 대부분 중복) |
| `strategic_v9` | 3% | 1519 | 1534 | 실제 역할 리셋 trajectory |
| `strategic_v12` | 3% | 1542 | 1559 | 리드 보존·창고 방어 |
| `strategic_v6` | 3% | 1390 | 1472 | 위험 회피·저정체 경로 |
| `strategic_v2` | 2% | 1314 | — | 전역 경제 배정 |
| `strategic_v5` | 2% | 1317 | — | 공격적 예측 요격 |
| `strategic_v1` | 1% | 1192 | — | 단순 기준 행동 (최약체, 대조군) |
| `strategic_v14` | 1% | 1220 | — | 약탈 대응 홈 수비 |

두 Elo 열은 상대 풀이 달라 척도가 다르다. 서로 직접 비교하지 않는다.

```python
from blackout_env import HeuristicPolicyMixture

teacher = HeuristicPolicyMixture(seed=20260912)

# 생성 직후 이미 첫 에피소드 정책이 샘플링되어 있다.
sample = teacher.current_sample
print(sample.policy_id)
print(sample.policy_seed)
print(sample.parameters)

actions = teacher.act(observations)
```

`reset()`을 명시적으로 호출하면 새 `policy_id`를 샘플링하고 새 `PolicySample`을 반환한다. `act()`는 Unity의
남은 매치 시간이 크게 증가하는 것도 새 매치로 자동 감지해 같은 방식으로 재샘플링하지만, 데이터 수집기는
첫 transition 전에 provenance를 기록할 수 있도록 명시적 `reset()`을 권장한다.

`resample_each_absorption=True`(기본값)이면 `act()`가 흡수 경계(`team_state[3]`가 증가하는 tick)도
감지해 `policy_id`와 정책 인스턴스는 그대로 둔 채 파라미터만 다시 샘플링해 제자리에서 적용한다 — `current_sample.policy_seed`와
`parameters`가 흡수마다 바뀔 수 있으므로, 흡수 단위로 provenance를 남기려면 매 흡수 경계 직후
`current_sample`을 다시 읽어야 한다.

```python
sample = teacher.reset()
episode_metadata["behavior_policy"] = {
    "policy_id": sample.policy_id,
    "policy_seed": sample.policy_seed,
    "parameters": dict(sample.parameters),
}
```

## 목적별 권장 혼합 예시

### 1. 안정적인 BC 초기 교사

V4와 그 주변을 80%로 두고 다른 경제·역할 행동을 조금만 섞는다.

```python
teacher = HeuristicPolicyMixture(
    seed=7,
    weights={
        "strategic_v3": 0.10,
        "strategic_v4": 0.50,
        "strategic_v4_near": 0.30,
        "strategic_v7": 0.08,
        "strategic_v9": 0.02,
    },
)
```

### 2. Offline RL용 넓은 support

직접 대전 성능이 낮더라도 공격·회피·역할 리셋 상태를 충분히 남긴다.

```python
teacher = HeuristicPolicyMixture(
    seed=7,
    weights={
        "strategic_v1": 0.07,
        "strategic_v2": 0.07,
        "strategic_v3": 0.09,
        "strategic_v4": 0.20,
        "strategic_v4_near": 0.17,
        "strategic_v5": 0.07,
        "strategic_v6": 0.08,
        "strategic_v7": 0.07,
        "strategic_v8": 0.02,
        "strategic_v9": 0.04,
        "strategic_v10": 0.05,
        "strategic_v11": 0.04,
        "strategic_v12": 0.03,
    },
)
```

### 3. 역할 정책 집중 수집

```python
teacher = HeuristicPolicyMixture(
    seed=7,
    weights={
        "strategic_v4": 0.20,
        "strategic_v7": 0.40,
        "strategic_v8": 0.10,
        "strategic_v9": 0.30,
    },
)
```

### 4. 특정 정책 또는 ablation 강제

```python
# V6만 수집
teacher = HeuristicPolicyMixture(
    seed=7,
    weights={"strategic_v6": 1.0},
    perturb=False,
)

# 레지스트리에서 직접 생성
from blackout_env.heuristics import make_heuristic
teacher = make_heuristic("strategic_v9")
```

## `perturb`의 의미

`perturb=True`이면 일반 정책에 다음 파라미터를 샘플링한다 (기본은 흡수 단위 재샘플링,
`resample_each_absorption=False`면 매치 단위).

- `use_specialists`: 92% 확률로 true (역할 체계를 바꾸므로 흡수 재샘플링과 무관하게 매치 단위로 고정)
- `replan_interval`: 8–13틱
- `threat_radius`: 0.135–0.185
- V5 추가: `intercept_margin_seconds` 0.20–0.40,
  `intent_temperature` 2.5–6.0, `velocity_ema` 0.35–0.75
- V6 추가: `risk_radius_tiles` 2.0–3.5, `risk_weight` 2.0–4.2
- V10 추가: 모드 확인 지연 14–22틱, 약탈 압박 흡수 임계 0.32–0.48,
  리드 보존 점수차 0.07–0.12
- V11 추가: 모드 확인 지연 8–16틱, 동시 약탈조 1–2기, 약탈 흡수 임계 0.42–0.66
- V12 추가: 모드 확인 지연 15–24틱, fortress 진입 리드 0.05–0.11
- V13 추가: 모드 확인 지연 6–14틱, 약탈조 2–3기, 공성 최소 가치 0.5–2.5
- V14 추가: 홈 수비 반경 5.5–8.5 타일
- V15 추가: Carrier가 고가 필드 배터리를 얼마나 강하게 우선할지 15.0–26.0
- V16 추가: 모드 확인 지연 8–17틱, 공성 전환 상대 창고 가치 5.0–11.0,
  홈 수비 반경 5.5–8.5, convoy 전환 필드 배터리 가치 40.0–70.0
- `strategic_v4_near`: 일반 변형 대신 V4 profile을 중첩 샘플링
- 그 밖의 정책(V1–V4, V7–V9, V17–V19)은 위의 공통 세 파라미터만 변형한다.

`perturb=False`이면 정책 버전은 계속 가중치에 따라 샘플링하지만 공통 파라미터는
`use_specialists=True`, `replan_interval=10`, `threat_radius=0.16`으로 고정되고 V4-near는 exact만
사용한다.

주의: V7–V10/V12/V14–V16는 `_role()`을 동적 역할 배분으로 재정의하고, V17–V19는 자체 역할 배정
(`carrier_quota`/`hunter_quota`)을 쓰므로 일반 혼합의 `use_specialists=False`가 역할을 끄는 스위치로
동작하지 않는다(V17–V19에서는 아예 쓰이지 않는다). 이 정책들의 역할을 끄거나 수량을 조절하려면 직접 생성해
`carrier_quota`, `hunter_quota`를 설정하거나 해당 정책을 혼합에서 제외한다.

## 상태 전환 정책(V10–V12, V16)의 사용법

이 정책들(그리고 V11을 상속한 V13)은 매 tick 무작위로 역할을 바꾸지 않는다. 관측된 조건이 연속 `mode_confirm_ticks`번 유지될
때만 새 모드로 전이하는 hysteresis를 사용한다. 조건은 모두 공개 `team_state`와 semantic map에서만
계산한다. 즉 BC 교사나 대회 정책으로 써도 특권 정보 누출이 없다.

| 정책 | 모드 | 공개 전이 신호 | unit별 결과 |
|---|---|---|---|
| V10 | `opening_economy` | 기본 초반 | 새 Hunter 변신을 늦추고 경제/Carrier를 우선 |
| V10 | `pressure_raid` | 상대 외부 창고에 가치가 있고 흡수 임박, 또는 크게 열세 | Hunter가 고가 화물/Carrier를 우선 추격하고 화물이 없으면 외부 창고를 순찰 |
| V10 | `closeout_defend` | 리드 상태 + 후반 | Hunter가 빈 Collector를 추격하지 않고 아군 창고를 순찰; 적 화물이 보일 때만 요격 |
| V16 | `storage_siege` | 보호되지 않은 적 외부 창고 가치가 `siege_value` 이상이고 남은 시간 20% 이상 | V13처럼 최대 3기 약탈조를 쓴다. 모드 우선순위는 `home_guard` > `storage_siege` > `convoy_rush` > V10의 세 모드 |
| V16 | `home_guard` | 적 화물이 아군 창고 `guard_radius` 안으로 진입 | V14처럼 Hunter가 그 화물만 요격하고, 화물이 없으면 아군 창고를 순찰 |
| V16 | `convoy_rush` | 초반 + 필드 배터리 가치가 `convoy_field_battery` 이상 | V15처럼 Hunter 변신을 미루고 Carrier가 고가 필드 배터리를 우선 운반 |
| V11 | `harvest` | 기본 | V4의 전역 경제 배정 |
| V11 | `raid_window` | 상대 외부 창고 가치 + 흡수 임박 또는 열세 | 최대 `raid_slots` Collector가 서로 다른 고가 적 창고 배터리를 노림 |
| V12 | `catch_up` | 기본/열세 | V7의 동적 역할 배분과 공격적 회복 |
| V12 | `fortress` | 리드 상태 + 경기 중후반 | 새 Hunter를 만들지 않고, 기존 Hunter는 아군 창고 7타일 안의 적 화물만 요격 |

각 인스턴스에는 현재 전략을 나타내는 `current_mode`, 변경 시점 목록 `strategy_transitions`, 그리고
`"pressure_raid:hunter"`처럼 모드가 접두사로 붙은 `role_assignments`가 있다. 데이터 수집기는 이를
에피소드 메타데이터뿐 아니라 모드 전환 transition에 함께 기록하는 편이 좋다.

```python
from blackout_env.heuristics import make_heuristic

teacher = make_heuristic("strategic_v10", mode_confirm_ticks=18)
actions = teacher.act(observations)
episode_metadata["strategy_mode"] = teacher.current_mode
episode_metadata["strategy_transitions"] = list(teacher.strategy_transitions)
```

## 전략적 사망과 역할 리셋

게임에서 Hunter/Carrier가 Collector로 돌아오는 유일한 방법은 사망 후 부활이다. V8/V9는 이 규칙을
명시적인 상태기계로 사용한다.

```text
HUNTER_ACTIVE
  └─ 조건 충족 → RESPEC_SEEK_ENEMY_HUNTER
                    ├─ 적 Hunter 소실/보호구역 진입 → HUNTER_ACTIVE
                    └─ 상호사망 및 Collector 관측 → COLLECTOR_COOLDOWN
                                                    └─ 쿨다운 종료 → 정상 동적 역할 배분
```

공통 안전 조건:

- 팀이 열세다.
- 실제로 수집할 필드 배터리와 회수할 경기 시간이 남아 있다.
- 일정 기간 고가치 적 수송이 없다.
- 아군과 적 Hunter가 모두 존재한다.
- 적 Hunter가 보호 스폰 밖에 있다.
- 경제 유닛을 대신 공격해 자살하지 않고 Hunter 상호사망만 노린다.

고가치 수송은 Carrier의 모든 화물, 5점 이상 배터리, 모든 특수 아이템이다. V9의 공개 진단 필드는
다음과 같다.

- `respec_attempts`: 역할 리셋 추격을 시작한 횟수
- `respec_completions`: Collector 부활까지 관측한 횟수
- `respec_diagnostics["max_inactive_ticks"]`: 고가치 수송 부재 최장 길이
- `respec_diagnostics["trailing_ticks"]`: 열세였던 decision tick 수
- `respec_diagnostics["both_hunters_ticks"]`: 양쪽 Hunter가 함께 존재한 tick 수
- `respec_diagnostics["all_gates_ticks"]`: 모든 리셋 조건이 동시에 열린 tick 수

## 데이터셋 provenance 권장 필드

최소한 에피소드 단위로 다음을 저장한다.

```python
episode_metadata = {
    "policy_id": sample.policy_id,
    "policy_seed": sample.policy_seed,
    "policy_parameters": dict(sample.parameters),
    "environment_seed": env_seed,
    "physical_team": physical_team,
    "spawn_side": spawn_side,
}
```

가능하면 transition/segment에 다음도 저장한다.

- 물리 unit index와 팀 관점 agent index
- `role_assignments`와 목표 종류
- 배터리/특수 아이템 보유 상태
- 흡수 구간 index와 흡수까지 남은 시간
- pickup, deposit, steal, death, transform, respec 이벤트
- 정지/막힘 진단값
- terminal winner, 실제 점수, 점수차
- V8/V9 역할 리셋 카운터와 진단 필드

동일 정책의 성공·실패 궤적을 모두 남기되, 학습 sample weight는 terminal 결과, 화물 보존, 정체와
역할 리셋의 경제적 회수 여부를 이용해 별도로 조절한다. 정책 ID만으로 expert 여부를 결정하지 않는다.

## 평가와 화면 실행

최소 승격 평가는 5개 이상의 seed를 양 진영에서 실행한다.

```bash
# 저장소 루트(blackout-env/)에서
./.venv/bin/python examples/benchmark_heuristics.py \
  --candidate v9 \
  --baseline v7 \
  --n-seeds 5 \
  --seed-rng 20260919 \
  --time-scale 200
```

출력에서 다음을 함께 확인한다.

- W-L-D와 평균 점수차
- seed별 양 진영 평균 점수차
- `idle_6s`
- `blocked_0.24s`
- V8/V9의 `respec=완료/시도`, `gates=조건 충족 tick`
- V10–V12의 `modes`, `switches`, 마지막 `strategy diversity` 요약(모드 점유율과 전환 횟수)

화면에서 보려면 배속을 1로 낮춘다.

```bash
./.venv/bin/python examples/benchmark_heuristics.py \
  --candidate v9 \
  --baseline v4 \
  --seeds 101 202 303 404 505 \
  --graphics \
  --time-scale 1
```

여러 Codex/터미널 세션에서 Unity 평가를 동시에 실행하면 관측 경계와 성능 측정이 흔들릴 수 있다.
평가 전 `benchmark_heuristics.py`, `evaluate_reward_*`, Unity player 프로세스가 이미 실행 중인지 확인하고
가능하면 한 Unity 평가만 단독 실행한다.

### 정책 상성 히트맵

모든 등록 정책의 행(row) 대 열(column) 승률을 PNG로 저장하려면 다음 명령을 사용한다. 각 비대각
셀은 같은 map seed에서 양쪽 진영을 한 번씩 교대하므로, 색은 행 정책의 side-swapped 승률이다.

```bash
./.venv/bin/python examples/tournament_heuristics.py \
  --n-seeds 5 \
  --seed-rng 20260913 \
  --workers 4 \
  --time-scale 200 \
  --output-dir reports/heuristic_tournament_20260913
```

완료 시 `reports/heuristic_tournament_20260913/win_rate_heatmap.png`에 히트맵을 저장하고,
동일 폴더에 재분석 가능한 `pair_results.csv`, `tournament.json`도 함께 저장한다. 기본 전체 정책군은
20개(`v1`–`v19`, `v4-near`)이므로 190개 비대각 쌍 × 5 seed × 양 진영 = 1,900경기다. 전체 정책 Elo만
필요하면 훨씬 적은 경기로 끝나는 `examples/elo_active.py`를 쓴다. `--policies v4 v7 v10 v11 v12`처럼
부분군을 먼저 확인한 뒤 전체를 돌릴 수도 있다.

## 재현성과 성능 주의사항

- 동일 mixture seed는 동일한 정책/파라미터 샘플열을 만든다.
- 환경 seed와 policy seed는 역할이 다르므로 둘 다 기록한다.
- 고배속 물리 실행은 일부 접촉 순서가 달라질 수 있으므로 반드시 진영 교대와 paired margin을 사용한다.
- 200배속에서는 decision observation 사이에 빈 관측 프레임이 생길 수 있다. 공식 benchmark는 pending
  transition을 다음 관측과 연결해 이동 신뢰성을 측정한다.
- Python 최적화는 정적 지도만 에피소드 캐시하고 배터리·특수 아이템은 decision tick마다 다시 읽는다.
- 최적화 전후 3,000틱 action byte hash가 일치했으며 warm 합성 측정은 팀당 약 186µs/tick이다.
- 정책 성능 개선과 실행 속도 개선은 별도로 검증한다. 캐시 변경도 action trace 회귀를 통과해야 한다.

## 새 정책 추가 체크리스트

1. 기존 클래스를 변경해 의미를 덮어쓰기보다 새 버전 클래스로 상속한다.
2. `POLICY_REGISTRY`, package export(`heuristics/__init__.py`), `HeuristicPolicyMixture`의 기본 가중치,
   `examples/tournament_heuristics.py`의 `POLICIES`, `train/policy_strength.py`의 Elo 표와 이 문서에 새 ID를
   추가한다. `POLICY_IDS`는 Elo 표 키를 (길이, 이름)으로 정렬한 순서라 새 ID가 기존 인덱스를 밀 수 있으므로,
   이미 저장된 데이터셋의 policy index와 호환되는지 확인한다.
3. 상태 메모리가 있으면 `reset()`에서 전부 초기화한다.
4. 공개 관측만 사용하고 Unity privileged state를 읽지 않는다.
5. 최소 5 seed × 양 진영 대전을 기존 권장판 또는 직접 부모와 수행한다.
6. 승패·점수차 외에 `idle_6s`, `blocked_0.24s`를 확인한다.
7. 역할/예측 정책은 고유 이벤트의 시도·완료율도 기록한다.
8. BC/offline-RL 가치는 직접 승률뿐 아니라 새롭고 일관된 상태-action support로 평가한다.
9. 기본 mixture 가중치는 총합과 희귀 정책의 과대표집 여부를 확인한다.
10. 정책 ID, 파라미터 범위와 평가 결과를 이 카탈로그에 갱신한다.
