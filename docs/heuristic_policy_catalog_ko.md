# BlackOut 휴리스틱 정책 카탈로그와 혼합 가이드

이 문서는 BlackOut의 BC(Behavior Cloning), offline RL, 온라인 강화학습 부트스트랩에 사용할 수 있는
휴리스틱 정책들의 차이, 검증 결과, 혼합 방법과 데이터 기록 규약을 한곳에 정리한다. 구현 세부와
이동 복구 원리는 `heuristic_policy_ko.md`를 함께 참고한다.

## 빠른 결론

- 실전 기준 정책과 평가 상대는 `strategic_v4`를 권장한다.
- V4 주변의 조밀한 행동 분포가 필요하면 `strategic_v4_near` 또는 `V4PolicyFamily`를 사용한다.
- 역할 배정 다양성은 `strategic_v7`, 의도적인 역할 리셋 궤적은 `strategic_v9`에서 얻는다.
- `strategic_v5`, `strategic_v6`, `strategic_v8`은 성능 최적점이 아니라 각각 공격적 요격,
  저정체 위험 회피, 보수적 역할 리셋이라는 희귀 상태 분포를 제공한다.
- 한 매치(420초) 안에서는 `policy_id`(어떤 버전인지)를 바꾸거나 action noise를 넣지 않는다.
  `HeuristicPolicyMixture`는 매치 시작 시 `policy_id`를 한 번만 고른다.
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
StrategicHeuristicV1
└── StrategicHeuristicV2
    └── StrategicHeuristicV3
        └── StrategicHeuristicV4  ← 현재 권장 기준
            ├── StrategicHeuristicV5  (예측 요격)
            ├── StrategicHeuristicV6  (위험 비용 경로)
            └── StrategicHeuristicV7  (동적 역할 배분)
                └── StrategicHeuristicV8  (보수적 사망 리셋)
                    └── StrategicHeuristicV9  (적극적 사망 리셋)

StrategicHeuristicV10  ── 국면 전환: opening economy / pressure raid / closeout defend
StrategicHeuristicV11  ── 흡수 직전 외부 창고 약탈 창(raid-window) 전담
StrategicHeuristicV12  ── 점수 리드 시 fortress, 열세/초반에는 catch-up
StrategicHeuristicV13  ── 외부 창고가 보이면 지속하는 3기 공성 약탈
StrategicHeuristicV14  ── 초반 Hunter 1기로 아군 창고 주변만 방어하는 counter-raid sentinel
StrategicHeuristicV15  ── Hunter를 포기하고 Carrier 1기의 고가 필드 배터리 운송에 올인
StrategicHeuristicV16  ── V10 director + storage siege / home guard / convoy rush 선택
StrategicHeuristicV17  ── V6 이동 계층 위의 팀 단위 planner: Hunter 3기(1기 적 본진 출구 진치기 + 2기 추격)로 운반 차단
    └── StrategicHeuristicV18  ── V17 전용 카운터: 아이템을 피하는 변신 경로 + 우리 출구 경비 Hunter
        └── StrategicHeuristicV19  ── V18 전용 카운터: 성소 소탕 + 성소 보초 Hunter + 가까운 성소 칸 경로
    └── StrategicHeuristicV17Teach  ── V17의 편성·가치를 상태만의 함수로 라벨하는 증류 교사 (heuristics/v17_teach.py)
        └── StrategicHeuristicV20  ── V17-teach + 스폰 진치기 대응: 위험 구역 우회, 본진 대기 후 함께 돌파, 빈손 Collector 회피
            └── StrategicHeuristicV21  ── V20 + 안전한 경로가 있을 때만 Hunter 변신 (모든 적 Hunter 3칸을 피하는 경로)
    └── StrategicRaider·CargoHunter·EarlyRush·Fortress·Duelist  ── V17 planner 위의 보류 상대 (약탈 우선, 화물 요격, 초반 소탕 후 편성 전환, 지역 방어, Hunter 사냥)

V4PolicyFamily  ── V4의 작은 근접 파라미터 변형 (기본은 매치 단위, `resample_each_absorption=True`면 흡수 단위)
HeuristicPolicyMixture ── 위 policy_id와 V4PolicyFamily를 함께 샘플링
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
| `strategic_v5` | `StrategicHeuristicV5` | 적 위치 EMA와 가능한 저장소 경로 ensemble로 Hunter 요격점 예측 | 미래 위치를 향하는 공격적 전투 샘플 | V4 대비 직접 대전 성능과 막힘이 나빠 기본 비중 3% |
| `strategic_v6` | `StrategicHeuristicV6` | 운반자 A* 비용에 적 위협장을 통합 | 벽/막힌 길로 회피하는 비율이 낮고 저정체 경로 제공 | 우회가 길어져 점수 성능이 낮을 수 있음 |
| `strategic_v7` | `StrategicHeuristicV7` | Carrier 최대 1기; 성소까지 실제 경로가 가까운 빈 Collector 배정; Hunter 투입 지연 | 역할과 운반 능력을 명시적으로 고려 | V4보다 짧은 막힘이 많아 기준판 대신 역할 변주용 |
| `strategic_v8` | `StrategicHeuristicV8` | 열세·수송 부재·필드 자원·시간·적 Hunter 조건을 모두 만족할 때 상호사망으로 역할 리셋 | 불필요한 자살을 강하게 억제 | 10경기에서 리셋 0회; 보수적 대조군, 기본 1% |
| `strategic_v9` | `StrategicHeuristicV9` | V8 조건을 완화해 실제 Hunter→사망→Collector trajectory 생성 | 리셋 4/4 성공, V7과 동률 성능 및 비슷한 신뢰성 | 승격판은 아니며 희귀 전략 데이터용 기본 3% |
| `strategic_v10` | `StrategicHeuristicV10` | 공개 점수·시간·흡수·상대 외부창고 가치로 `opening_economy`→`pressure_raid`→`closeout_defend` 전환; Hunter의 목표도 국면별로 변경 | 같은 지도에서도 경제 확장, 약탈 압박, 리드 보호 궤적을 모두 제공 | 새 다양성 정책; V4와의 대전·신뢰성 평가는 별도 기록 후 비중 조정 |
| `strategic_v11` | `StrategicHeuristicV11` | 흡수 직전 또는 열세일 때 최대 2기 Collector를 상대 외부 창고의 고가 배터리에 배정 | 약탈 타이밍과 다중-unit 협공 상태를 의도적으로 많이 생성 | 정상 수집보다 약탈에 치우친 policy support; 교사 주력으로는 사용하지 않음 |
| `strategic_v12` | `StrategicHeuristicV12` | 리드 후반에는 새 Hunter 변신을 늦추고, 기존 Hunter가 빈 적을 추격하지 않고 아군 창고 방어 | 수비/호위와 공격 포기라는 명확한 counterfactual 상태를 제공 | 상대가 적극적으로 역전할 때 기회비용이 생길 수 있음 |
| `strategic_v13` | `StrategicHeuristicV13` | 흡수 임박 여부와 무관하게 노출된 상대 외부 창고에 최대 3기 약탈조를 유지 | 수비형·창고 보존형 상대로 반복 약탈과 다중-unit 압박 trajectory를 생성 | Hunter 수비나 빠른 필드 경제에 취약하도록 의도된 고위험 정책 |
| `strategic_v14` | `StrategicHeuristicV14` | 초반부터 Hunter 1기를 만들고, 적 화물이 아군 창고 반경 안에 들어올 때만 요격; 그 외에는 홈 순찰 | V13/V11 같은 약탈형을 상대로 한 home-defense와 반격 관측을 제공 | 공격·운반 인력 하나를 고정 소비하므로 수동적 경제형에게 점수 손해 가능 |
| `strategic_v15` | `StrategicHeuristicV15` | Carrier 1기와 Hunter 0기를 고정하고 Carrier가 고가 필드 배터리를 우선 운반 | 약탈 대신 처리량을 택하는 장거리 convoy trajectory를 제공 | 적 Hunter·약탈을 막지 못하므로 전투형 정책과 뚜렷한 상성 차이를 만들 수 있음 |
| `strategic_v16` | `StrategicHeuristicV16` | V10의 `opening_economy`·`pressure_raid`·`closeout_defend`에, 공개 창고 가치/위협/필드 자원으로 `storage_siege`·`home_guard`·`convoy_rush`를 추가 | 하나의 에피소드 안에서 조건부 대전략 전환을 관측할 수 있어 장기 horizon BC와 offline RL에 유용 | 각 모드의 조건이 드문 지도에서는 V10과 유사하게 보일 수 있으므로 모드 provenance를 반드시 저장 |
| `strategic_v4_near` | `V4PolicyFamily` | V4 exact와 세 종류의 좁은 파라미터 변형을 에피소드 단위 샘플링 | V4 주변 decision boundary를 조밀하게 커버 | 완전히 다른 전략 상태는 거의 만들지 않음 |

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

따라서 기본 혼합에서는 V14와 V16을 각각 3%, 4%로 유지하고, V13/V15는 약점과 반례를 충분히
수집하기 위한 3% 희귀 정책으로 유지한다. 이 결과는 성능만으로 V13/V15를 제거하지 않는 근거이기도
하다. 완성된 원시 결과와 히트맵은 `reports/heuristic_tournament_all17_workers18_20260913/`에 보관한다.

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
| 역할 | Carrier 1기, Hunter 3기, Collector 1기 (`carrier_quota=1`, `hunter_quota=3`) |
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
- V17 진치기 Hunter는 적 Hunter를 공격하지 않고, 추격 Hunter도 자기 스폰 7칸 밖의 적 Hunter는 무시한다.

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
| 성소 소탕 | 모든 Hunter가 자기 자리로 가기 전에 성소 `sweep_radius`(9칸) 안의 적 Collector부터 처치 |
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

## V20: 스폰 진치기 대응 (2026-09-29)

`heuristics/v20_anticamp.py`. 증류 교사 V17-teach(`StrategicHeuristicV20Teach`, `--teacher v20-teach`)이자 그 라벨을
따라 움직이는 휴리스틱(`strategic_v20`)이다. mixture 기본 가중치에는 넣지 않았다(기존 수집 재현).

### 문제

적은 우리 본진에 못 들어오므로 출구 바로 밖의 Hunter 하나가 나오는 Collector·Carrier를 전부 잡는다(Hunter는 접촉만으로
둘 다 처치하고, 속도 6 대 4라 쫓기면 못 빠져나간다). V17-teach로 V19를 상대하면 1000 유닛틱당 8.16기가 죽었고(V17 상대
0.98), 그 81%가 본진 4칸 안, 경기의 32% 동안 출구가 막혀 있었다 — V19 상대 −38.9의 원인.

### 동작 (적 Hunter가 우리 본진 3칸 안에 있을 때, Collector·Carrier만)

| 구성 요소 | 내용 |
|---|---|
| 우회 | 진치는 Hunter 3칸 안 칸(우리 본진 제외)을 피해 목표에 갈 수 있으면 그 경로 |
| 대기 후 함께 돌파 | 본진 안에서 안전 경로가 없는 유닛은 혼자면 본진에서 대기, 2기 이상이면 동시에 출발 — Hunter는 하나만 쫓는다 |
| 추격 회피 | V4의 위협 반발을 빈손 Collector에도 적용(V17-teach는 짐·Carrier만) |
| 나머지 | V17-teach 그대로. 우리 Hunter는 이미 스폰 진치기 적 Hunter를 최우선 교환 대상으로 본다(V17 가치 12) |

### 평가 (교사가 직접 조종, 보류 시드 900000–900011 × 양 진영 = 상대당 24경기, `reports/anticamp/`)

| 상대 | V17-teach | V20 | 짝 차이 (V20 − V17-teach) | 사망/1000틱 | 본진 근처 사망 |
|---|---:|---:|---:|---|---|
| V19 | −38.9 | −23.3 | **+15.6 ± 6.7** | 8.16 → **3.75** | 81% → 55% |
| V17 | +17.8 | +14.7 | −3.1 ± 13.7 | 0.98 → 0.63 | 66% → 44% |
| V4 | +49.8 | +48.5 | −1.3 ± 5.0 | 1.08 → 0.96 | 25% → 18% |

V19 상대만 좋아지고(2.3 SE) 나머지는 그대로. V17 도입 때 "Hunter 주변을 막는 이동 변형은 20–27% 손해"였던 것과 달리,
출구가 막혔을 때만 켜지는 규칙이라 경제를 멈추지 않는다.

## V21: 안전한 변신만 (2026-09-30)

`heuristics/v21_safe_transform.py`, 교사 `--teacher v21-teach`, 휴리스틱 `strategic_v21`(mixture 기본 가중치에는 없음).

### 문제 (V20 교사 대 V19, 보류 시드 24경기의 샤드 분석, `reports/v19_analysis/`)

V17-teach의 편성 라벨은 무엇이 가로막든 Hunter 3기를 채운다. V19는 Hunter 4기(우리 출구 1, 중앙 성소 보초 1, 성소 주변
소탕 2)를 둬서, 성소로 가는 변신 시도 2,491회 중 2%만 Hunter가 되고 17%가 가는 길에 죽었다. 경기당 사망 35회의 70%가
변신하러 가던 중이었고, Hunter는 평균 0.5기 — 경제 유닛을 성과 없이 소모했다.

### 동작

Hunter 변신 후보는 **모든 적 Hunter(자기 본진 밖) 3칸 안을 피해 성소까지 가는 경로가 있는 Collector만**이고, 그 경로로
간다(우리 본진은 위험 구역에서 제외). 없으면 Collector로 남아 경제를 한다. Carrier 성소는 본진 안이라 그대로. 나머지는 V20.

### 평가 (교사 직접 조종, 보류 시드 × 양 진영, 상대당 24경기, `reports/anticamp/`)

| 상대 | V17-teach | V20 | **V21** | V21 − V20 (짝) |
|---|---:|---:|---:|---:|
| V19 | −38.9 | −23.3 | **+3.4** (승 46%) | **+26.7 ± 10.2** |
| V17 | +17.8 | +14.7 | +12.8 | −1.9 ± 7.6 |
| V4 | +49.8 | +48.5 | +49.7 | +1.2 ± 6.7 |

V19 상대: 변신 중 사망 70% → 0%, 변신 성공 2% → 6%(사망 0%), 평균 Hunter 0.50 → 1.04, 사망/1000틱 3.75 → 2.44.
V21 − V17-teach는 V19 상대 +42.3 ± 9.3.

약탈 가치: V17 계열은 이미 상대 창고 배터리를 필드 배터리의 `steal_weight` 1.8배로 매기고(흡수 전 도착 가능할 때만),
할 일 없는 Collector는 상대 외부 창고 옆에서 대기한다.

## 새 상대 정책과 신경망식 실행 오류 래퍼 (2026-09-30)

학생이 교사(V21-teach)와 기존 상대 풀을 모두 넘어서서, 증류용 보류(held-out) 상대로 추가했다. 전부 `POLICY_REGISTRY`에
등록했지만 mixture 기본 가중치에는 넣지 않았다. 목표는 강도보다 **기존 정책군에 없는 전략 축**이다.

### 기존 정책의 행동군 (비교 대표 선정 근거)

V1–V19 44,732경기 결합 승률표(`reports/elo_augment_20260918/combined_tournament.json`)에서 두 정책의 "제3자 상대 승률
행"의 RMS 차이와 상호 승률을 비교했다. 쌍당 승률 SE ≤ 0.05라서, 같은 정책끼리라도 RMS 차이는 약 0.068까지 나온다(잡음 바닥).

| 행동군 | 정책 | 근거 | 비교 대표 |
|---|---|---|---|
| 중위권 V4 군 | v3, v4, v6, v7, v8, v9, v10, v11, v12, v13 | 모든 쌍의 행 차이 0.021–0.069 (잡음 바닥 이하), 상호 승률 0.44–0.57, Elo 1460–1510. 안에서 v7·v8·v12(0.021–0.031)와 v3·v11(0.034)이 특히 붙어 있다 | `strategic_v4` |
| V15 / V16 | 각각 단독 | 행은 비슷하지만(0.062) v16이 v15를 62% 이김. v4 대비 행 차이 0.11–0.13, v4에 59–68% 짐 | — |
| V2 / V5 | 각각 단독 | 행 차이 0.090, 상호 36:64 | — |
| V1, V14 | 각각 단독 | 최하위권, 다른 모든 정책과 행 차이 0.13 이상 | — |
| V17 | 단독 | V18/V19와 행 차이 0.128, 두 정책에 16–21% | `strategic_v17` |
| V18 / V19 | 각각 단독 | 제3자 행은 거의 같지만(0.034) V19가 V18을 69% 이김 — 카운터 관계라 합치지 않는다 | `strategic_v19` |

즉 "비슷한 계열"은 V4–V10+V16이 아니라 **v3·v4·v6–v13**이다(v5는 약한 별도 정책, v16은 v4에 68% 지는 별도 정책). V20/V21은
이 토너먼트에 없고 V17-teach 기반이다. 아래 비교는 V4 군을 `strategic_v4` 하나로 대표한다.

### 전략 정책 (모두 V17 planner 위, `heuristics/<모듈>.py`)

| ID | 모듈 | 전략 축 | 동작 |
|---|---|---|---|
| `strategic_raider` | `raider.py` | 약탈 우선 경제 | Collector 2기(`raid_squad`)는 필드 배터리를 아예 안 줍고 적 외부 창고 옆에서 대기·약탈, 나머지는 약탈 가치 3.5배(`steal_weight`), 필드 0.6배(`field_weight`). 적재는 보호 창고 우선(`exposure_cost` 2.0). Hunter 2기 중 1기는 약탈 중인 창고와 도둑 주변 5칸 안의 적(수비자·적 Hunter·적재하러 오는 화물)을 처치하는 호위 |
| `strategic_cargo_hunter` | `cargo_hunter.py` | 화물만 파괴 | Hunter 3기가 빈손 Collector와 적 Hunter(우리 스폰 진치기 제외)는 무시하고 화물·Carrier만 노림. 추격 대신 적의 가장 가까운 창고까지 최단 경로 위에서 우리가 먼저 닿는 첫 칸으로 요격(속도 6 Carrier도 잡음). 화물이 없으면 적 본진 출구와 적 외부 창고에서 매복 |
| `strategic_early_rush` | `early_rush.py` | 경기 중 편성 전환 | 첫 35초(`rush_seconds`)는 Hunter 4기, Carrier 0기로 적 Collector를 가까운 순으로 소탕(대상이 없으면 적 본진 출구). 이후 Hunter 1기 + Carrier 1기를 원하고, 남는 Hunter는 적 Hunter와 교환해 죽어 Collector로 돌아옴 — 편성을 되돌리는 유일한 정책 |
| `strategic_fortress` | `fortress.py` | 지역 방어 + 무위험 경제 | Hunter 2기가 우리 본진 출구와 우리 외부 창고(가치 순)에 서서 반경 6칸(`guard_radius`) 안만 교전. 적 Hunter 5칸(`danger_tiles`) 안의 작업은 아예 제외, 적재는 보호 창고(`exposure_cost` 3.0), 약탈 가치 1.0, 할 일 없는 유닛은 본진으로 귀환 |
| `strategic_duelist` | `duelist.py` | Hunter 사냥 | Hunter 2기가 맵 어디든 적 Hunter를 가까운 순으로 찾아가 맞교환(양쪽 사망 → 적은 다시 성소까지 걸어가야 함). 적 Hunter가 없으면 우리 화물/Carrier를 적 쪽에서 1.5칸 떨어져 호위하고 4칸 안의 적을 처치 |

### 신경망식 실행 오류 래퍼 (`train/distill/perturb.py`)

서술자 `perturb:<key=value,...>:<기본 서술자>` — 기본은 `self`를 뺀 모든 서술자(`strategic_v19`,
`teacher:v21-teach`, `ckpt:<경로>`, 중첩 `perturb:` 포함). 전략은 기본 정책 그대로, 실행만 망가뜨린다. 매치 seed로 시드되고
`reset()`마다 같은 난수열을 다시 쓴다. 적용 순서: lag → 정책 → sticky → dither → eps → freeze.

| 키 | 의미 |
|---|---|
| `eps=p` | 유닛·틱마다 확률 p로 무작위 8방향 |
| `dither=f` | 유닛틱의 f 비율을 `dither_ticks`(12)틱 역전 구간으로: `dither_period`(3)틱마다 정방향/역방향 교대(f/2가 역방향) |
| `sticky=k` | 유닛마다(무작위 위상) k틱에 한 번만 새 action, 그 사이 반복 |
| `lag=k` | k틱 전 팀 관측으로 행동 |
| `freeze=f` | 유닛틱의 f 비율을 `freeze_ticks`(50틱, 약 2초) 길이 정지 구간으로 |
| `seed=s` | 매치 seed에 더할 값 |

구간 시작 확률은 `f / (L·(1−f))`라서 장기 비율이 f다(테스트에서 확인). 예:
`perturb:eps=0.2:strategic_v21`, `perturb:sticky=4,lag=3:teacher:v21-teach`.

### 강도 확인 (V21 교사 직접 조종, 보류 시드 900000–900003 × 양 진영 = 상대당 8경기, `reports/new_opponents/`)

교사 관점 점수차(음수일수록 상대가 강함). 8경기라 ±SE가 크다 — 대략의 서열만 본다.

| 상대 | 교사 점수차 | 교사 승률 | 같은 시드 짝 차이 |
|---|---:|---:|---:|
| `strategic_v17` (참고) | +4.1 ± 19.8 | 62% | v4 대비 −32.5 ± 18.9 |
| `strategic_v19` (참고) | +23.9 ± 14.5 | 62% | v4 대비 −12.8 ± 1.8 |
| `strategic_v4` (V4 군 대표) | +36.6 ± 10.5 | 75% | — |
| `strategic_cargo_hunter` | +38.8 ± 12.5 | 88% | v4 대비 +2.1 ± 12.9 |
| `strategic_duelist` | +44.8 ± 14.9 | 88% | v4 대비 +8.1 ± 21.3 |
| `strategic_fortress` | +54.1 ± 8.0 | 100% | v4 대비 +17.5 ± 18.3 |
| `strategic_early_rush` | +58.9 ± 6.2 | 100% | v4 대비 +22.2 ± 9.4 |
| `strategic_raider` | +59.9 ± 6.1 | 100% | v4 대비 +23.2 ± 9.8 |
| `perturb:freeze=0.2:strategic_v19` | +24.6 ± 15.3 | 62% | v19 대비 +0.8 ± 20.0 |
| `perturb:dither=0.2:strategic_v19` | +48.5 ± 12.2 | 88% | v19 대비 +24.6 ± 23.8 |
| `perturb:sticky=6:strategic_v19` | +49.2 ± 9.3 | 88% | v19 대비 +25.4 ± 21.0 |
| `perturb:eps=0.2:strategic_v19` | +67.6 ± 5.4 | 100% | v19 대비 +43.8 ± 12.6 |
| `perturb:lag=6:strategic_v19` | +71.5 ± 8.4 | 100% | v19 대비 +47.6 ± 16.0 |
| `perturb:eps=0.1,dither=0.1,sticky=3,lag=3,freeze=0.1:strategic_v19` | +79.2 ± 3.0 | 100% | v19 대비 +55.4 ± 14.5 |

- 새 전략 5개는 모두 V17/V19보다 약하고 V4 군과 비슷하거나(cargo_hunter, duelist) 더 약하다(raider, early_rush, fortress).
  V17 계열의 강점인 적 본진 출구 진치기를 전부 버렸기 때문이다(교사 사망/1000틱: V19 상대 2.41, 새 전략 상대 0.38–0.51).
  V21 교사가 이미 Hunter를 피해 다니므로 소탕(early_rush)과 약탈(raider, 교사는 보호 창고 위주로 적재)이 잘 먹히지 않는다.
  목적은 다양성이라 강화 튜닝은 하지 않았다.
- 래퍼는 V19를 크게 약화한다. lag 6틱과 eps 0.2가 가장 치명적(V19의 요격·진치기가 늦거나 흩어짐)이고, freeze 0.2는
  영향이 거의 없다(Hunter가 서서 진치는 정책이라 멈춤이 곧 대기). 학생 평가에는 강한 기본 정책에 약한 오류(eps 0.05–0.1,
  lag 2–3)를 거는 편이 "강하지만 엉성한" 상대를 만든다.

```bash
./.venv/bin/python examples/distill_heldout_eval.py --teacher v21-teach --seeds 4 --workers 4 --out reports/new_opponents \
    --opponents strategic_v4 strategic_v17 strategic_v19 strategic_raider strategic_cargo_hunter strategic_early_rush \
    strategic_fortress strategic_duelist perturb:eps=0.2:strategic_v19 perturb:lag=6:strategic_v19 ...
```

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

`perturb=False`이면 정책 버전은 계속 가중치에 따라 샘플링하지만 공통 파라미터는
`use_specialists=True`, `replan_interval=10`, `threat_radius=0.16`으로 고정되고 V4-near는 exact만
사용한다.

주의: V7–V10/V12/V14–V16는 `_role()`을 동적 역할 배분으로 재정의하므로 일반 혼합의 `use_specialists=False`가
동적 역할을 끄는 스위치로 동작하지 않는다. 이 정책들의 역할을 끄거나 수량을 조절하려면 직접 생성해
`carrier_quota`, `hunter_quota`를 설정하거나 해당 정책을 혼합에서 제외한다.

## 상태 전환 정책(V10–V12, V16)의 사용법

세 정책은 매 tick 무작위로 역할을 바꾸지 않는다. 관측된 조건이 연속 `mode_confirm_ticks`번 유지될
때만 새 모드로 전이하는 hysteresis를 사용한다. 조건은 모두 공개 `team_state`와 semantic map에서만
계산한다. 즉 BC 교사나 대회 정책으로 써도 특권 정보 누출이 없다.

| 정책 | 모드 | 공개 전이 신호 | unit별 결과 |
|---|---|---|---|
| V10 | `opening_economy` | 기본 초반 | 새 Hunter 변신을 늦추고 경제/Carrier를 우선 |
| V10 | `pressure_raid` | 상대 외부 창고에 가치가 있고 흡수 임박, 또는 크게 열세 | Hunter가 고가 화물/Carrier를 우선 추격하고 화물이 없으면 외부 창고를 순찰 |
| V10 | `closeout_defend` | 리드 상태 + 후반 | Hunter가 빈 Collector를 추격하지 않고 아군 창고를 순찰; 적 화물이 보일 때만 요격 |
| V16 | `storage_siege` | 보호되지 않은 적 외부 창고 가치가 `siege_value` 이상 | V13처럼 최대 3기 약탈조를 쓰되, 수비 위협이나 초반 field-rush 조건이 오면 다음 모드로 전환 |
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
cd /Users/mac/project/26rl/blackout-env

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
17개이므로 136개 비대각 쌍 × 5 seed × 양 진영 = 1,360경기다. `--policies v4 v7 v10 v11 v12`처럼
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
2. `POLICY_REGISTRY`, package export, benchmark 선택지와 이 문서에 새 ID를 추가한다.
3. 상태 메모리가 있으면 `reset()`에서 전부 초기화한다.
4. 공개 관측만 사용하고 Unity privileged state를 읽지 않는다.
5. 최소 5 seed × 양 진영 대전을 기존 권장판 또는 직접 부모와 수행한다.
6. 승패·점수차 외에 `idle_6s`, `blocked_0.24s`를 확인한다.
7. 역할/예측 정책은 고유 이벤트의 시도·완료율도 기록한다.
8. BC/offline-RL 가치는 직접 승률뿐 아니라 새롭고 일관된 상태-action support로 평가한다.
9. 기본 mixture 가중치는 총합과 희귀 정책의 과대표집 여부를 확인한다.
10. 정책 ID, 파라미터 범위와 평가 결과를 이 카탈로그에 갱신한다.
