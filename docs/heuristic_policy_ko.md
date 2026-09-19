# BlackOut BC 부트스트랩용 전략 휴리스틱

> 이 문서는 V1(`strategic_v1`)의 설계 문서다. V2–V19와 `strategic_v4_near`의 차이, 현재 Elo·혼합 가중치,
> 평가 명령은 [`heuristic_policy_catalog_ko.md`](heuristic_policy_catalog_ko.md)를 본다.

## 목적과 인터페이스

`StrategicHeuristicV1`은 대회용 공개 관측만 사용하는 상태 보유형 팀 정책이다. Unity 내부 상태를
직접 읽지 않으므로 이 정책의 `(observation, action)`을 그대로 BC 데이터로 사용할 수 있다.
출력은 각 유닛의 연속 2D 이동 방향이며, 수집·적재·변신·전투는 게임의 접촉 규칙으로 자동 실행된다.

실제 활성 밸런스는 420초, 20초 흡수, 목표 100점, 팀당 외부 창고 4개와 보호 창고 1개다.
문서의 외부 창고 3개 설명보다 런타임 asset을 우선한다. 또한 현재 게임은 창고 점수가 흡수 전에
100점에 도달해도 즉시 끝나므로, 승리 임계치에 가까울수록 운반 완료가 최우선이다.

## 의사결정 계층

1. 팀 내 첫 유닛은 Carrier, 둘째는 Hunter, 나머지 셋은 Collector 역할을 맡는다.
2. 역할 유닛이 사망해 Collector로 부활하면 해당 성소로 다시 이동한다. 단, 이동 중 아이템을
   자동 습득했다면 변신으로 화물을 파괴하지 않고 먼저 아군 창고에 적재한다.
3. 아이템 보유자는 화물 전체를 받을 용량이 남은 아군 창고 중 가장 가까운 연결 영역의 중심으로 이동한다
   (창고는 아이템을 통째로만 받으므로). 모든 창고가 가득 차면 가장 가까운 창고 바로 옆 칸에서 다음 흡수를
   기다린다. 중심 진입은 경계 픽셀에서 멈춰 `OnUnitEnter`가 누락되는 현상을 막는다.
4. 빈 Collector/Carrier는 배터리 수량, 거리, 적 창고 여부, 다음 흡수 임박도를 합친 점수로 목표를
   선택한다. 팀원끼리 같은 픽셀을 중복 선택하지 않도록 목표를 예약한다.
5. 필드 배터리가 없으면 적 창고를 순찰·약탈한다. 적 창고 배터리는 흡수가 임박할수록 우선도가
   높아져 상대 확정을 차단한다.
6. Hunter는 화물 보유 적과 Carrier를 우선 추격하고, 그 다음 가까운 적을 차단한다. 적 본진(스폰 주변)
   안의 적은 닿을 수 없으므로 제외한다.
7. 배터리 운반자와 Carrier는 가까운 적 Hunter(그리고 Carrier에게는 모든 적)로부터 국소 반발력을
   받아 화물 소멸 위험을 낮춘다.
8. 필드 목표가 고갈되면 Collector/Carrier는 약탈 가능한 외부 적 창고들을 순환한다. 보이는 적이 없는
   Hunter는 가장 가까운 방어 가능한(본진 밖) 아군 창고 중심을 지킨다.

## 이동 계획

- semantic map의 wall channel을 이용해 8방향 A*를 수행한다.
- 대각선 코너 통과는 두 직교 인접 칸이 모두 통과 가능할 때만 허용한다.
- 목표/역할 변경, 일정 틱 경과, 움직이는 추격 목표, 정체 감지 시 경로를 다시 계산한다.
- A*가 연결 경로를 찾지 못하면 목표를 향해 직선으로 벽을 밀지 않고 현재 연결 영역의 통과 가능한
  이웃을 탐색한다.
- 연속 정체(12틱 정지 또는 짧은 진동)가 감지되면 유닛 index에 따라 반대 방향의 수직 이탈을 6틱 동안 주어
  유닛 간 대칭 충돌과 오목한 모서리 고정을 해소한다.
- pickup/deposit/transform 목표의 도착 상태가 24틱 동안 변하지 않으면 접촉 이벤트 누락으로 보고
  12틱 동안 영역 밖으로 이탈한 뒤 재진입한다.
- graphic row는 위에서 아래로 증가하지만 world y는 아래에서 위로 증가한다. 정책은 이 부호 차이를
  명시적으로 변환하며 Team B 좌표를 별도로 mirror하지 않는다.

## BC 데이터 권장 방식

정책은 교체하지 않고 누적 보존한다.

| policy_id | 클래스 | 주요 차이 |
|---|---|---|
| `strategic_v1` | `StrategicHeuristicV1` | 안정적인 역할·A*·정체 복구 기준선 |
| `strategic_v2` | `StrategicHeuristicV2` | 경로 거리 기반 전역 수집 작업 배정 |
| `strategic_v3` | `StrategicHeuristicV3` | V2 + 흡수 시각·매복 위험·보호 여부 기반 창고 선택 |
| `strategic_v4` | `StrategicHeuristicV4` | V3 + 동시 운반자의 창고 진입 타일 분산; 현재 권장판 |
| `strategic_v5` | `StrategicHeuristicV5` | V4 + 적 운반 경로 예측 요격; 공격적 저가중치 변주 |
| `strategic_v6` | `StrategicHeuristicV6` | V4 + 위협장 비용 A*; 저정체·보수적 변주 |
| `strategic_v4_near` | `V4PolicyFamily` | V4와 거의 같은 exact/balanced/responsive/cautious 근접 변형군 |
| `strategic_v7` | `StrategicHeuristicV7` | V4 + 클래스 제한·운반 능력·경기 국면을 반영한 동적 역할 배분 |
| `strategic_v8` | `StrategicHeuristicV8` | V7 + 엄격한 조건에서 Hunter 상호 사망을 이용한 역할 리셋 실험판 |
| `strategic_v9` | `StrategicHeuristicV9` | V8의 대기·점수·회수 조건을 완화한 적극적 역할 리셋 변주 |

V10–V19는 이 표 이후에 추가됐다(카탈로그 참조). 현재 가장 강한 정책은 V19/V17/V18이다.

`HeuristicPolicyMixture(seed=...)`는 매치마다 등록된 20개 버전 중 하나를 샘플링하고 `replan_interval`,
`threat_radius`, specialist 구성에 제한된 변주를 준다(수치 파라미터는 기본적으로 20초 흡수마다 다시 뽑는다). 데이터 수집기는 매 trajectory에
`mixture.current_sample`의 `policy_id`, `policy_seed`, `parameters`를 저장해야 한다. 같은 seed의
mixture는 동일한 정책열을 재현하므로 offline RL의 behavior-policy provenance와 중요도 가중에도
사용할 수 있다. 변주가 필요 없는 평가에서는 `perturb=False`를 사용한다.

기본 mixture 가중치는 2026-09-16 Elo 적합으로 정했다(V17 14%, V4 11%, V18/V19 각 9%, V5 2% 등; 전체 표는
카탈로그). V5에서는 요격 margin/의도 온도/속도 EMA, V6에서는 위험 반경/가중치도 함께 변주된다. 특정
ablation 데이터가 필요하면 `weights={"strategic_v6": 1.0}`처럼 명시한다. 매 tick action noise는
추가하지 않으므로 한 trajectory 안의 행동 의도는 일관된다.

V4 주변만 조밀하게 샘플링하려면 `V4PolicyFamily(seed=...)`를 직접 사용한다. 일반 mixture 안에서도
`strategic_v4_near`가 하나의 정책으로 샘플링되므로 버전 혼합과 근접 변형을 동시에 쓸 수 있다.
예를 들어 `weights={"strategic_v4": 0.4, "strategic_v4_near": 0.5,
"strategic_v7": 0.1}`은 검증된 V4 중심 분포에 역할 배분 변주를 소량 섞는다.

V7은 Carrier를 게임의 생존 최대치인 1기로 제한하고, 빈 Collector 중 성소까지의 실제 경로가 가장
짧은 유닛을 배정한다. Hunter는 적 운반 활동, 경기 국면, 남은 필드 배터리를 보고 뒤늦게 투입한다.
V8의 전략적 사망은 적 운반 활동이 장기간 없고, 팀이 열세이며, 수집할 배터리와 회수 시간이 남고,
보호 구역 밖 적 Hunter와 상호 사망할 수 있을 때만 시작한다. 부활 뒤에는 쿨다운 동안 Collector로
유지해 즉시 Hunter로 되돌아가는 루프를 막는다. 수집 데이터에는 `role_assignments`와 V8의
`respec_attempts`, `respec_completions`도 함께 기록하는 것이 좋다.
V8은 5시드 양 진영 평가에서 실제 리셋이 0회였다(행동이 V7과 거의 같아 현재 기본 mixture에서는
V7/V9/V12와 한 몫을 나눠 각 3%). V9는 의도적인 역할 리셋 trajectory를 얻기 위한 탐색 분기이며 권장
승격판이 아니다. V9 대 V7의
5시드 양 진영 10경기는 5승 5패, 평균 점수차 +1.6이었다. 리셋은 4회 시도해 4회 모두 완료됐고,
6초 정지는 양쪽 모두 0건, 짧은 막힘은 1,000 유닛틱당 V9 0.350 대 V7 0.356이었다. 즉 경쟁력은
동률이지만 역할 리셋 성공 trajectory와 약간 다른 상태 분포를 만드는 데이터 다양화 정책으로 유효하다.

## 실행 성능

200배속에서는 정책의 Python 비용도 병목이 되므로 결과를 바꾸지 않는 에피소드 캐시를 사용한다.
고정 보호 창고/연결 영역, 시작 픽셀별 거리 맵, `(시작, 목표)`별 A* 경로를 재사용하고, 동적 배터리와
특수 아이템 좌표는 한 팀 decision tick에 한 번만 스캔한다. 거리/A* 캐시는 메모리 상한이 있으며
에피소드 리셋 때 폐기되므로 다른 절차 생성 맵으로 새지 않는다. 3,000틱 고정 관측열에서 최적화 전후
action SHA-256이 일치했고, 합성 V8 측정은 warm 5,000틱 기준 팀당 약 186µs/tick이었다.

- 매 transition에 team perspective 관측, 5개 행동, physical unit index, 역할, 목표 종류, episode seed,
  흡수 구간 index를 저장한다.
- terminal winner와 실제 점수차를 최우선 필터로 사용한다. shaping return만으로 expert episode를
  고르면 navigation PBRS가 큰 패배 경기를 잘못 채택할 수 있다.
- `pickup→own storage→absorption`에 성공한 화물, 수량이 큰 배터리, 위협 회피 후 생존한 운반 경로에
  높은 sample weight를 준다. 배터리 보유 사망, 장시간 정체, 반복 성소 진입은 낮춘다.
- 한 경기 전체를 한 trajectory로만 취급하지 말고 20초 흡수 경계도 segment metadata로 보존한다.
  현재 QMIX trainer의 bootstrap horizon과 맞아 BC 이후 RL 전환이 쉬워진다.
- 고정 holdout seed를 양 진영에서 각각 실행해 승률, 실제 점수차, 100점 도달 시간, 흡수 구간별
  점수 증가를 비교한다. Random은 smoke baseline이고, 학습 checkpoint와 이전 heuristic snapshot을
  회귀 baseline으로 유지한다.

## 실행

모든 명령은 저장소 루트(`blackout-env/`)에서 실행한다. `run_match`와 `run_series`는 실제 Unity terminal
winner를 사용하며, 선택적으로 seed를 받아 재현 가능한 평가를 수행한다. (초기 버전의 `evaluate_heuristic.py`는
아래 벤치마크·건틀릿 스크립트로 대체돼 삭제됐다.)

버전 간 승격 평가는 다음처럼 실행한다. 기본값은 무작위 5시드, 양 진영 10경기이며 6초 이상
완전 정지와 이동 명령 중 막힘 구간을 함께 출력한다.

```bash
./.venv/bin/python examples/benchmark_heuristics.py \
  --candidate v4 --baseline v1 --n-seeds 15 --seed-rng 20260912 --time-scale 200
```

화면에서 V4 플레이를 보려면 `--graphics --time-scale 1`을 추가한다.
