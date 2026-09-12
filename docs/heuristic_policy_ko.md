# BlackOut BC 부트스트랩용 전략 휴리스틱

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
3. 아이템 보유자는 가장 가까운 아군 창고의 연결 영역 중심으로 이동한다. 중심 진입은 경계 픽셀에서
   멈춰 `OnUnitEnter`가 누락되는 현상을 막는다.
4. 빈 Collector/Carrier는 배터리 수량, 거리, 적 창고 여부, 다음 흡수 임박도를 합친 점수로 목표를
   선택한다. 팀원끼리 같은 픽셀을 중복 선택하지 않도록 목표를 예약한다.
5. 필드 배터리가 없으면 적 창고를 순찰·약탈한다. 적 창고 배터리는 흡수가 임박할수록 우선도가
   높아져 상대 확정을 차단한다.
6. Hunter는 화물 보유 적과 Carrier를 우선 추격하고, 그 다음 가까운 적을 차단한다.
7. 배터리 운반자와 Carrier는 가까운 적 Hunter(그리고 Carrier에게는 모든 적)로부터 국소 반발력을
   받아 화물 소멸 위험을 낮춘다.
8. 필드 목표가 고갈되면 Collector/Carrier는 약탈 가능한 외부 적 창고들을 순환하고, Hunter는
   아군 창고들을 순환 방어한다. 정적인 한 지점에서 경기 종료까지 대기하지 않는다.

## 이동 계획

- semantic map의 wall channel을 이용해 8방향 A*를 수행한다.
- 대각선 코너 통과는 두 직교 인접 칸이 모두 통과 가능할 때만 허용한다.
- 목표/역할 변경, 일정 틱 경과, 움직이는 추격 목표, 정체 감지 시 경로를 다시 계산한다.
- A*가 연결 경로를 찾지 못하면 목표를 향해 직선으로 벽을 밀지 않고 현재 연결 영역의 통과 가능한
  이웃을 탐색한다.
- 연속 정체가 감지되면 유닛 index에 따라 반대 방향의 수직 nudge를 주어 유닛 간 대칭 충돌과
  오목한 모서리 고정을 해소한다.
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

`HeuristicPolicyMixture(seed=...)`는 에피소드마다 위 버전을 샘플링하고 `replan_interval`,
`threat_radius`, specialist 구성에 제한된 변주를 준다. 데이터 수집기는 매 trajectory에
`mixture.current_sample`의 `policy_id`, `policy_seed`, `parameters`를 저장해야 한다. 같은 seed의
mixture는 동일한 정책열을 재현하므로 offline RL의 behavior-policy provenance와 중요도 가중에도
사용할 수 있다. 변주가 필요 없는 평가에서는 `perturb=False`를 사용한다.

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

```bash
cd /Users/mac/project/26rl/blackout-env
./.venv/bin/python examples/evaluate_heuristic.py \
  --build build/mac/BlackOut.app --opponent random --seeds 101 202 303
```

각 seed에서 진영을 교대하므로 총 경기 수는 seed 수의 두 배다. `run_match`와 `run_series`는 실제
Unity terminal winner를 사용하며, 선택적으로 seed를 받아 재현 가능한 평가를 수행한다.

버전 간 승격 평가는 다음처럼 실행한다. 기본값은 무작위 5시드, 양 진영 10경기이며 6초 이상
완전 정지와 이동 명령 중 막힘 구간을 함께 출력한다.

```bash
./.venv/bin/python examples/benchmark_heuristics.py \
  --candidate v4 --n-seeds 15 --seed-rng 20260912 --time-scale 100
```

화면에서 V4 플레이를 보려면 `--graphics --time-scale 1`을 추가한다.
