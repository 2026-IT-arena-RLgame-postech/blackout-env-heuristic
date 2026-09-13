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
- 한 에피소드 안에서는 정책을 바꾸거나 action noise를 넣지 않는다. 정책과 파라미터는 에피소드
  시작 시 한 번 샘플링해 trajectory의 의도를 일관되게 유지한다.
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

V4PolicyFamily  ── V4의 작은 에피소드 단위 파라미터 변형
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

## V4 근접 변형군

`V4PolicyFamily`는 매 에피소드마다 아래 profile 하나를 샘플링한다. 매 tick noise는 사용하지 않는다.

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

| policy_id | 기본 가중치 | 데이터 내 역할 |
|---|---:|---|
| `strategic_v1` | 10% | 단순 기준 행동 |
| `strategic_v2` | 9% | 전역 경제 배정 |
| `strategic_v3` | 12% | 안전 적재 |
| `strategic_v4` | 23% | 주 교사 정책 |
| `strategic_v4_near` | 18% | V4 주변 조밀한 변형 |
| `strategic_v5` | 3% | 공격적 예측 요격 |
| `strategic_v6` | 7% | 위험 회피·저정체 경로 |
| `strategic_v7` | 5% | 동적 역할 배분 |
| `strategic_v8` | 1% | 보수적 역할 리셋 대조군 |
| `strategic_v9` | 3% | 실제 역할 리셋 trajectory |
| `strategic_v10` | 4% | 국면별 역할/목표 전환 |
| `strategic_v11` | 3% | 흡수 직전 다중-unit 약탈 |
| `strategic_v12` | 2% | 리드 보존·창고 방어 |

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

`reset()`을 명시적으로 호출하면 다음 정책을 샘플링하고 새 `PolicySample`을 반환한다. 정책은 Unity의
남은 시간이 크게 증가하는 것도 새 에피소드로 감지하지만, 데이터 수집기는 첫 transition 전에
provenance를 기록할 수 있도록 명시적 `reset()`을 권장한다.

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

`perturb=True`이면 일반 정책에 다음 파라미터를 에피소드 단위로 샘플링한다.

- `use_specialists`: 92% 확률로 true
- `replan_interval`: 8–13틱
- `threat_radius`: 0.135–0.185
- V5 추가: `intercept_margin_seconds` 0.20–0.40,
  `intent_temperature` 2.5–6.0, `velocity_ema` 0.35–0.75
- V6 추가: `risk_radius_tiles` 2.0–3.5, `risk_weight` 2.0–4.2
- V10 추가: 모드 확인 지연 14–22틱, 약탈 압박 흡수 임계 0.32–0.48,
  리드 보존 점수차 0.07–0.12
- V11 추가: 모드 확인 지연 8–16틱, 동시 약탈조 1–2기, 약탈 흡수 임계 0.42–0.66
- V12 추가: 모드 확인 지연 15–24틱, fortress 진입 리드 0.05–0.11
- `strategic_v4_near`: 일반 변형 대신 V4 profile을 중첩 샘플링

`perturb=False`이면 정책 버전은 계속 가중치에 따라 샘플링하지만 공통 파라미터는
`use_specialists=True`, `replan_interval=10`, `threat_radius=0.16`으로 고정되고 V4-near는 exact만
사용한다.

주의: V7–V10/V12는 `_role()`을 동적 역할 배분으로 재정의하므로 일반 혼합의 `use_specialists=False`가
동적 역할을 끄는 스위치로 동작하지 않는다. 이 정책들의 역할을 끄거나 수량을 조절하려면 직접 생성해
`carrier_quota`, `hunter_quota`를 설정하거나 해당 정책을 혼합에서 제외한다.

## 상태 전환 정책(V10–V12)의 사용법

세 정책은 매 tick 무작위로 역할을 바꾸지 않는다. 관측된 조건이 연속 `mode_confirm_ticks`번 유지될
때만 새 모드로 전이하는 hysteresis를 사용한다. 조건은 모두 공개 `team_state`와 semantic map에서만
계산한다. 즉 BC 교사나 대회 정책으로 써도 특권 정보 누출이 없다.

| 정책 | 모드 | 공개 전이 신호 | unit별 결과 |
|---|---|---|---|
| V10 | `opening_economy` | 기본 초반 | 새 Hunter 변신을 늦추고 경제/Carrier를 우선 |
| V10 | `pressure_raid` | 상대 외부 창고에 가치가 있고 흡수 임박, 또는 크게 열세 | Hunter가 고가 화물/Carrier를 우선 추격하고 화물이 없으면 외부 창고를 순찰 |
| V10 | `closeout_defend` | 리드 상태 + 후반 | Hunter가 빈 Collector를 추격하지 않고 아군 창고를 순찰; 적 화물이 보일 때만 요격 |
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
13개이므로 78개 비대각 쌍 × 5 seed × 양 진영 = 780경기다. `--policies v4 v7 v10 v11 v12`처럼
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
