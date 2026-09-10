# API 레퍼런스

## 목차

- [BlackOutEnv](#blackoutenv)
- [ObsPreprocessor (레거시)](#obspreprocessor-레거시)
- [SemanticId (레거시)](#semanticid-레거시)
- [에이전트 유틸리티](#에이전트-유틸리티)
- [Competition — run_match / run_series](#competition)
- [Competition — BaseModel / load_checkpoint](#basemodel--load_checkpoint)

---

## BlackOutEnv

```python
from blackout_env import BlackOutEnv
```

PettingZoo `ParallelEnv` 구현. `mlagents_envs.UnityEnvironment` 래퍼.

### 생성자

```python
env = BlackOutEnv(
    env_path="path/to/BlackOut.exe",   # None = Unity Editor에 연결
    semantic_config_path="semantic_map_config.json",
    map_w=96,            # 텍스처 가로 (Unity resolutionScale × mapWidth와 일치해야 함)
    map_h=96,            # 텍스처 세로
    worker_id=0,         # base_port에 더해지는 오프셋
    base_port=None,      # None이면 OS가 빈 포트 자동 선택 (병렬 훈련 권장)
    no_graphics=True,    # Unity 빌드에 --no-graphics 전달
    time_scale=1.0,      # Unity Time.timeScale (헤드리스 빌드: 20~100 권장)
)
```

**파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `env_path` | `str \| None` | — | Unity 빌드 경로. `None`이면 에디터에 연결 |
| `semantic_config_path` | `str \| Path` | — | `semantic_map_config.json` 경로 |
| `map_w`, `map_h` | `int` | `96` | visual obs 텍스처 크기 |
| `worker_id` | `int` | `0` | `base_port`가 지정된 경우에만 사용 |
| `base_port` | `int \| None` | `None` | `None`이면 자동 포트 (Ray 등 병렬 환경에 권장) |
| `no_graphics` | `bool` | `True` | Unity 빌드 전용. 에디터 연결 시 무시 |
| `time_scale` | `float` | `1.0` | 에디터 연결 시 `1.0` 유지 |

### 메서드

#### `reset(seed=None, options=None)`

```python
obs, infos = env.reset(seed=42)
```

- `seed`: `int | None`. 지정 시 Unity `Random.InitState(seed)` 호출 → 맵/아이템 배치 재현
- 반환: `(obs_dict, infos_dict)`

#### `step(actions)`

```python
obs, rewards, terminations, truncations, infos = env.step(actions)
```

- `actions`: `dict[agent_name, float32[2]]`
- `truncations`: 항상 `False` (모든 에피소드 종료는 termination)
- `infos`: 매 스텝 `score_0`, `score_1`, `time_left` 포함. 에피소드 종료 스텝에만 `winner` 추가

**infos 구조:**

| 키 | 타입 | 설명 |
|---|---|---|
| `score_0` | `float` | 팀 A 점수 / 목표 점수 |
| `score_1` | `float` | 팀 B 점수 / 목표 점수 |
| `time_left` | `float` | 남은 시간 비율 (0~1) |
| `winner` | `int` | 종료 스텝에만 존재. `0`=팀A, `1`=팀B, `-1`=무승부 |

#### `close()`

Unity 프로세스 종료. 훈련 루프 종료 후 반드시 호출.

#### `observation_space(agent)` / `action_space(agent)`

모든 에이전트가 동일한 공간을 공유합니다.

**Observation space** (`spaces.Dict`):

| 키 | 형태 | 범위 | 설명 |
|---|---|---|---|
| `"graphic"` | `float32[H, W, C]` | `[0, 1]` | 팀 시점 semantic map (타일 종류 one-hot + 배터리 개수 + 아이템 one-hot) |
| `"team_state"` | `float32[4]` | 대체로 `[0, 1]` | `[own_score, opp_score, episode_time_left, absorption_time_left]` |
| `"agent_states"` | `float32[10, 12]` | `[-1, 1]` | 유닛 10개(양팀 전체) 상태 테이블, unit_0~9 순 |

`C = 8 + 1 + (n_items - 1)` (item 0은 항상 스택형 배터리로 채널 8의 스칼라에 인코딩됨)  
유닛 위치는 `graphic`에 없고 `agent_states`에만 있습니다.

**Action space** (`spaces.Box`):

| 형태 | 범위 | 설명 |
|---|---|---|
| `float32[2]` | `[-1, 1]` | `[dx, dy]` 이동 벡터 |

### Observation 상세 — graphic

채널 0-7은 타일 종류 one-hot(`0.0`/`1.0`), 채널 8은 배터리 개수 스칼라(one-hot 아님), 나머지는 아이템 one-hot입니다:

| 채널 | 의미 | 값 |
|---|---|---|
| 0 | void | `0.0`/`1.0` |
| 1 | wall | `0.0`/`1.0` |
| 2 | site_hunter | `0.0`/`1.0` |
| 3 | site_carrier | `0.0`/`1.0` |
| 4 | spawn_ally | `0.0`/`1.0` |
| 5 | spawn_enemy | `0.0`/`1.0` |
| 6 | storage_ally | `0.0`/`1.0` |
| 7 | storage_enemy | `0.0`/`1.0` |
| 8 | 배터리 개수 | `count / 15` (스칼라) |
| 9 | item_1 (BuffSpeed) | `0.0`/`1.0` |
| 10 | item_2 (DebuffSpeed) | `0.0`/`1.0` |
| 11 | item_3 (BuffSize) | `0.0`/`1.0` |
| 12 | item_4 (DebuffSize) | `0.0`/`1.0` |

`ally`/`enemy` 채널(4↔5, 6↔7)은 이미 관찰 주체 팀 시점으로 뒤집혀 있습니다(`BlackOutEnv` 내부 `flip_team_perspective()`에서 처리). 유닛은 `graphic`에 전혀 등장하지 않습니다.

### Observation 상세 — agent_states

`float32[10, 12]` — 유닛 10개(고정: 인덱스 0-4=팀A, 5-9=팀B) × 12개 필드. 관찰 주체 팀 시점으로 `team` 컬럼만 부호가 다릅니다.

| Offset | 길이 | 필드 | 비고 |
|---|---|---|---|
| 0-1 | 2 | `pos_x`, `pos_y` | `[-1, 1]` 정규화 좌표 |
| 2 | 1 | `team` | 아군 `+1.0`, 적군 `-1.0` |
| 3-8 | 6 (`n_items+1`) | `holding_item` one-hot | index 0=없음, index 1=배터리(값=`count/15`, flat 1.0 아님), index 2+=기타 아이템 |
| 9-11 | 3 (`n_classes`) | `class` one-hot | 이 유닛의 클래스 |

`agent_state_size = 2 + 1 + (n_items+1) + n_classes`

### Observation 상세 — team_state

`float32[4]` = `[own_score, opp_score, episode_time_left, absorption_time_left]`. 두 점수는 팀별로 재정렬되어 index 0이 항상 "내 점수"입니다. 시간 값 두 개는 양 팀에 공통이며 `[0, 1]` 범위입니다.

---

## ObsPreprocessor (레거시)

> **⚠️** 이 클래스는 `BlackOutEnv`가 더 이상 내부적으로 사용하지 않습니다. `BlackOutEnv`는
> 실제로 `MyObsPreprocessor`(공개 API로 export되지 않음, `blackout_env.env.my_obs_preprocessor`)를
> 써서 위의 `graphic`/`team_state`/`agent_states` 형식을 만듭니다. 여기 문서화된
> `ObsPreprocessor.preprocess_vector()`/`preprocess_graphic()`은 옛 단일 `vector`+`graphic`
> 딕셔너리 방식(6~11채널 `SemanticId` 스킴)을 위한 것으로, 그 방식으로 raw obs를 직접
> 다루고 싶을 때만 쓸모가 있고 `BlackOutEnv`가 실제로 반환하는 값과는 무관합니다.

```python
from blackout_env import ObsPreprocessor, load_semantic_config
```

Unity raw obs → 모델 입력 변환. 스테이트리스.

### 생성자

```python
cfg = load_semantic_config("semantic_map_config.json")
preprocessor = ObsPreprocessor(cfg, n_items=1, n_classes=3)
```

### 속성

| 속성 | 타입 | 설명 |
|---|---|---|
| `vector_obs_size` | `int` | 전처리 후 vector 크기 |
| `n_graphic_channels` | `int` | graphic 채널 수 = `item_id_offset + n_items` |
| `RAW_VECTOR_SIZE` | `int` | raw vector 크기 (45) |
| `RAW_CLASS_SLOT` | `int` | raw vector에서 classId 위치 (40) |
| `RAW_SCALAR_START` | `int` | own_score 시작 위치 (41) |
| `RAW_UNIT_INDEX_SLOT` | `int` | unitIndex 위치 (44) — 전처리 시 제거 |

### 메서드

#### `preprocess_vector(raw)`

```python
vector = preprocessor.preprocess_vector(raw_float32_45)
# → float32[vector_obs_size]
```

holdingItemId, classId를 one-hot으로 확장. unitIndex 제거.

#### `preprocess_graphic(raw)`

```python
graphic = preprocessor.preprocess_graphic(raw_float32_HxWx1)
# → float32[H, W, n_graphic_channels]
```

픽셀 ID를 binary channel masks로 변환. 입력은 ML-Agents가 정규화한 `[0, 1]` 값.

#### `flip_team_perspective(graphic)`

```python
team_b_graphic = preprocessor.flip_team_perspective(team_a_graphic)
```

ally/enemy 채널 스왑 (ch2↔ch3, ch4↔ch5). 팀 B 관점 graphic 생성.

---

## SemanticId (레거시)

> **⚠️** `EMPTY`/`ALLY_STORAGE`/`ALLY_UNIT`/... 상수는 위 레거시 `ObsPreprocessor`의 6~11채널
> 스킴에 대응합니다. 지금 `BlackOutEnv`가 반환하는 `graphic`(void/wall/site_hunter/.../battery/item_1~4,
> 13채널)과는 채널 의미도 개수도 다르므로 **현재 `graphic` obs를 인덱싱하는 데 쓰면 안 됩니다.**
> 현재 채널 상수를 공개 API로 export하는 클래스는 아직 없습니다 — 위 "Observation 상세 — graphic"
> 표의 채널 번호를 직접 참조하세요.

```python
from blackout_env import SemanticId
```

전처리 후 graphic obs의 채널 인덱스 상수.

### 클래스 상수

```python
SemanticId.EMPTY          # 0
SemanticId.WALL           # 1
SemanticId.ALLY_STORAGE   # 2
SemanticId.ENEMY_STORAGE  # 3
SemanticId.ALLY_UNIT      # 4
SemanticId.ENEMY_UNIT     # 5
SemanticId.ITEM_ID_OFFSET # 6
SemanticId.BASE_IDS       # (0, 1, 2, 3, 4, 5)
```

### 클래스 메서드

| 메서드 | 설명 |
|---|---|
| `item_channel(item_index)` | KnownItems 인덱스 → 채널 인덱스 (`6 + item_index`) |
| `item_index(channel)` | 채널 인덱스 → KnownItems 인덱스. 아이템 채널이 아니면 `ValueError` |
| `is_item(channel, n_items)` | 채널이 유효한 아이템 채널인지 확인 |
| `all_channels(n_items)` | 전체 채널 인덱스 리스트 반환 |
| `name(channel, n_items=0)` | 채널 인덱스 → 이름 문자열 (예: `"ally_unit"`, `"item_0"`) |

```python
# 사용 예시
ally_mask = graphic[:, :, SemanticId.ALLY_UNIT]
item0     = graphic[:, :, SemanticId.item_channel(0)]
SemanticId.name(4)          # "ally_unit"
SemanticId.is_item(7, n_items=2)  # True
```

---

## 에이전트 유틸리티

```python
from blackout_env import team_of, team_a_agents, team_b_agents, all_agents
# 또는
from blackout_env.env.constants import agent_name, unit_index, team_of, ...
```

| 함수 | 설명 | 예시 |
|---|---|---|
| `agent_name(unit_index)` | unitIndex → agent name | `agent_name(3)` → `"unit_3"` |
| `unit_index(agent)` | agent name → unitIndex | `unit_index("unit_3")` → `3` |
| `team_of(agent)` | agent name → 팀 번호 (0 or 1) | `team_of("unit_6")` → `1` |
| `team_a_agents()` | Team A agent name 리스트 | `["unit_0", ..., "unit_4"]` |
| `team_b_agents()` | Team B agent name 리스트 | `["unit_5", ..., "unit_9"]` |
| `all_agents()` | 전체 agent name 리스트 | `["unit_0", ..., "unit_9"]` |

```python
# 팀별 obs 분리
a_obs = {k: v for k, v in obs.items() if team_of(k) == 0}
b_obs = {k: v for k, v in obs.items() if team_of(k) == 1}
```

상수:

| 상수 | 값 | 설명 |
|---|---|---|
| `BEHAVIOR_NAME` | `"BlackOutUnit"` | Unity behavior 이름 |
| `MAP_BEHAVIOR_NAME`\* | `"BlackOutMap"` | MapObsAgent behavior 이름 |
| `N_AGENTS` | `10` | 전체 에이전트 수 |
| `N_TEAM_A` | `5` | 팀당 에이전트 수 |

\* `MAP_BEHAVIOR_NAME`은 패키지 최상위(`from blackout_env import ...`)로는 export되지 않습니다.
`from blackout_env.env.constants import MAP_BEHAVIOR_NAME`으로 직접 가져와야 합니다.

---

## Competition

```python
from blackout_env import run_match, run_series
```

### `run_match(env, model_a, model_b, swap_teams=False)`

단일 에피소드 실행.

```python
result = run_match(env, model_a, model_b)
print(result.winner)              # 0=model_a, 1=model_b, None=무승부
print(result.team_a_total_reward) # float
print(result.episode_steps)       # int
```

- `swap_teams=True`: model_a가 팀 B로 플레이. 페어니스를 위해 `run_series`가 내부에서 자동 스왑.
- `winner`는 항상 model_a/model_b 기준으로 보고 (팀 스왑 여부 반영됨).

### `run_series(env, model_a, model_b, n_matches=10)`

N 경기 시리즈. 짝수 번째 경기마다 팀 스왑.

```python
series = run_series(env, model_a, model_b, n_matches=10)
print(series.model_a_wins)   # int
print(series.model_b_wins)   # int
print(series.draws)          # int
print(series.series_winner)  # 0 or 1 or None
```

---

## BaseModel / load_checkpoint

```python
from blackout_env import BaseModel, load_checkpoint
```

### `BaseModel` (ABC)

```python
class MyModel(BaseModel):
    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """
        obs   : {agent_name: {"graphic": float32[H, W, C], "team_state": float32[4],
                               "agent_states": float32[10, 12]}}
        return: {agent_name: float32[2]}  — (dx, dy) in [-1, 1]
        """
        ...
```

### `load_checkpoint(model_class, checkpoint_path, ...)`

PyTorch `nn.Module` 체크포인트를 로드해 `BaseModel`로 래핑.

```python
model = load_checkpoint(
    MyPolicy,                      # nn.Module 서브클래스
    "checkpoint.pt",
    state_dict_key="policy_state", # None이면 raw state dict
    device="cuda",
    # model_class 생성자 kwargs
    n_graphic_channels=13,
    agent_state_size=12,
    team_state_size=4,
)
```

**체크포인트 저장 형식:**

```python
# raw state dict
torch.save(model.state_dict(), "checkpoint.pt")
# → load_checkpoint(..., state_dict_key=None)

# dict 형식 (추가 메타데이터 포함 가능)
torch.save({"policy_state": model.state_dict(), "step": 1000}, "checkpoint.pt")
# → load_checkpoint(..., state_dict_key="policy_state")  ← 기본값
```

**`nn.Module` forward 인터페이스:**

```python
class MyPolicy(nn.Module):
    def forward(
        self,
        graphic: torch.Tensor,       # (B, C, H, W)   float32  ← CHW 순서
        team_state: torch.Tensor,    # (B, 4)         float32
        agent_states: torch.Tensor,  # (B, 10, 12)    float32
    ) -> torch.Tensor:               # (B, 2)         float32
        ...
```

`load_checkpoint`가 반환하는 `CheckpointModel`은 `graphic`을 `(B, H, W, C) → (B, C, H, W)`로
자동 변환하고, `team_state`/`agent_states`는 배치 차원만 붙여 그대로 전달합니다.
