# 내부 구조 및 개발자 가이드

패키지가 어떻게 나뉘어 있는지, 환경 한 스텝이 Unity에서 Python까지 어떻게 흐르는지, 관측을 바꿀 때 무엇을
함께 고쳐야 하는지를 정리한다. 학습·평가 절차는 `models/run11_step80k/README.md`와
`docs/run11_research_baseline.md`를 본다.

## 목차

- [패키지 구조](#패키지-구조)
- [BlackOutEnv 내부 흐름](#blackoutenv-내부-흐름)
- [모델 쪽 관측 처리](#모델-쪽-관측-처리)
- [Obs 수정 가이드](#obs-수정-가이드)
- [테스트](#테스트)
- [의존성 버전 제약](#의존성-버전-제약)

---

## 패키지 구조

```
blackout_env/
├── __init__.py                 — 공개 API 재출(환경, 휴리스틱, 모델 로더, 경기 함수)
├── semantic_map_config.json    — Unity StreamingAssets와 같은 설정(n_items, n_classes). 패키지 기본값
├── env/
│   ├── blackout_env.py         — BlackOutEnv: PettingZoo ParallelEnv, Unity 실행·gRPC·관측 수집
│   ├── my_obs_preprocessor.py  — MyObsPreprocessor: 비트 패킹 그래픽·유닛 상태표·팀 상태 디코딩 (현행)
│   ├── obs_preprocessor.py     — 옛 ObsPreprocessor. 지금은 load_semantic_config와 기반 클래스로만 쓰임
│   ├── semantic_id.py          — 옛 채널 스킴 상수(레거시, 현재 graphic과 맞지 않음)
│   ├── team_frame.py           — Team B 관측·행동을 Team A 좌표계로 거울 변환
│   ├── seed_channel.py         — Python → Unity 시드 전달 SideChannel
│   └── constants.py            — 에이전트 이름·팀 유틸리티
├── competition/match.py        — run_match(), run_series(), MatchResult, SeriesResult
├── heuristics/                 — V1–V19 규칙 기반 정책, 데이터 수집용 mixture (docs/heuristic_policy_catalog_ko.md)
├── model/
│   ├── base.py                 — BaseModel ABC (act(obs) → actions)
│   ├── my_model.py             — MyModel: 그래픽 인코더 + 유닛 토큰 어텐션 + IQN Q 헤드 + SPR
│   ├── my_policy.py            — MyPolicy: 학습된 MyModel을 BaseModel로 감싸 8방향 argmax 행동
│   ├── derived_obs.py          — 모델 입력에서 계산하는 파생 특징(유닛 채널, 창고 여유, 벽 샘플)
│   ├── action_mask.py          — 벽 쪽으로 막힌 방향 마스크
│   ├── loader.py               — load_checkpoint(), load_my_policy_checkpoint()
│   └── modules/                — 인코더, 어텐션(RoPE, GQA), IQN 헤드, QMIX 믹서, SPR 예측기
└── train/
    ├── offline_pretrain.py     — Run 11 학습 진입점(데이터셋 + 온폴리시, 모든 플래그)
    ├── qmix_trainer.py         — QMIXConfig, QMIXTrainer(손실·BBF 리셋·수집 스텝), 옛 온라인 CLI
    ├── collect_heuristic_dataset(_parallel).py — 휴리스틱 데이터 수집
    ├── offline_dataset.py, replay_buffer.py, segment_tree.py, dead_segments.py — 데이터 적재·PER·소진 구간 제거
    ├── onpolicy_collect.py, periodic_eval.py — 학습 중 V4 상대 온폴리시 수집·평가
    ├── reward_v2.py, returns.py, reward_shaping.py — 리워드 v2, n-step 리턴, 막힘 페널티
    ├── *_monitor.py, input_reliance.py, policy_strength.py — 학습 진단·BC 가중치
    └── parallel/               — 실험적 다중 GPU actor/learner (Run 11과 무관, 리워드 v2 미지원)
```

---

## BlackOutEnv 내부 흐름

Unity 쪽에는 두 종류의 에이전트가 있다.

- `MapObsAgent`(behavior `BlackOutMap`) 하나가 매 스텝 **팀 A 시점 그래픽 1장**과 **공유 상태 벡터
  float32[44]**를 보낸다.
- `BlackOutUnit` 10개는 자기 `unitIndex` float 하나만 보내고(행동·보상 라우팅용), 연속 행동 float32[2]를 받는다.

그래픽과 상태를 유닛마다 보내지 않으니 gRPC 패킷이 작다. 인코딩 상세(비트 배치, 채널 표, 벡터 슬롯)는
`env/my_obs_preprocessor.py` 모듈 docstring이 기준이다.

### 초기화

```
__init__()
├── semantic_map_config.json 로드 → n_items, n_classes (다른 키는 옛 전처리기만 씀)
├── MyObsPreprocessor 생성
├── observation_space / action_space 정의 (graphic 24×24×13, team_state 4, agent_states 10×12)
├── SeedChannel, EngineConfigurationChannel(time_scale) 생성
└── UnityEnvironment 실행·연결 (unity_shaping=False면 -noRewardShaping 인자로 Unity 쉐이핑 끔)
```

### step당 흐름

```
step(actions)
├── _send_actions(actions)      BlackOutUnit × 10에 clip된 float32[2], MapObsAgent에 빈 행동
├── unity_env.step()            gRPC 1 라운드트립
└── _collect_obs()
    ├── _collect_map_obs()
    │     visual obs (C,H,W) → preprocess_team_graphics → 팀 A/B 그래픽 (팀 B는 ally/enemy 채널만 교환)
    │     raw_state[44] → preprocess_agent_states / preprocess_team_states (두 팀 시점 모두)
    │     점수·남은 시간·흡수까지 남은 시간 → _latest_scalars
    └── BlackOutUnit × 10 (DecisionSteps + TerminalSteps)
          unitIndex로 에이전트 이름 결정 → _build_obs(팀별 캐시에서 dict 조립)
          종료 시 마지막 점수로 승자 결정(리셋 직후 값이 섞이면 직전 점수로 되돌림)
```

`infos[agent]`에는 `score_0`, `score_1`, `time_left`, `absorption_time_left`와 종료 시 `winner`
(`0`=팀 A, `1`=팀 B, `-1`=무승부)가 들어간다. `rewards`는 Unity가 계산한 값이다 — Run 11 학습은 이것을 쓰지
않고 `train/reward_v2.py`로 Python에서 다시 계산한다.

### 주의

- 팀 B 관측은 **채널 라벨만** 뒤집혀 있다. 격자·유닛 행 순서·행동 좌표는 월드 기준 그대로다.
- Unity 프로세스를 강제로 끊으면 남을 수 있다 — `pkill -f MacOS/RLGame2026`으로 확인.

---

## 모델 쪽 관측 처리

`MyModel`은 env 관측을 그대로 받지 않고 두 단계를 더 거친다.

1. **팀 좌표계 통일** (`env/team_frame.py`): 팀 B 시점 관측을 팀 A 좌표계로 거울 변환하고, 모델이 고른
   8방향 행동을 다시 월드 좌표로 되돌린다. 한 네트워크가 두 진영을 같은 모양으로 본다.
2. **파생 특징** (`model/derived_obs.py`): 유닛 위치 채널, 창고 여유, 가져올 수 있는 배터리 맵과 유닛별 5×5 보간
   벽 샘플(±1칸, 간격 0.5)을 텐서 연산으로 만든다. 데이터셋에는 원본 관측만 저장되므로 이 특징을 바꿔도 재수집은
   필요 없지만, 입력 크기가 바뀌면 체크포인트는 호환되지 않는다.

행동은 env가 받는 연속 `(dx, dy)`가 아니라 **8개 이산 방향**이다(`model/my_policy.py`의 `DIRECTION_VECTORS`). `MyPolicy`가 방향 인덱스를
단위 벡터로 바꿔 env에 넘긴다.

---

## Obs 수정 가이드

### 공유 상태 벡터에 값 추가

1. **Unity** — `MapObsAgent.CollectObservations()`에 값 추가, Behavior Parameters의 Vector Observation Size 갱신
2. **`my_obs_preprocessor.py`** — `RAW_VECTOR_SIZE`(현재 44), 스칼라 슬롯 상수, 필요하면 `team_state_size`
3. **`blackout_env.py`** — `_collect_map_obs`의 `_latest_scalars` 구성, observation_space
4. 유닛별 값이면 `UNIT_BLOCK_SIZE`(현재 4)와 `preprocess_agent_states`, `agent_state_size`

### 그래픽에 아이템 종류 추가

→ Unity 저장소의 `Documentation/adding_item_type.md` 참고

1. `semantic_map_config.json`(Unity StreamingAssets와 패키지 사본 둘 다)의 `n_items` +1
2. 비배터리 아이템 인덱스는 비트 7–9(3비트)라 최대 7종 — 넘으면 Unity `SemanticMapRenderer`와
   `preprocess_team_graphics`의 비트 배치를 함께 바꾼다
3. ally/enemy 구분이 있는 채널이면 팀 B 교환 로직과 `team_frame.py` 확인

### 체크포인트 호환성

`n_graphic_channels`, `agent_state_size`, `team_state_size`, 파생 특징 크기, `hidden_size`가 바뀌면 기존
체크포인트는 로드되지 않는다. 오프라인 데이터셋은 원본 관측을 저장하므로 파생 특징만 바뀐 경우 재사용할 수 있다.

---

## 테스트

Unity 없이 도는 pytest 스위트가 `tests/`에 있다(약 150개, 수 초).

```bash
python -m pytest -q tests
```

주요 대상: 팀 좌표계 변환(`test_team_frame`, `test_policy_team_equivariance`), 파생 특징(`test_derived_obs`),
행동 마스크, 리워드 v2와 n-step 리턴(`test_reward_v2*`), 소진 구간 제거, 휴리스틱(`test_strategic_heuristic`),
경기 승자 판정(`test_match_outcome`), 학습 모니터. 새 기능에는 같은 곳에 케이스를 더한다.

Unity가 필요한 확인은 `examples/`의 스크립트로 한다(예: `examples/run_random.py --build <app>`).

---

## 의존성 버전 제약

```toml
# pyproject.toml
requires-python = ">=3.10"
dependencies = ["pettingzoo>=1.24.0", "gymnasium>=0.29.0", "numpy>=1.23.5", "tqdm>=4.70.0"]
[project.optional-dependencies]
fast = ["numba>=0.59"]   # 휴리스틱 거리 맵 JIT. 없으면 느린 순수 Python 경로
```

| 항목 | 내용 |
|---|---|
| Python | 3.10.x 필수. `mlagents-envs 1.1.0`이 3.11+ 미지원 |
| `mlagents-envs` | `pettingzoo==1.15.0`을 선언하지만 실제로 import하지 않음 → 의존성에서 빼고 `--no-deps`로 따로 설치 |
| protobuf / grpcio | `protobuf>=3.6,<3.21`, `grpcio<=1.48.2`. 더 새 버전이면 Unity gRPC 연결이 조용히 깨진다 |
| torch, tensorboard | 의존성에 넣지 않고 따로 설치. `uv sync`/`uv add`는 위 핀과 torch를 되돌리니 `uv pip install`만 쓴다 |
| Unity | `com.unity.ml-agents 4.0.2` ↔ Python `mlagents-envs 1.1.0`, 에디터 6000.4.11f1 |

설치 순서는 README의 "Local Installation"과 `models/run11_step80k/README.md` 2단계를 따른다.
