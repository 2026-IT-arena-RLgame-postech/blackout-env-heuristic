# 공식 형식 제출물 (P24c / P24c + R6 탐색)

공식 `ist-tech-AI-games/blackout-env` README의 `policy.py` + `checkpoint.pt` 형식이다. 원본 관측만 받아서 fork 관측을 복원하고 P24c로 행동한다.

```python
from blackout_env import load_checkpoint          # 공식 패키지
from policy import BlackoutPolicy
model = load_checkpoint(BlackoutPolicy, "checkpoint.pt", vector_size=96, n_channels=11)
```

## 구성

| 단계 | 위치 | 내용 |
|---|---|---|
| 1. 추적기 (상태 있음) | `bo_fork/`: 체크포인트 안에 zip(`sim_src`)으로도 들어 있음 | 규칙이 정확한 시뮬레이터를 관측 위치로 구동한다. RNG는 InitState(0) 체인으로 맞추므로 맵이 없어도 된다. 출력은 fork 방송(raw 44 + packed map 24×24)이다. |
| 2. 그래프 (상태 없음) | `checkpoint.pt` → `onnx`(onnxruntime), `torchscript`(onnxruntime이 없을 때) | fork 관측 디코딩, 팀 B 미러링, P24c 망, argmax와 각도 오프셋, 월드 좌표 (dx, dy) 복원까지 한다. `blackout_env/official/policy_graph.py` |
| 3. 탐색 (선택) | `base_net`, `rollout_net`, `value_net` 버퍼가 있을 때만 | §61 `o4f_r24c`: 5초부터 25결정마다 r24cd256으로 8가지 분기 × 상대 변형 4개를 탐색하고, 리프 가치는 V iter2로 평가한다. `blackout_env/official/search_player.py` |

- 결정은 2틱마다 한 번이고, 그 사이 틱에는 같은 행동을 유지한다(학습 때의 주기와 같다).
- 매치 경계는 타이머로 감지한다. 인스턴스 하나에 추적기 하나를 둔다. 같은 인스턴스가 양 팀을 맡아도 틱마다 한 번만 관측한다.
- 추적기에서 예외가 나면 경고를 남기고 추적기를 재시작한다. 그 틱에는 직전 행동을 유지한다.
- 필수 의존성은 numpy와 torch뿐이다. onnxruntime과 numba가 있으면 더 빠르다(탐색은 numba가 사실상 필요하다).
- 환경 변수:
  - `BLACKOUT_POLICY_BACKEND=torchscript`: ONNX 대신 TorchScript를 쓴다.
  - `BLACKOUT_POLICY_SEARCH=0`: 탐색을 끈다.
  - `BLACKOUT_SEARCH_DEVICE=cpu|cuda`: 탐색 장치를 고른다.
  - `BO_SIM_NO_NUMBA=1`: numba 없이 순수 Python으로 돌린다.

## 빌드와 검증

```bash
# onnx, onnxscript, onnxruntime, torch가 있는 venv에서 (protobuf가 최신이어야 함)
PYTHONPATH=. python examples/build_official_submission.py --states states.npz --out dist/official_submission   # states.npz: examples/official_submission_states.py
PYTHONPATH=. python examples/build_official_submission.py --states states.npz --out dist/official_submission_search --search
# Unity 없이: 공식 load_checkpoint + 공식 주기 시뮬레이터, P24c 참조 행동과 매 결정 비교
PYTHONPATH=. python examples/official_submission_check.py --official-env <공식 repo clone> --two-files
# 공식 빌드 + 공식 BlackOutEnv/run_series (공식 패키지를 PYTHONPATH에)
PYTHONPATH=<공식 repo> python submission/official/run_unity_series.py --sub dist/official_submission --build BlackOut.x86_64
```

`--states`는 fork 방송 샘플 npz다(`raw` [N, 44], `map` [N, 24, 24]). ONNX와 TorchScript를 torch 그래프와 비교해서, 하나라도 argmax가 다르면 빌드를 중단한다.
