# 데이터 수집 속도 — 2026-09-15

휴리스틱 데이터셋 수집 경로(`QMIXTrainer.collect_step` + `HeuristicPolicyMixture`, Unity 1개)를 프로파일링해서
찾은 병목 두 개와 그 조치 기록. 측정 스크립트는 3,000스텝, time scale 20, 다른 Unity 프로세스가 없는 조건.

## 결과

| 설정 | 처리 속도 (스텝/초) | Python CPU/스텝 | Unity CPU/스텝 |
|---|---|---|---|
| 아침 상태 (네비 포텐셜 수정 전, 순수 Python protobuf) | 169.6 | 4.64ms | 1.34ms |
| 팀 단위 네비 포텐셜 1차 구현 | 127.0 | 4.72ms | 3.26ms |
| 네비 포텐셜 최적화 | 190.0 | 4.51ms | 0.83ms |
| **+ C++ protobuf** | **495.5** | **1.23ms** | 0.82ms |

최종 상태에서 벽시계 시간은 Unity 대기가 대부분이고, 휴리스틱 행동 선택이 약 24%다.

## 병목 1: Unity 네비 포텐셜 Φ 계산

- 팀 단위 배터리 배정을 넣은 1차 구현은 매 물리틱마다 유닛 수만큼 `Dictionary` 기반 거리장을 만들고, 이웃 칸마다
  `MapManager.IsWalkable`을 불러 Unity CPU가 3배 가까이 늘었다.
- 단일 경기의 벽시계 시간으로는 약 7% 차이로만 보여서 처음에 놓쳤다. **C# 리워드 코드를 바꾸면 실제 수집 경로에서
  Unity 프로세스(`RLGame2026`)의 CPU 시간/스텝을 잴 것.**
- 조치(`IndividualNavPotentialCalculator.cs`): 팀별 이동 가능 여부를 에피소드 단위로 캐시, 재사용 배열·힙 기반의
  할당 없는 Dijkstra, 배터리를 다 찾으면 조기 종료, 저장소 거리장을 틱당 (팀, 아이템 종류)마다 한 번만 계산.
- 검증: 같은 시드 2경기에서 궤적 동일, 리워드는 한 경기 완전 일치(최대 5.6e-9), 다른 경기는 6개 값이 최대 5e-4 차이
  (모두 −d/+d 짝, 유닛별 경기 합 차이 0 — 가치가 거의 같은 배터리 배정이 반올림으로 몇 틱 뒤바뀐 경우).

## 병목 2: 순수 Python protobuf

- `mlagents_envs` 1.1.0이 `protobuf<3.21`로 고정하고, PyPI의 protobuf 3.20.3에는 Apple Silicon용 C++ 확장 wheel이
  없어서 `py2.py3-none-any`(순수 Python)판이 설치된다. 메시지를 float 하나씩 해석해 수집 시간의 약 20%를 썼다.
- 조치: 공식 소스(`protobuf-python-3.20.3.tar.gz`, SHA-256 `63c58288...ba515`)로 C++ 확장 wheel을 직접 빌드해
  `.venv`에 설치. 버전이 같아서 ML-Agents와 호환된다.
- 검증: `api_implementation.Type() == "cpp"`, 기존 테스트 통과, 같은 시드 경기에서 순수 Python 구현
  (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`)과 **관측 105,020개 SHA-256 및 리워드가 비트 단위로 동일**.

### 주의: `uv sync`를 하면 순수 Python판으로 되돌아간다

빌드한 wheel은 lock 파일에 없어서, `uv sync` 등으로 환경을 다시 맞추면 PyPI의 순수 Python판이 재설치되고 속도
향상이 조용히 사라진다. 확인과 재설치:

```bash
.venv/bin/python -c "from google.protobuf.internal import api_implementation; print(api_implementation.Type())"
uv pip install --python .venv/bin/python --reinstall --no-deps build/wheels/protobuf-3.20.3-cp310-cp310-macosx_11_0_arm64.whl
```

wheel은 `build/wheels/`(git 추적 제외)에 있다. 없으면 다시 빌드:

```bash
curl -sSLO https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protobuf-python-3.20.3.tar.gz
tar xzf protobuf-python-3.20.3.tar.gz && cd protobuf-3.20.3
./configure --disable-shared --with-pic CXXFLAGS="-O2" && make -j"$(sysctl -n hw.ncpu)"
cd python && /path/to/blackout-env/.venv/bin/python setup.py bdist_wheel --cpp_implementation
# dist/protobuf-3.20.3-cp310-cp310-macosx_11_0_arm64.whl — _message.so가 libprotobuf를 정적 링크해 시스템 라이브러리만 의존
```
