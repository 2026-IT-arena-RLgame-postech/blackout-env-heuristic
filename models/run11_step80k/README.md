# Run 11 · 80k 체크포인트

지금까지 가장 강한 체크포인트(2026-09-18 기준). `step_80000.pt`는 Git LFS로 저장된다(`git lfs pull`).

- 학습 런: `checkpoints/offline/20260917-130437_59475` (2026-09-17 13:04–16:15, 200k 스텝)
- 학습 당시 코드: blackout-env `6b080de`(데이터 `0009fd7`, 보상 `eca9bb4`) · Unity 빌드: blackout `94fabbd`
- SHA-256: `9f089c1f1d3d915f926fb3eaf31ee81f40e2dd6334ba4b388f61e043da2e85cd`
- 기록: `docs/offline_pretrain_runs.md`의 "Run 11 계획 / 결과 / 체크포인트 상대 Elo"

## 무엇으로 학습했나

| 항목 | 값 |
|---|---|
| 알고리즘 | QMIX + IQN + SPR(5.0) + BC(α 1.0), PER, BBF 리셋 40k마다, n-step 10→3 / γ 0.97→0.997, fp16 + compile |
| 관측 | 24×24 시맨틱 맵 + 유닛 토큰(타일 내 위상 없음, 5×5 보간 벽 샘플) |
| 보상 | 리워드 v2 `FITTED_20260917B`(파이썬 계산: 확정 점수 + 유닛별 포텐셜, n-step 안에서 학습기 γ로 쉐이핑) + 막힘 페널티 0.02 |
| 데이터 | `heuristic_mixv6_live_20260917`: 휴리스틱 mixture(V17–V19 32%), 유닛 행동 10% 균등 무작위, 배터리 소진 시 경기 종료, 스트림당 100만 행 |
| 온폴리시 | 배치의 30%, V4 상대, 10k마다 1.5만 틱, 배터리 소진 시 경기 종료 |

## 성능

- 휴리스틱 척도: V1(Elo 1192)에게도 한 점도 못 딴다 — 절대 Elo는 V1 아래로만 알 수 있다.
- Run 11 체크포인트끼리(final = 0, `examples/elo_checkpoints.py`): **80k +78 ± 29**, 140k +31 ± 29, 160k +22 ± 29.
  80k는 final에게 61/96(64%).
- Run 13 final과: Run 11 final이 +115 ± 54 Elo.
- V4 상대 GUI 6경기(시드 404/505/606): 0승 6패, 평균 점수차 −76.

알려진 약점(측정: `measure` 단계):
- 첫 10초 배달이 V4의 약 1/3 — 초반 수거 속도.
- 외곽 창고에 넣은 배터리의 대부분을 10–20초에 도둑맞는다.
- Hunter가 벽 바로 옆 칸에서 방향을 자주 뒤집는다(진동). 스폰 창고 편중.

## 처음부터 재현하기 (다른 환경)

1. 두 저장소를 같은 부모 폴더에 `run11-80k` 브랜치로 받는다.
   ```
   git clone -b run11-80k https://github.com/cucumbersaurus/blackout-env.git
   git clone -b run11-80k https://github.com/cucumbersaurus/blackout.git
   cd blackout-env && git lfs pull
   ```
2. 파이썬 환경: 저장소 `README.md`의 "Local Installation"(Python 3.10, `mlagents-envs==1.1.0`는 `--no-deps`) 뒤에
   `pip install ".[fast]" torch tensorboard "protobuf>=3.6,<3.21" "grpcio>=1.11.0,<=1.48.2"`(uv면 `uv pip install`,
   `uv sync`/`uv add`는 쓰지 않는다 — 저장소 README 참고). 동작 확인한 버전: Python 3.10.12, torch 2.14.0(MPS),
   numba 0.67.0(휴리스틱 가속, 없으면 느린 경로), protobuf 3.20.3, grpcio 1.48.2, tensorboard 2.20.0.
   protobuf/grpcio가 이보다 새 버전이면 Unity gRPC 연결이 조용히 깨진다.
3. Unity 빌드(Unity 6000.4.11f1, macOS): `blackout-env`에서
   ```
   models/run11_step80k/run11_pipeline.sh build      # ../blackout -> build/mac/BlackOut.app
   ```
   Unity 경로가 다르면 `UNITY=/path/to/Unity`. 새로 받은 프로젝트도 첫 임포트 포함 약 2분(M5 Pro).
4. 학습 재현: `collect`(디스크 57GB, 18 워커) → `train`. 학습은 스트림당 100만 행을 메모리에 올린다 — RAM 48GB(M5 Pro)에서
   스왑과 함께 동작했다. CUDA면 `DEVICE=cuda`, 코어가 적으면 `COLLECT_WORKERS=`로 줄인다.
   학습은 시드를 고정하지 않아 같은 명령이라도 체크포인트가 달라진다 — Run 11 안에서도 체크포인트 간 Elo가
   ±100 넘게 오르내렸으니, 재학습 후에는 `elo` 단계로 상위 체크포인트를 고른다.

## 재현 점검 (2026-09-18)

GitHub에서 두 브랜치를 새로 받아 파이프라인을 작게 돌려 확인했다:
LFS 체크포인트 해시 일치 · `build`(새 클론, 2분, 150MB) · `collect`(4천 행) · `train`(300스텝, eval과 V4 온폴리시
수집 포함, `final.pt` 저장) · `elo`(Run 11 체크포인트) · `measure`(80k: V4 상대 8경기 −72.5, 배달 70, 스폰 창고 47%).
점검에서 고친 것: `offline_pretrain --help` 오류(`%` 미이스케이프), Unity 빌드 출력 경로 하드코딩, 빌드 경로 상대경로 처리.
GUI 단계(`gui`/`selfplay`/`vs`)는 같은 스크립트를 창 모드로 돌리는 것이라 창 없이 확인한 경로와 같다.
파이썬 환경 설치는 이 점검에 포함하지 않았다(기존 환경 사용).

## 재현 / 평가

```
models/run11_step80k/run11_pipeline.sh build      # Unity 빌드
models/run11_step80k/run11_pipeline.sh collect    # 데이터 (약 6분, 57GB)
models/run11_step80k/run11_pipeline.sh train      # 학습 (약 3시간, 80k는 step_80000.pt)
models/run11_step80k/run11_pipeline.sh gui        # V4 상대 GUI
models/run11_step80k/run11_pipeline.sh selfplay   # 자기 자신과 GUI
models/run11_step80k/run11_pipeline.sh vs checkpoints/offline/<run>/final.pt
models/run11_step80k/run11_pipeline.sh measure    # 진동·창고·시간대별 도난 측정
models/run11_step80k/run11_pipeline.sh elo checkpoints/offline/<run>
```

이 브랜치의 코드는 학습 당시보다 뒤 버전이다. 기본값은 Run 11과 같게 동작하지만 두 가지가 다르다.
- 수집기의 `--noise-frac` 기본값이 0이 됐다 — 스크립트가 0.1을 명시한다.
- 조기 종료 경기의 점수 기록 버그(`583c5aa`)가 고쳐졌다. Run 11 학습 중 periodic eval 점수차 일부와 온폴리시
  조기 종료 경기의 승패가 이 버그의 영향을 받았을 수 있다.
