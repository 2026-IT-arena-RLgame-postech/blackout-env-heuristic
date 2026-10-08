# CUDA 학습 (poscat 서버 RTX 2080 Ti) — 2026-10-03

서버(poscat_gpu1/2: RTX 2080 Ti 11 GB × 4, Turing sm_75 = fp16 텐서코어 O, **bf16·TF32 없음**, 28코어, 공유)에서
오프라인 학습 루프 세 개를 빠르게 돌리기 위한 변경과 측정. Mac(MPS) 기본 동작은 바꾸지 않았다(`--device auto`가 Mac에선 mps).

## 바뀐 것

공통 도우미 `blackout_env/train/accel.py`:
- `pick_device("auto")`: cuda > mps > cpu, torchrun 아래에선 `LOCAL_RANK` GPU. cuda면 `cudnn.benchmark`.
- `check_amp`: 2080 Ti에서 `--amp bf16`은 에러(에뮬레이션이라 느림) → `--amp fp16` + GradScaler.
- `adamw`: cuda면 fused AdamW(같은 수식, 커널 하나).
- `load_npz`: 비압축 npz(np.savez)를 memmap으로 → 랭크 여럿이 페이지 캐시 하나를 공유(이전엔 랭크마다 RAM에 9 GB).
- `Prefetcher`(배치 스레드 + pinned) / `to_device`(pinned일 때만 non_blocking — MPS·CPU는 이전처럼 동기 복사).
- `Dist`: torchrun 환경 → NCCL 프로세스 그룹, DDP 래핑, `all_reduce_sum`, barrier (torchrun 없으면 전부 no-op).
- `cuda_host_threads` (pick_device가 cuda에서 호출): torch CPU 스레드를 4개로(직접 준 `OMP_NUM_THREADS`는 존중, torchrun의 1은 무시) — 아래 속도 절.

모델:
- `attention_block._flash_gqa`: Flash 커널이 없는 CUDA(Turing, fp32)에선 매 호출 예외→폴백 대신 바로 폴백
  (K/V repeat + SDPA mem-efficient). 예외 경로는 torch.compile이 추적을 못 해 전체 컴파일이 깨졌다. MPS 경로 그대로.

트레이너:
1. `examples/rollout_net_train.py` (RolloutNet r5/r6, S3 전문가 반복):
   - 스텝 전체(맵 uint16 해제 + 라벨 비트 풀기 + 순전파 + 손실)를 `TrainStep` 모듈 하나로 → `--compile`이 그래프 하나.
     맵 해제·라벨 언팩을 GPU로(이전엔 CPU에서 bool [B,5,32] 생성).
   - `BatchSource`: 이전 루프와 **같은 RNG 추첨**(같은 배치, 테스트로 확인). 배치는 스레드 프리페치, 또는
     `--gpu-data`(기본: cuda이고 빈 메모리 절반에 들어가면) — 분할 전체를 GPU에 올리고 인덱스만 보냄(공유 서버에서 CPU 부하에 덜 민감).
   - `--amp fp16`(+GradScaler; 맨 `--amp`는 이전처럼 bf16), `--compile [--compile-mode reduce-overhead]`(CUDA 그래프, 1장 전용),
     `--channels-last`(cuda 기본 on: cuDNN fp16 NHWC, ~5%), `--in-memory` 자동(프로세스 1개면 이전처럼 RAM, torchrun이면
     memmap을 한 번 순차로 읽어 페이지 캐시 공유 — 차가운 9 GB memmap에 무작위 접근하면 r5 학습이 2배 느렸다).
   - DDP: `torchrun --nproc_per_node N ... train` — `--batch`는 전역 배치(랭크당 batch/N), 랭크 r은 seed+1000r로 추첨.
     **손실 정규화는 전 랭크 합**(라벨 가중치 합·각도 행 가중치 합을 all_reduce/N) → DDP 평균 기울기 = 전역 배치의 가중 평균 기울기(평균의 평균이 아님).
     `loss.py`의 두 손실에 `total=` 인자 추가(기본 None = 이전과 동일).
   - `bench` 하위 명령: host 수집 / H2D / 디바이스 스텝 / 겹친 루프 ms.
2. `blackout_env/value/train.py` + `examples/value_train.py` (S1 가치망), `examples/value_branch_train.py` (S1.6):
   auto device, fused AdamW, pinned 프리페치, `--compile` = TRUNK_MODULES(MPS와 같음, 아래 함정),
   S1 트레이너는 DDP(torchrun; `--batch-size` 전역, 평가·저장은 랭크 0), 로그에 `ms_step`/`rows_s`(평가 제외).
3. 증류 학습자 `train/distill/run.py` `Learner`: auto device, fused AdamW, `--compile` = COMPILED_MODULES(MPS와 같음).
   오프라인 A/B 도구 `examples/distill_offline_ab.py`(샤드로 N스텝 학습 + 보류 샤드 라벨/각도 정확도, Unity 없음).

torch.compile 함정 (torch 2.11 + cu128, `tests/test_accel.py`):
- **MPS의 inductor 임베딩 기울기 버그는 CUDA에 없다**: fp32에서 MyModel 전체를 컴파일한 기울기 = eager (상대 오차 ≤ cuDNN
  conv 가중치 기울기 자체의 비결정성 ~1e-3 — fp64 기준으로 eager도 같은 수준; 배치 브로드캐스트 임베딩 3개의 기울기 노름 비 1.000000).
- **그러나 MyModel 전체를 fp16 autocast로 컴파일하면 출력이 틀린다**: 트렁크 토큰(최대 ~380)에서 컴파일 vs eager 차이 24–72,
  같은 입력에 호출마다 다른 값, S1 보류 평가에서 행 수천 개가 NaN(R² NaN). fp32 컴파일, eager fp16, 그리고 **서브모듈
  (graphic_encoder / vector_encoder / attention) 각각을 컴파일하면 정상**(차이 0.008 = fp16 잡음); 파생 특징·RoPE·임베딩
  각각도 정상 → 합친 `_trunk` 그래프에서만 깨진다(SDPA 백엔드를 바꿔도 같음). 원인 미확정 → **CUDA에서도 MPS와 같은
  서브모듈 컴파일**(`value/train.compile_value_net`, distill `COMPILED_MODULES`). 컴파일은 `dynamic=False`가 아니어도 되지만
  RolloutNet 스텝은 `dynamic=False`(배치 크기 고정). RolloutNet은 이 문제가 없다(fp16 + 스텝 전체 컴파일 = eager, 아래 A/B).
- 학습 손실만 보면 안 보인다(전체 컴파일 fp16 가치망의 학습 손실은 eager와 같았다) — 평가를 같이 봐야 한다.

## 속도 (ms/스텝, 행/초)

서버는 공유라 같은 설정도 다른 사용자·에이전트의 CPU 부하(측정 중 load 10 → 77, 28코어)에 따라 1.5–4배 흔들린다.
"단독"은 load ~10일 때 한 작업만 돌린 값. GPU는 남는데 CPU가 포화되면 스텝이 커널 실행(파이썬)에 묶인다 —
그때는 CUDA 그래프(`--compile-mode reduce-overhead`), `--gpu-data`, 그리고 **CPU 스레드 4개 제한**(`accel.cuda_host_threads`)이 효과가 크다:
r5 데이터, fp16+compile, load 65에서 겹친 루프 54 ms(torch 기본 = 코어당 1스레드) → 15–20 ms(2–4스레드).
torchrun의 `OMP_NUM_THREADS=1`은 반대로 한가한 호스트에서 DDP 2장 스텝을 19 → 9 ms로 늦췄다(1스레드면 호스트 쪽 복사·옵티마이저 주변 CPU 연산이 느림).

### RolloutNet (0.67M, 배치 1024, S3 재학습 설정: r5에서, s3data_x1, lr 3e-4)

| 설정 | ms/스텝 | 행/초 | 비고 |
|---|---|---|---|
| Mac MPS fp32 (r5 학습 로그) | 55 | 18.6k | r6c 재학습 로그는 126 ms(평가·부하 포함) |
| 2080 Ti, 이전 코드 eager fp32 (단독) | 26.9 | 38k | |
| 2080 Ti, 새 코드 eager fp32 (단독) | 19.2 | 53k | GPU 디코드 + 프리페치 |
| 2080 Ti, fp16 + compile (단독) | **7.1** | **144k** | 3000스텝 27초(컴파일 포함), Mac 대비 **7.7×** |
| 2080 Ti, fp16 + compile + CUDA 그래프 | 6.7–7.5 | 137–153k | load 14에서도 7.5(기본 모드는 10.5) |
| 2080 Ti ×2 DDP, 전역 1024 | 9.8 | 105k | 랭크당 512 → 스텝 오버헤드가 지배 |
| 2080 Ti ×4 DDP, 전역 1024 | 13.1 | 78k | |
| 2080 Ti ×1, 배치 4096 | 25.2 | 163k | |
| 2080 Ti ×2 DDP, 전역 2048 | 10.2 | 200k | |
| 2080 Ti ×4 DDP, 전역 4096 | 16.5 | 248k | 1장(1024) 대비 1.7× |

(DDP 행은 load 10–15, 스레드 수정 전(랭크당 7스레드). load 19에서는 ×4 전역 4096이 49 ms로 느려졌다.)
bench(`rollout_net_train.py bench`, 데이터 상주)로 본 스텝 내부: fp16+compile GPU 시간 ~7 ms 중 conv 역전파(가중치 기울기)·
NCHW↔NHWC 변환·mem-efficient 어텐션 역전파(11토큰인데 64×64 타일)가 대부분. NHWC로 변환 커널 제거 −5%, 어텐션 MATH 백엔드는 차이 없음.
**결론: RolloutNet은 1장 + `--amp fp16 --compile`**(S3 6000스텝 재학습 ≈ 45초, Mac 12분). 0.67M 망은 배치 1024를 나누면
랭크당 일이 너무 작아 DDP 이득이 없고, 전역 4096에서도 4장이 1.7×뿐(게다가 lr/스텝 재조정이 필요).

### S1 가치망 (MyModel 트렁크 5.8M, p5a에서, lr 3e-4)

| 설정 | 배치 | ms/스텝 | 행/초 |
|---|---|---|---|
| Mac MPS fp16 + compile(TRUNK) (s1 / s1b 로그, 평가 포함) | 256 | 84 / 65 | 3.0k / 3.9k |
| 2080 Ti, 이전 코드 eager fp32 | 256 | 61.8 | 4.1k |
| 2080 Ti, 새 코드 eager fp32 | 256 | 54.1 | 4.7k |
| 2080 Ti, eager fp16 (bench) | 256 | 39.6 | 6.5k |
| **2080 Ti, fp16 + compile(TRUNK_MODULES) = 채택** (bench, load ~15) | 256 | **28.5** | **9.0k** |
| 같은 설정, 배치 1024 (bench) | 1024 | 61.6 | 16.6k (GPU 바운드) |
| (참고) fp16 + 망 전체 compile — 출력이 틀려서 버림 | 256 | 19.2 | 13.3k |

DDP(같은 레시피 전역 256 = 4×64, 전역 1024 = 4×256)는 측정한 어떤 조건(load 15–70)에서도 1장보다 빠르지 않았다
(예: 전역 1024: 1장 62.5 ms vs 4장 98.8 ms, 랭크당 7스레드·전체 compile 때). 5.8M 파라미터 = 기울기 23 MB all-reduce가
P2P 없는 2080 Ti(SHM 경유, 10.8 MB에 3 ms)에서 스텝마다 ~7 ms, 거기에 4랭크 파이썬이 포화된 CPU를 나눠 쓴다.
정합성은 맞으니(아래) 한가한 서버에서 `--batch-size 1024`로 다시 재 볼 가치는 있지만, **지금 권장은 1장**.

### 증류 학습자 (P11 레시피: logits K=32 + 각도 헤드, P10에서 lr 1e-4, 오프라인 샤드)

| 설정 | 배치 | ms/스텝 | 행/초 | 조건 |
|---|---|---|---|---|
| Mac MPS 기본 (메모리 노트, 2026-09-30) | 256 | 218 | 1.2k | |
| Mac MPS fp16 + compile + q8 (P11 실제 설정) | 256 | 84 | 3.0k | |
| 2080 Ti, 이전 코드 eager fp32 | 256 | 59.3 | 4.3k | load ~40 |
| 2080 Ti, 새 코드 eager fp32 | 256 | 62.3 | 4.1k | load ~40 |
| **2080 Ti, fp16 + compile(COMPILED_MODULES)** | 256 | **41.3** | **6.2k** | load ~70 |
| 2080 Ti, fp16 + compile, GPU만 (step bench fwd+bwd) | 1024 | 66 | 15.5k | |

step bench(`distill_step_bench.py --device cuda`)로 보면 부하가 큰 호스트에서는 배치를 GPU로 옮기는 데(2 MB uint8 지도,
페이지블 메모리) 10–25 ms가 들어 스텝의 1/3을 먹는다. 서버 쪽 학습자를 만든다면 RolloutNet처럼 배치 생성을 학습 스텝과
겹치게(별도 스레드/GPU 상주 링) 하는 것이 다음 할 일. GPU 계산만 보면 배치 1024에서 15k 행/초(MPS 최선의 5×).

## 정합성 (A/B)

- **RolloutNet, S3 설정** (3000스텝, 같은 배치): 이전 코드 eager fp32 = 새 코드 eager fp32 (마지막 손실 0.386095 / 0.386095,
  보류 hit 0.8874 / 오차 13.600° / 2° 이내 0.7667 / 첫 10초 hit 0.8991 — 전부 같음) ≈ fp16+compile (0.8874 / 13.60° / 0.7667)
  ≈ CUDA 그래프 (0.8873 / 13.60°) ≈ DDP 2장·4장 (0.8872–0.8876 / 13.57–13.63°). 손실 스케일 131072 고정(건너뛴 스텝 없음).
  (컴파일 기울기: `test_compile_rollout_net_gradients_match_eager_cuda`.)
- **RolloutNet, r5 설정** (P11 샤드 + 시뮬레이터 DAgger 50% + P11 KL 0.5, r4에서 lr 1e-3, 2000스텝): 이전 코드 eager fp32
  hit 0.9014 / 10.80° / 2° 0.6431 / P11 일치 0.7792 vs fp16+compile 0.9014 / 10.81° / 0.6429 / 0.7788 (인메모리 기본 후 재실행 0.9008 / 10.87°).
- **S1 가치망** (3000스텝, 보류 2만 행): 이전 코드 eager fp32 R² 2/5/10/20초 0.580/0.607/0.613/0.648, 최종 0.761, Brier 0.0293;
  새 eager fp32 0.580/0.606/0.611/0.646, 0.760, 0.0282; **fp16 + compile(TRUNK)** 0.584/0.609/0.617/0.652, 0.760, 0.0286;
  DDP 2장(전역 256) 0.584/0.612/0.606/0.630, 0.754, 0.0294 (다른 배치 추첨 → 20초 R²만 −0.02).
  (망 전체 compile + fp16은 같은 실행에서 R² NaN — 위 함정.)
- **증류 학습자** (3000스텝, 보류 샤드 4만 행): 이전 코드 eager fp32 정확도 0.9296 / 각도 오차 9.81° / 2° 이내 0.6303 / 첫 10초 4.37°;
  새 eager fp32 0.9296 / 9.82° / 0.6303; fp16 + compile(COMPILED_MODULES) 0.9300 / 9.77° / 0.6295 / 4.35° (건너뛴 스텝 0, 손실 스케일 16384).
- 테스트: Mac `pytest -q tests` 495 통과 / 10 건너뜀(CUDA 전용 8개 포함), 서버(gpu2, 2080 Ti) `pytest -q tests` 501 통과 / 4 건너뜀 (CUDA 테스트 8개 포함).

## 명령 (서버 체크아웃에서, `export PYTHONPATH=$PWD`; 긴 런은 `setsid nohup ... &`, 끝나면 `scripts/remote_sync.sh procs <host>`)

```bash
# r5 방식 (P11 샤드 + 시뮬레이터 DAgger 50% + P11 KL, r4에서 20k) — 1장 (~2.5–4분)
CUDA_VISIBLE_DEVICES=0 python examples/rollout_net_train.py train --data checkpoints/rollout_net/data \
    --extra checkpoints/rollout_net/simdata/train_ex23.npz --extra-frac 0.5 --init checkpoints/rollout_net/r4.pt \
    --p11-kl 0.5 --lr 1e-3 --steps 20000 --batch 1024 --amp fp16 --compile --out checkpoints/rollout_net/r5_cuda.pt
# 같은 것을 4장으로 (전역 4096 = 랭크당 1024, 같은 행 수면 --steps 5000; lr은 따로 검증 필요; 1.7×)
torchrun --standalone --nproc_per_node 4 examples/rollout_net_train.py train <위와 같은 인자> --batch 4096 --steps 5000
# S3 재학습 (r5 → r6, 6000스텝): 1장, 분할이 GPU에 올라감 (~45초)
CUDA_VISIBLE_DEVICES=0 python examples/rollout_net_train.py train --data checkpoints/rollout_net/s3data_x1 \
    --init checkpoints/rollout_net/r5.pt --steps 6000 --lr 3e-4 --eval-every 2000 --amp fp16 --compile \
    --compile-mode reduce-overhead --out checkpoints/rollout_net/s3_r6x.pt
# S1 가치망: 1장 권장 (datasets/value_s1_r64 39 GB는 아직 서버에 없음: scripts/remote_sync.sh up poscat_gpu2 datasets/value_s1_r64)
CUDA_VISIBLE_DEVICES=0 python examples/value_train.py train --cache datasets/value_s1_r64 --out-dir checkpoints/value/<run> \
    --init checkpoints/distill/frozen/p5a_v22.pt --steps 40000 --amp fp16 --compile
# S1 가치망 4장 (동작·정합성 확인됨, 속도 이득은 이 공유 서버에서 미확인): 전역 1024 = 랭크당 256, 같은 행 수면 --steps 1/4
torchrun --standalone --nproc_per_node 4 examples/value_train.py train --cache datasets/value_s1_r64 \
    --out-dir checkpoints/value/<run> --init checkpoints/distill/frozen/p5a_v22.pt --batch-size 1024 \
    --steps 10000 --warmup 250 --amp fp16 --compile
# S1.6 분기 가치망 (1장; 반복마다 목표 재계산)
CUDA_VISIBLE_DEVICES=0 python examples/value_branch_train.py train --data datasets/value_s2v --init <iter k-1>.pt \
    --out-dir <dir> --iters 1 --start-iter <k> --steps 1500 --group-roots 16 --rank-weight 2 --final-beta 10 --shard-frac 0 --compile
# 증류 학습자 오프라인 A/B / 스텝 벤치
CUDA_VISIBLE_DEVICES=0 python examples/distill_offline_ab.py checkpoints/distill/p11/shards --steps 3000 \
    --init checkpoints/distill/frozen/p10_v22.pt --out reports/cuda/distill_ab.json -- --policy-head logits --action-bins 32 \
    --act-arc-bins 1 --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 --opening-weight 1.5 --lr 1e-4 --amp fp16 --compile
```
torchrun은 랭크를 자식으로 띄우고 같이 죽는다. 서버 데이터: gpu2 `~/blackout-env/checkpoints/rollout_net/{data,simdata/train_ex23.npz,
simval,s3data_x1,r4.pt,r5.pt}`, `datasets/value_s2v`, `checkpoints/distill/{p11/shards,frozen}`, `checkpoints/value/{frozen,s2v_c}`.
