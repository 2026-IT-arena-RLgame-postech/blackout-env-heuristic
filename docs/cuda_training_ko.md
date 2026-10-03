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
- `Dist`: torchrun 환경 → NCCL 프로세스 그룹, DDP 래핑, `all_reduce_sum`, barrier. **torchrun이 거는
  `OMP_NUM_THREADS=1`이 2080 Ti 스텝을 절반으로 늦춰서**(측정: DDP 2장 20 ms → 9 ms) intra-op 스레드를 `코어/랭크`(≤8)로 되돌린다.

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

서버는 공유라 다른 사용자·에이전트의 CPU 부하(load 10–37/28코어)에 따라 같은 설정이 1.5–3배 흔들린다.
아래 "단독"은 load ~10일 때 한 작업만 돌린 값. CPU가 포화되면 커널 실행 오버헤드 때문에 CUDA 그래프(`reduce-overhead`)나 `--gpu-data`가 덜 느려진다.

### RolloutNet (0.67M, 배치 1024, S3 재학습 설정: r5에서, s3data_x1, lr 3e-4)

| 설정 | ms/스텝 | 행/초 | 비고 |
|---|---|---|---|
| Mac MPS fp32 (r5 학습 로그) | 55 | 18.6k | r6c 재학습 로그는 126 ms(평가·부하 포함) |
| 2080 Ti, 이전 코드 eager fp32 | 26.9 | 38k | |
| 2080 Ti, 새 코드 eager fp32 | 19.2 | 53k | GPU 해제 + 프리페치 |
| 2080 Ti, fp16 + compile | **7.1** | **144k** | 3000스텝 27초(컴파일 포함), Mac 대비 7.7× |
| 2080 Ti, fp16 + compile + CUDA 그래프 | 6.7 | 153k | bench(데이터 상주): GPU 바운드 ~6.7 ms |
| 2080 Ti ×2 DDP, 전역 1024 | 9.8 | 105k | 랭크당 512 → 스텝 오버헤드가 지배 |
| 2080 Ti ×4 DDP, 전역 1024 | 13.1 | 78k | |
| 2080 Ti ×1, 배치 4096 | 25.2 | 163k | |
| 2080 Ti ×2 DDP, 전역 2048 | 10.2 | 200k | |
| 2080 Ti ×4 DDP, 전역 4096 | 16.5 | 248k | 1장 1024 대비 1.7× |

(DDP 행은 load 10–15에서. load 19일 때는 ×4 전역 4096이 49 ms까지 느려졌다 — 4랭크 × 파이썬 실행이 CPU 경쟁에 가장 약함.)
**결론: RolloutNet은 1장 + fp16 + compile이 정답**(S3 6000스텝 재학습 ≈ 45초; Mac 12분). 0.67M 망은 배치 1024에서 랭크를
나누면 랭크당 일이 너무 작아 DDP 이득이 없고, 큰 전역 배치(4096)에서도 4장이 1.5–1.7×뿐(그리고 lr/스텝 재조정이 필요).

### S1 가치망 (MyModel 트렁크 5.8M, p5a에서, lr 3e-4)

| 설정 | 배치 | ms/스텝 | 행/초 |
|---|---|---|---|
| Mac MPS fp16 + compile(TRUNK) (s1 / s1b 로그, 평가 포함) | 256 | 84 / 65 | 3.0k / 3.9k |
| 2080 Ti, 이전 코드 eager fp32 | 256 | 61.8 | 4.1k |
| 2080 Ti, 새 코드 eager fp32 | 256 | 54.1 | 4.7k |
| 2080 Ti, fp16 + compile(TRUNK만, MPS 방식) | 256 | 28.5 | 9.0k |
| 2080 Ti, fp16 + compile(전체) | 256 | 19.2–22.2 | 11.5–13.3k |
| 2080 Ti, fp16 + compile(전체) + CUDA 그래프 | 256 | 16.4 | 15.6k |
| 2080 Ti, fp16 + compile(전체) | 1024 | 58.6 | 17.5k (GPU 바운드) |
DDP_VALUE_ROWS

### 증류 학습자 (P11 레시피: logits K=32 + 각도 헤드, 배치 256, P10에서 lr 1e-4)

DISTILL_ROWS

## 정합성 (A/B)

- **RolloutNet** (3000스텝, 같은 배치): 이전 코드 eager fp32 = 새 코드 eager fp32 (마지막 손실 0.386095 vs 0.386095, 보류 hit
  0.8874 / 오차 13.600° / 2° 이내 0.7667 완전히 같음) ≈ fp16+compile (0.8874 / 13.60° / 0.7667) ≈ CUDA 그래프 (0.8873 / 13.60°)
  ≈ DDP 2장·4장 (0.8872–0.8876 / 13.57–13.63°). 손실 스케일 131072 고정(건너뛴 스텝 없음).
R5_AB
- **S1 가치망**: 이전 코드 eager fp32 vs 새 eager fp32: 보류 R² 2/5/10/20초 0.580/0.607/0.613/0.648 vs 0.580/0.606/0.611/0.646,
  Brier 0.0293 vs 0.0282. VALUE_AB
- **증류 학습자** (3000스텝, 보류 샤드 4만 행): 이전 코드 eager fp32 정확도 0.9296 / 각도 오차 9.81° / 2° 이내 0.6303 / 첫 10초 4.37°,
  새 eager fp32 0.9296 / 9.82° / 0.6303, fp16 + 전체 compile 0.9300 / 9.77° / 0.6297 / 4.35° (건너뛴 스텝 0).
- 테스트: Mac `pytest -q tests` 495 통과 / 9 건너뜀(CUDA 전용), 서버 TESTS_SERVER.

## 4장으로 돌리는 명령 (서버 체크아웃에서, `PYTHONPATH=$PWD`)

```bash
# r5 방식 (P11 샤드 + 시뮬레이터 DAgger 50% + P11 KL, r4에서 20k) — 1장 권장
CUDA_VISIBLE_DEVICES=0 python examples/rollout_net_train.py train --data checkpoints/rollout_net/data \
    --extra checkpoints/rollout_net/simdata/train_ex23.npz --extra-frac 0.5 --init checkpoints/rollout_net/r4.pt \
    --p11-kl 0.5 --lr 1e-3 --steps 20000 --batch 1024 --amp fp16 --compile --out checkpoints/rollout_net/r5_cuda.pt
# 4장 (전역 배치 4096 = 랭크당 1024; 같은 에폭이면 --steps 5000, lr은 별도 검증 필요)
torchrun --standalone --nproc_per_node 4 examples/rollout_net_train.py train ... --batch 4096 --steps 5000 --amp fp16 --compile
# S3 재학습 (r5 → r6, 6000스텝): 1장, 데이터가 GPU에 올라감
CUDA_VISIBLE_DEVICES=0 python examples/rollout_net_train.py train --data checkpoints/rollout_net/s3data_x1 \
    --init checkpoints/rollout_net/r5.pt --steps 6000 --lr 3e-4 --eval-every 2000 --amp fp16 --compile --out ...
# S1 가치망 4장 (전역 1024 = 랭크당 256; 배치 256 레시피와 같은 행 수면 --steps 1/4)
torchrun --standalone --nproc_per_node 4 examples/value_train.py train --cache datasets/value_s1_r64 \
    --out-dir checkpoints/value/<run> --init checkpoints/distill/frozen/p5a_v22.pt --batch-size 1024 \
    --steps 10000 --lr 3e-4 --warmup 250 --amp fp16 --compile
# S1.6 분기 가치망 (1장; 반복마다 목표 재계산)
CUDA_VISIBLE_DEVICES=0 python examples/value_branch_train.py train --data datasets/value_s2v --init <iter k-1>.pt \
    --out-dir <dir> --iters 1 --steps 1500 --group-roots 16 --rank-weight 2 --final-beta 10 --shard-frac 0 --amp fp16 --compile
```
torchrun 런은 `setsid`로 띄우고 끝나면 `scripts/remote_sync.sh procs <host>`로 고아 확인. 랭크는 torchrun과 함께 죽는다.
