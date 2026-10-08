# 체크포인트 받기

`.pt`는 `.gitignore`로 저장소에서 빠져 있다. 재현에 필요한 33개(약 1 GB)는 GitHub 릴리스 `ckpt-2026-10-08`에 묶음(`blackout-env-checkpoints-2026-10-08.tar`)으로 올려 두었다.

```bash
scripts/fetch_checkpoints.sh
```

저장소 루트에서 실행하면 `checkpoints/` 아래 원래 경로로 풀고 `docs/checkpoints_manifest.md5`로 확인한다.

| 묶음 | 경로 | 쓰임 |
|---|---|---|
| 학생 18개 | `checkpoints/distill/frozen/*.pt` | D6 → P24c 계보(설계 §§, Elo-24 §62). 챔피언 **P24c** = `p24c_v321.pt` |
| 롤아웃 망 | `checkpoints/rollout_net/{r3,r4,r5,r6}/r2*d256.pt`, `{v25,v27,v28}/*.pt`, `r5.pt`, `arch/f3_x_d2.pt` | 탐색 상대·롤아웃 모델(R1–R6). 탐색 챔피언 = P21e + `r4/r21d256.pt`(§54), 후보 P24c + `r6/r24cd256.pt`(§61) |
| 가치망 | `checkpoints/value/frozen/{s1b_value,s2v_iter5}.pt`, `value/s2v_f3xd2/iter2.pt` | 탐색 잎 가치(기본 `s2v_f3xd2/iter2`) |

나머지(학습 중간 스냅샷, `checkpoints/offline` 등 ~36 GB)는 올리지 않았다. 학습 스크립트(`reports/*/scripts/`, `scripts/*_launch.sh`)로 다시 만들 수 있다.
