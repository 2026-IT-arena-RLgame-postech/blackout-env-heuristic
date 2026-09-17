# Run 11 기준 연구 노트

이후 연구의 출발점. Run 11(특히 80k 체크포인트)을 기준선으로 두고 무엇을 바꿔 볼지, 어떻게 비교할지, 이미 해 봐서
안 된 것은 무엇인지를 한곳에 모았다. 작성 2026-09-18, 브랜치 `run11-80k`(blackout-env, blackout 모두).
세부 근거는 `docs/offline_pretrain_runs.md`(Run 11–13 섹션), `docs/reward_v2_design.md`.
`main`과 `run11-80k` 두 브랜치에 같은 노트가 있다. 체크포인트(`models/run11_step80k/`)와 파이프라인 스크립트, 빌드
경로·`--help` 수정은 `run11-80k`에만 있으니 연구는 그 브랜치에서 시작한다.

---

## 1. 기준선

| 항목 | 값 |
|---|---|
| 체크포인트 | `models/run11_step80k/step_80000.pt` (LFS, SHA-256 `9f089c1f…`) · 원본 `checkpoints/offline/20260917-130437_59475/` (5k마다 + final) |
| 재현 | `models/run11_step80k/run11_pipeline.sh build → collect → train` (README에 새 환경 절차, 클론에서 점검 완료) |
| 학습 당시 코드 | blackout-env `6b080de` · Unity blackout `94fabbd`(브랜치는 빌드 경로 수정 `5986025` 포함) |
| 알고리즘 | QMIX + IQN + SPR(5.0) + BC(α 1.0, TD3+BC식 스케일) · PER · BBF shrink-and-perturb 리셋 40k마다 · n-step 10→3, γ 0.97→0.997 · fp16 + compile · batch 64 |
| 관측 | 24×24 시맨틱 맵 + 유닛 토큰. 유닛 지역 특징은 5×5 보간 벽 샘플(±1칸, 간격 0.5). 타일 내 위상 없음 |
| 보상 | 리워드 v2 `FITTED_20260917B`(파이썬, 확정 점수 차 + 종료 ±5 + 유닛별 포텐셜을 n-step 안에서 학습기 γ로 쉐이핑) + 막힘 페널티 0.02/유닛-틱 |
| 데이터 | `heuristic_mixv6_live_20260917`(재수집 필요): 휴리스틱 mixture(V17–V19 32%), 유닛 행동 10% 균등 무작위, 배터리 소진 시 경기 종료, 스트림당 100만 행, 57GB |
| 온폴리시 | 배치의 30%, V4 상대(진영 교대), 10k마다 약 1.5만 틱, 소진 시 경기 종료, FIFO 26만 행 |
| 비용 | 학습 16.8–18.8 스텝/초(M5 Pro, MPS) → 200k 약 3–3.5시간 · 수집 18 워커 약 6분 · RAM 48GB에서 스왑과 함께 동작 |

브랜치 코드는 Run 11 이후 기능(BC 강도 가중, 가우시안 노이즈 데이터, Q 데이터셋, mixture 온폴리시)을 포함하지만
모두 기본값 꺼짐이다. 기본값이 달라진 것은 수집기 `--noise-frac`(0.1 → 0, 파이프라인이 0.1 명시) 하나.

## 2. 성능과 평가 방법

### 현재 수치

- **휴리스틱 대비**: V1(Elo 1192)에게도 한 점도 못 딴다(0/354). V4 상대 점수차 약 −70~−78.
- **Run 11 체크포인트 상대 Elo**(final = 0): 80k **+78 ± 29**, 140k +31, 160k +22, 60k −58, 180k −88, 100k −116,
  20k −153, 120k −163, 40k −189. 80k는 final에게 61/96(64%).
- **다른 런**: Run 11 final이 Run 13 final보다 +115 ± 54.
- **V4 상대 측정** 80k(`measure`, 8경기): 점수차 −72.5, 배달 70, 스폰 창고 47%(V4 25%).

### 새 런을 비교하는 법 (권장 순서)

1. **학습 중**: eval(V4, 6경기)과 온폴리시 점수차는 잡음이 크다(경기당 표준편차 약 10점, 6경기 SE 약 4점). 추세만 본다.
   eval 시드(101/202/303)와 온폴리시 시드(`10000 + 스텝`)는 런마다 같다 — 특정 시드 창에서 여러 런이 같이 무너진
   적이 있다(Run 10·11 모두 110k).
2. **체크포인트 고르기**: `run11_pipeline.sh elo <run_dir>`(체크포인트끼리, final = 0, SE 50, 약 400경기 20분).
   상위 묶음만 `--target-se 30`으로 더.
3. **Run 11 80k와 직접 비교**: `examples/elo_checkpoints.py --checkpoints base=models/run11_step80k/step_80000.pt
   new=<ckpt> --anchors --reference base --target-se 50`. 모델끼리 상대 Elo만 의미 있다(둘 다 휴리스틱 척도 밖).
4. **행동 측정**: `run11_pipeline.sh measure <ckpt>` — 벽 거리별 방향 반전, 창고별 배달, 5초 구간별 도난·분실·배달.
5. **GUI**: `run11_pipeline.sh gui|selfplay|vs`(시드 404/505/606 고정이면 런 간 같은 경기 비교 가능).

주의:
- 평가·측정 스크립트의 모델 추론은 CPU 1스레드가 빠르다(1.9 ms/행동, MPS 3.8 ms). 워커 수로 속도를 낸다.
- 평가를 강제로 끊으면 워커·Unity 프로세스가 남는다 — `pkill -f MacOS/RLGame2026` 확인.
- 점수 기록 버그(조기 종료 경기의 다음 경기 점수 0–2를 읽음)는 `583c5aa`에서 고쳐졌다. 그 전에 기록된 eval 값(Run 11
  학습 로그 포함)은 좋게 나온 것이 섞여 있다. 비교는 고친 코드로 다시 잰 값으로.

## 3. 알려진 약점 (측정값)

| 약점 | 수치 | 측정 |
|---|---|---|
| **초반 수거 속도** | 첫 10초 배달 V4의 약 1/3(Run 11 final 175 대 526, 10경기 합) | `measure_match_phases.py` |
| **외곽 창고 도난** | 외곽 창고에 넣은 것의 80–88%를 10–20초에 도둑맞음(V4는 6%). 기지(4×4) 안 창고는 못 훔친다 | 같음 |
| **스폰 창고 편중 / 창고 위치 암기** | 스폰 창고 배달 45–56%(휴리스틱 25–34%). 창고를 지도에서 지워도 그쪽으로 가고, 새로 그려 넣으면 휴리스틱 반응의 1/7 | `measure_movement_and_storage.py`, `probe_storage_reliance.py` |
| **벽 옆 진동** | 벽 바로 옆 칸 방향 반전: Hunter 47–66%, 그 외 21–35%(V17 25%/17%, V4 35%/13%). 벽 2칸 이상에서는 휴리스틱과 비슷 | `measure_movement_and_storage.py` |
| **Hunter 위치 과민** | 경계 갇힘 13.0%, 타일 중앙 0.24칸 이동에 행동 변경 18.5%(V17 8.0% / 0%). 비Hunter는 1.9% / 7.6%로 거의 해결 | `probe_boundary_chatter.py` |
| **학습이 단조롭지 않음** | 체크포인트 Elo가 80k 정점 → 100–120k −116~−163 → 140–160k 회복 → 180k −88. BBF 리셋(40k 간격) 직후 구간과 겹침 | `elo_checkpoints.py` |
| **짐 든 적 처치 보상** | 처치 틱의 팀 보상 ≈ 0(떨어진 배터리가 적 근처라 가치 이동이 없음) | `evaluate_reward_v2.py` E2 |

## 4. Run 11 이후 해 봤지만 나아지지 않은 것

| 시도 | 결과 | 교훈 |
|---|---|---|
| BC를 시연자 Elo로 가중(V19 1.94 … V1 0.18) — Run 12 | Hunter 2.9–3.6기로 편성 붕괴, 점수차 최하 | 강한 정책의 **편성까지** 복제하면 실행 못 하는 전술을 배운다 |
| BC 데이터에서 무작위 행동 제거 + 가우시안(±45°) Q 데이터 — 12b | Q 행동 간 범위 5.8(균등 노이즈 3.6–3.8), 배달·창고 다양성 약간 손해 | Q에는 **모든 방향**을 보여주는 균등 노이즈가 필요 |
| 온폴리시 상대 mixture — 12b/12d | V4로 되돌려도 나아지지 않음 | 중립. 평가는 V4 고정 유지 |
| 가우시안 BC + 균등 20% Q 데이터 — Run 13 | 80k에서 Run 11 대비 −70 대 −78로 앞섰으나 final에서 역전, Run 11 final 대비 −115 Elo | 80k 이후 Hunter 편중 재발(0.9 → 2.1기) |
| 추론 관성(직전 방향 Q가 최댓값 0.3 이내면 유지) — 측정만 | 벽 옆 반전 절반 이하 | 재학습 없는 즉효 후보, 점수 영향은 미측정(당시 점수 버그) |
| 리워드 v2(Run 10 → 11) | 짐 든 적 처치 3–9배, 상대 배달 감소, 대신 수거 감소로 점수차 비슷 | 보상만으로는 한계, 관측·데이터와 함께 |

## 5. 연구 후보 (우선순위 순)

각 후보는 Run 11 설정에서 **하나만** 바꾸고, 30–80k 짧은 런 + `elo`(Run 11 80k 대비) + `measure`로 판정하는 것을 기본으로 한다.
짧은 런은 200k 일정으로 켜고 중간에 끊는다(일정을 줄이면 BBF 리셋·어닐링 간격이 바뀐다).

1. **BBF 리셋 강도·주기** (비용 낮음, 관측 호환) — Elo가 리셋 직후 크게 떨어진다. `--reset-interval`(기본 steps/5 = 40k),
   `--reset-warmup-steps`(2000), shrink/perturb 비율(`QMIXConfig.reset_alpha_cnn` 0.8 / `reset_alpha_attention` 0.925,
   BBF 공식 0.5 — 현재 CLI 플래그 없음)을 바꿔 80k·120k·160k Elo를 본다. 리셋을 끈 런도 기준으로 하나.
2. **창고 관측 특징** (체크포인트 비호환) — 유닛별로 가장 가까운 우리 창고까지 경로 거리·방향, 자리 있음, 안전도(도둑
   계열 적과의 거리). 도난(80–88%), 스폰 편중, 창고 위치 암기를 한 번에 겨냥. 계산은 `train/reward_v2.py`의 경로 거리·
   창고 위험도에 이미 있다.
3. **벽 샘플 범위 ±2칸** (체크포인트 비호환) — 벽 옆 칸 반전이 옆옆 칸에서 벽이 안 보이다 켜지는 구조와 맞는다.
   5×5 간격 1.0 또는 9×9 간격 0.5. `model/derived_obs.py`. 2와 같은 런에 묶어도 되지만 귀속을 원하면 분리.
4. **추론 관성 / 행동 반복** (재학습 없음) — 80k에 관성 0.1/0.3과 k틱 행동 반복을 넣어 `elo`(관성 없는 80k 대비)와
   `measure`. 점수가 오르면 학습에도 평활화 손실이나 행동 반복을 넣는 근거.
5. **초반 10초 분석** (분석만) — V4 대비 느린 원인을 편성·경로·진동으로 나눠 잰다. 스폰 직후 유닛별 첫 줍기까지 틱,
   이동 거리 대비 변위.
6. **보상 보정** — 짐 든 적 처치 크레딧(처치 틱에 떨어진 배터리의 소유 이동을 반영), 외곽 창고 위험도 강화, 진치기
   과대평가. 가중치 조정은 `examples/fit_reward_v2.py`(보류 세트 포함)로.
7. **시드 분산** — 같은 설정 2–3회 반복으로 런 간 Elo 분산을 먼저 잰다. 1–3의 효과 크기가 이 분산보다 커야 의미.
8. **평가 시드 다양화** — 온폴리시 시드 `10000 + 스텝`을 런별 랜덤 오프셋으로. 특정 창고 배치에서의 붕괴가 여러 런에
   같은 위치로 찍히는 것을 막는다.

## 6. 주요 파일

| 파일 | 역할 |
|---|---|
| `blackout_env/train/offline_pretrain.py` | 학습 진입점(플래그 전부) |
| `blackout_env/train/qmix_trainer.py` | `QMIXConfig`, 손실(IQN·SPR·BC), BBF 리셋, 수집 스텝 |
| `blackout_env/model/my_model.py`, `model/derived_obs.py` | 네트워크, 유닛 지역 특징(벽 샘플) |
| `blackout_env/train/reward_v2.py` | 리워드 v2(포텐셜, 경로 거리, 캐시), `FITTED_20260917B` |
| `blackout_env/train/offline_dataset.py`, `dead_segments.py`, `onpolicy_collect.py` | 데이터 로드·소진 구간 제거·온폴리시 수집 |
| `blackout_env/train/collect_heuristic_dataset(_parallel).py` | 휴리스틱 데이터 수집 |
| `blackout_env/heuristics/mixture.py` | 수집 mixture와 가중치(Elo 주석) |
| `examples/elo_checkpoints.py` | 최소 경기 Elo(앵커 고정 또는 체크포인트 기준) |
| `examples/measure_movement_and_storage.py`, `measure_match_phases.py` | 진동·창고·구간별 도난 측정 |
| `examples/probe_boundary_chatter.py`, `probe_storage_reliance.py` | 경계 진동·창고 의존 오프라인 프로브 |
| `examples/evaluate_checkpoint_vs_heuristic.py`, `evaluate_checkpoint_vs_checkpoint.py` | GUI/헤드리스 경기 |
| `examples/fit_reward_v2.py`, `evaluate_reward_v2.py`, `record_value_matches.py` | 리워드 v2 적합·평가 |
| `models/run11_step80k/` | 체크포인트, 파이프라인, 재현 README |
