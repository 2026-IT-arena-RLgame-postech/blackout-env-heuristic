# RL fine-tune 요약 (2026-10-20 … 10-31)

증류 학생 P24c(V32.1 교사)에서 시작해 PPO로 미세조정한 결과 정리. 세부 수치는 각 문서에 있다.

## 최고 정책: L1 (`checkpoints/rl/l1_league/final.pt`)

계보: P24c → league B (PPO, 9-상대 풀, PFSP hard, per-opponent advantage centering, gated self-play snapshot)
→ **E4** (β 0.02 → 0.01) → **F1** (+V33을 풀에) → **F2** (+lr 6e-6) → **L1** (F2 시작, 풀 9 B + V33 + F2 + F3, 20M 결정).

| 비교 (256 시드 × 2면, 짝 비교) | L1 결과 |
|---|---|
| avg3 (p24c / p22 / V32.1) vs F2 | +4.2±1.0 |
| H2H vs F2 | +19.1±2.2 (승률 67.5 %) |
| H2H vs F3 | −6.3±2.4 (승률 44.1 %) |
| V33 (최강 휴리스틱) vs E4 | +52.8 (F2 +45.5, F3 +33.3) |
| OOD-26 vs F2 | PASS: 평균 +72.6, 최악 +56.5 (v30_h4hunt) |

## 무엇이 효과 있었나

| 실험 | 결과 | 문서 |
|---|---|---|
| KL β 0.02 → 0.01 | +4.6 avg3 → 기본값 | ppo_ablation_results.md |
| β 0.005 / 0 | 더 나쁨 (β 0은 특화) | 〃 |
| GAE λ 0.99, offset/fine 방향 헤드, KL anchor reset | 효과 없거나 손해 | 〃 |
| V33을 상대 풀에 (F1) | V33 −12 → +0.3 | 〃 |
| lr 3e-6 → 6e-6 (HPO) | final +6.2/+4.5, 학습 곡선 +2.2 → 기본값 | hpo_results.md |
| 그 외 HPO (epochs, minibatch, clip, lr 1.2e-5/1.5e-6) | 효과 없거나 손해 | 〃 |
| F2·F3를 리그 상대로 (L1) | 모든 지표에서 F2보다 우세 | ppo_ablation_results.md |
| Softmax Collapse 수정 (arXiv 2610.31642) | 최대 마진 13.2 < fp32 한계 16.6 → 불필요 | softmax_margin_check.md |

## 학습 속도

- 프로파일 (ppo_perf2.md): 한 iteration 22 s = 롤아웃 trunk 9.5 (망 forward 8.3, 휴리스틱은 겹쳐서 0.07) + leaf 1.4 + update 11.1.
  배치를 반으로 나눠 휴리스틱과 추론을 겹치는 안은 이득 ≤5 % → 보류.
- **비동기 actor-learner + decoupled PPO = 기본값** (`async_collect=auto`): GPU 2개가 보이면 두 번째 GPU에서 다음 배치를
  수집하면서 첫 번째 GPU에서 학습. 1.8× 빠르고 (16.6 vs 30.4 s/it) 학습 곡선은 동기와 노이즈 범위 내 (+0.7/+0.6).
  decoupled 없이 그냥 비동기는 −3점이었다. 실행: `GPU=2,0` (비동기) 또는 `GPU=2` (동기).

## 릴리즈

- `rl-l1-2026-10-31`: L1의 공식 형식 제출물 (`policy.py` + `checkpoint.pt`, 탐색 없음). 빌드:
  `examples/build_official_submission.py --checkpoint checkpoints/rl/l1_league/final.pt`.

## 다음 후보

- L2 리그 (L1 시작, L1을 풀에 추가) — GPU0이 비면 비동기 2-GPU로.
- F3에 대한 H2H 열세(−6) 해소: 상대 가중치 조정.
