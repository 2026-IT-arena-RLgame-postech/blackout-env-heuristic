# 문서 안내

처음이면 위에서부터 읽는다. 날짜가 붙은 문서는 그 시점의 기록이라 이후 코드와 다를 수 있다.

## 시작

| 문서 | 내용 |
|---|---|
| [../models/run11_step80k/README.md](../models/run11_step80k/README.md) | Run 11 80k 체크포인트: 학습 설정, 성능, 새 환경에서 재현하는 절차 |
| [run11_research_baseline.md](run11_research_baseline.md) | 이후 연구의 기준선: 평가 방법, 약점 측정값, 이미 해 본 것, 다음 후보 |
| [gameplay_ko.md](gameplay_ko.md) / [gameplay_en.md](gameplay_en.md) | 게임 규칙 |

## 핵심 세 영역

모델 구조와 학습 루프보다 먼저 이해할 것. Unity 문서는 `blackout` 저장소(같은 부모 폴더)의 `Documentation/`에 있다.

| 영역 | 문서 |
|---|---|
| 관측(Unity ↔ Python) | `../blackout/Documentation/changes_since_team_version.md`(팀 공유 버전 이후 바뀐 것), `../blackout/Documentation/ml_agent_design.md`, 이 저장소의 `blackout_env/env/my_obs_preprocessor.py` docstring, [internals.md](internals.md) |
| 보상 | [reward_v2_design.md](reward_v2_design.md)(Run 11이 쓴 Python 리워드), `../blackout/Documentation/reward_shaping.md`(Unity 쪽 Ψ/Φ 쉐이핑, Run 11에선 끔), [heuristic_findings_for_reward_20260916.md](heuristic_findings_for_reward_20260916.md)(근거가 된 측정) |
| 휴리스틱 | [heuristic_policy_catalog_ko.md](heuristic_policy_catalog_ko.md)(V1–V19, 무엇을 언제 쓰나), [heuristic_policy_ko.md](heuristic_policy_ko.md)(V1 설계) |

## 레퍼런스

| 문서 | 내용 |
|---|---|
| [api.md](api.md) | `BlackOutEnv`, 관측 형식, 경기 함수, 체크포인트 로더 |
| [internals.md](internals.md) | 패키지 구조, 스텝 흐름, 관측을 바꿀 때 함께 고칠 곳, 테스트, 의존성 |
| [reward_v2_design.md](reward_v2_design.md) | Run 11이 쓴 리워드 v2(유닛별 포텐셜)의 설계와 가중치 적합 |
| [heuristic_policy_catalog_ko.md](heuristic_policy_catalog_ko.md) | V1–V19 휴리스틱 카탈로그, 데이터 수집 mixture, Elo |
| [heuristic_policy_ko.md](heuristic_policy_ko.md) | V1 휴리스틱 설계(이동 복구 등) |
| [heuristic_findings_for_reward_20260916.md](heuristic_findings_for_reward_20260916.md) | 휴리스틱 실험으로 측정한 게임 역학(리워드 v2의 근거) |

## 기록

| 문서 | 내용 |
|---|---|
| [offline_pretrain_runs.md](offline_pretrain_runs.md) | Run 5–13 계획·결과·진단 로그(길다, 섹션 제목으로 찾기) |

## design/ — 판단 기준과 보류된 설계

| 문서 | 내용 |
|---|---|
| [design/model_capacity_saturation_criteria.md](design/model_capacity_saturation_criteria.md) | 모델을 키워야 하는지 TensorBoard 지표로 판단하는 기준 |
| [design/qplex_migration_criteria.md](design/qplex_migration_criteria.md) | QMIX 단조성 제약이 병목인지(QPLEX 전환) 판단하는 기준 |
| [design/kda_architecture_plan.md](design/kda_architecture_plan.md) | 보류된 시간축 메모리(KDA) 아키텍처 설계. 구현 안 됨 |
| [design/reward_v2_fitted_20260917b.json](design/reward_v2_fitted_20260917b.json) | `FITTED_20260917B`(`train/reward_v2.py`) 가중치 값 |

## archive/ — 지난 단계의 기록

현재 코드와 맞지 않는 부분이 있다. 근거를 추적할 때만 본다.

| 문서 | 내용 |
|---|---|
| [archive/reward_proposal.md](archive/reward_proposal.md) | Unity 쪽 옛 리워드(팀 Ψ + 개인 Φ) 설계안. 리워드 v2 이전 |
| [archive/reward_hypotheses.md](archive/reward_hypotheses.md) | Run 5 시절 리워드 가설(H1–H10)과 검증 방법 |
| [archive/run6_diagnosis_20260916.md](archive/run6_diagnosis_20260916.md) | Run 6 전패 진단(팀 좌표계, 행동 마스크 도입 근거) |
| [archive/perf_experiments_20260913.md](archive/perf_experiments_20260913.md) | 학습 스텝 속도 실험 |
| [archive/perf_experiments_20260915.md](archive/perf_experiments_20260915.md) | 데이터 수집 속도 실험 |
| [archive/heuristic_benchmark_20260912.md](archive/heuristic_benchmark_20260912.md) | 초기 휴리스틱(V2–V5) 승격 평가 |
