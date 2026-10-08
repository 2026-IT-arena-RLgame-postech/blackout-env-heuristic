# 에이전트 작업 기억 사본 (2026-10-08 스냅샷)

Claude Code가 이 프로젝트를 진행하며 세션 사이에 유지한 작업 기억(`~/.claude/projects/.../memory/`) 중 프로젝트와 관련된 것을 복사한 것이다.
다른 사람이 진행 상황과 결정 배경을 파악하도록 둔다. **원본은 계속 바뀌므로 이 사본은 2026-10-08 시점 기준이고, 오래된 항목은 사실과 다를 수 있다** — 숫자·챔피언은 `docs/experiments_summary_ko.md`와 설계 문서(`docs/policy_v2_design_ko.md`)가 우선이다.

- 영어로 쓰였고, 파일마다 앞머리(frontmatter)에 이름·요약·종류가 있다. `[[이름]]`은 다른 기억 파일로의 연결이다.
- 종류: `project_*` = 진행 상황·계획·환경, `feedback_*` = 사용자가 정한 작업 방식·판단 규칙(이유 포함).
- 제외: 개인 작업 습관(보고 언어·형식, 사용량 예산), 이미 끝난 맥 사용 기간·전원 모드 같은 개인 기기 설정.
- 접속 정보(IP·포트·사용자명)는 지웠다. 서버 이름(`poscat_gpu1`, `poscat_gpu2`, `wsl_gpu`)은 각자의 `~/.ssh/config` 별칭이다.

## 먼저 읽을 것

| 파일 | 내용 |
|---|---|
| [project_inflight_2026_10_08_final.md](project_inflight_2026_10_08_final.md) | 최종 상태: 챔피언, 결과 위치, 다음 단계 |
| [../next_experiments_ko.md](../next_experiments_ko.md) | 다음 후보 실험·앞으로 시도할 것 전체 목록 |
| [feedback_autonomous_improvement_loop.md](feedback_autonomous_improvement_loop.md) | 개선 루프: 탐색 → 규칙화 → 증류·롤아웃 재학습 → 새 탐색 |
| [feedback_heuristic_dev_paused.md](feedback_heuristic_dev_paused.md) | 휴리스틱 개발 보류와 해제 조건 |
| [feedback_ood_robustness.md](feedback_ood_robustness.md), [feedback_weak_opp_winrate.md](feedback_weak_opp_winrate.md) | 채택 판단 규칙(OOD, 약한 상대는 승률 기준) |

## 진행 기록 (시간순)

- [project_distill_next_steps.md](project_distill_next_steps.md) — D/P 증류 계열 전체 기록(2026-09-30~10-04)
- [project_search_teacher_plan.md](project_search_teacher_plan.md) — 탐색 계획 S1–S4와 결과
- [project_kda_architecture_plan.md](project_kda_architecture_plan.md) — 기억 있는 학생(KDA) 계획
- [project_inflight_2026_10_04.md](project_inflight_2026_10_04.md), [project_inflight_2026_10_05.md](project_inflight_2026_10_05.md), [project_inflight_2026_10_06.md](project_inflight_2026_10_06.md) — 날짜별 진행 중 작업·결과(10-06 파일은 10-08까지 이어짐)

## 환경·운영

- [project_remote_gpu_servers.md](project_remote_gpu_servers.md), [project_wsl_gpu.md](project_wsl_gpu.md) — 서버 사용 규칙(gpu2 = GPU, gpu1 = CPU만, nice 등)
- [project_branch_layout.md](project_branch_layout.md), [feedback_sync_and_commit.md](feedback_sync_and_commit.md), [project_long_runs_detached.md](project_long_runs_detached.md) — 브랜치·동기화·긴 실행
- [project_heuristic_numba_perf.md](project_heuristic_numba_perf.md), [project_distill_step_perf.md](project_distill_step_perf.md), [project_unity_throughput_scaling.md](project_unity_throughput_scaling.md), [project_heuristic_variation_cadence.md](project_heuristic_variation_cadence.md) — 성능·수집 관련 함정
- [feedback_metric_choice_matters.md](feedback_metric_choice_matters.md), [feedback_use_gui_for_eval_runs.md](feedback_use_gui_for_eval_runs.md), [feedback_plot_style.md](feedback_plot_style.md), [project_focus_areas.md](project_focus_areas.md) — 측정·평가·그림 양식·우선순위
