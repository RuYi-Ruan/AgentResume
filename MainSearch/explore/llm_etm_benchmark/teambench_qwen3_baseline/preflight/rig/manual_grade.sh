#!/usr/bin/env bash
# Manual re-run of the official DIST1_queue_race grader against the mock workspace.
# Reproduces exactly what harness.run_all.grade_run() invokes:
#   bash <task>/grade.sh <workspace> <reports> <submission> <task_dir> [expected.json]
# Uses POSIX paths because this script runs under WSL bash (see REVIEW_PACKAGE.md §5).
set -u

R=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/b1_mock_wsl/ablation_runs/DIST1_queue_race/20260928_021242_22d8c765
T=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref/tasks/DIST1_queue_race

echo "# command: bash \$T/grade.sh \$R/workspace \$R/reports \$R/submission \$T \$R/reports/expected.json"
bash "$T/grade.sh" "$R/workspace" "$R/reports" "$R/submission" "$T" "$R/reports/expected.json"
echo "GRADER_EXIT=$?"
echo "# reports/score.json:"
cat "$R/reports/score.json"
