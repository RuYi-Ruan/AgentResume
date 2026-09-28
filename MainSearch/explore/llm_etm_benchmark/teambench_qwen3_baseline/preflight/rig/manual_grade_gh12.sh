#!/usr/bin/env bash
# Manual re-run of the official GH12_click_envvar_flag grader on the b4c oracle workspace,
# with wall-clock timing (review package §8 grading-time claim).
set -u
R=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/b4c_oracle_short/ablation_runs/GH12_click_envvar_flag/20260928_021531_f7c67e83
T=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref/tasks/GH12_click_envvar_flag
start=$(date +%s.%N)
bash "$T/grade.sh" "$R/workspace" "$R/reports" "$R/submission" "$T" "$R/reports/expected.json"
rc=$?
end=$(date +%s.%N)
echo "GRADER_EXIT=$rc"
echo "GRADER_WALL_SEC=$(echo "$end - $start" | bc)"
cat "$R/reports/score.json"
