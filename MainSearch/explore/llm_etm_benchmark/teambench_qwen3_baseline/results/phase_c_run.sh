#!/usr/bin/env bash
# Phase C -- official 9-run calibration (3 frozen tasks x 3 conditions x seed 0).
#
# Frozen by the reviewer rulings (PROTOCOL_REVISION.md):
#   * fixed environment = WSL Linux python3 (ruling 2)
#   * official default budgets: oracle 20 turns, restricted 30, full 20/phase + 2 remediations
#     -> NO --max-turns / --max-remediation override is passed (ruling 5)
#   * no --base-url; the third-party endpoint is reached through the runtime adapter route
#   * nothing in teambench_ref is modified; this script lives outside the clone
set -u

REF=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
BASE=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline
OUT=$BASE/results

cd "$REF" || exit 1
export PATH="$HOME/.local/bin:$PATH"
export TEAMBENCH_REF="$REF"
export PYTHONPATH=.
# Official retry default (8). Retryable failures print "[retry] n/8 ..." in this log,
# so nothing is retried silently; non-retryable errors raise immediately.
unset TEAMBENCH_MAX_RETRIES
export TEAMBENCH_TRACE_LOG="$OUT/phase_c_trace.jsonl"

echo "=== phase C start $(date -Is) ==="
echo "pwd      = $(pwd)"
echo "python3  = $(python3 --version 2>&1)"
echo "git HEAD = $(git rev-parse HEAD)"
echo "trace    = $TEAMBENCH_TRACE_LOG"
echo

# -u: the harness runs for a long time through tee (a pipe); without it stdout is
# block-buffered and the raw log would only be written when the process exits.
python3 -u "$BASE/adapters/run_qwen3.py" \
  --model qwen3-8b \
  --tasks GH12_click_envvar_flag SPEC5_config_system CROSS1_api_contract \
  --seeds 0 \
  --conditions oracle restricted full \
  --output "$OUT/ablation_results.json"
rc=$?

echo
echo "=== phase C exit=$rc  $(date -Is) ==="
exit $rc
