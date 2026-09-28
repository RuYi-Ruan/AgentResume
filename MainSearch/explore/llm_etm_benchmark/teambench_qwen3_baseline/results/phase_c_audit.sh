#!/usr/bin/env bash
# Phase C privilege audit -- run against every run tree of the calibration plus the
# adapter trace of the same process (PROTOCOL_REVISION.md 3.6 / 6.3).
#
# Exit code: 0 = clean (only low/info findings), 1 = medium/high/critical finding.
# A non-zero exit makes the audited calibration run VOID; it must be reported and the
# pipeline stopped -- never silently re-run.
set -u

BASE=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline
REF=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
OUT=$BASE/results

cd "$BASE" || exit 1

python3 -u audit_privileges.py \
  --tasks-dir "$REF/tasks" \
  --json "$OUT/audit_report_phase_c.json" \
  "$OUT/ablation_runs" \
  --trace "$OUT/phase_c_trace.jsonl" 2>&1 | tee "$OUT/audit_phase_c.txt"
rc=${PIPESTATUS[0]}

echo
echo "=== phase C audit exit=$rc $(date -Is) ==="
exit $rc
