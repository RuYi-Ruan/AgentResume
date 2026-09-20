#!/usr/bin/env bash
# M19 formal chain with resume support (safe to re-run; completed calls are skipped).
set -u
cd "$(dirname "$0")/.." || exit 1
PY="C:/Users/86134/.conda/envs/agentresume/python.exe"
RUN="v1_waitfix_20260915"
LOG="data/m19/$RUN/formal_chain.log"
mkdir -p "data/m19/$RUN"
for g in ${GROUPS:-1 2 3 4 6 7}; do
  echo "===== group $g formal $(date) =====" >>"$LOG"
  "$PY" scripts/m19_run.py qwen --group "$g" --split formal_test --events 50 --repeats 3 \
      --live --allow-formal 2>&1 \
      | grep -v -E "Gym has been|Please upgrade|Users of this|See the migration" >>"$LOG"
  echo "===== group $g done rc=$? $(date) =====" >>"$LOG"
done
echo "ALL DONE $(date)" >>"$LOG"
