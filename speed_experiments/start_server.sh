#!/bin/bash
# Standalone benchmark server — separate port/log from the real pipeline's
# servers, does not touch similar_samples_llms_eval_lite/ in any way.
set -e
PORT=8010
GPU=0
LOG=${SIMIT_ROOT}/speed_experiments/server.log
BIN=${SIMIT_ROOT}/.venv/bin/mstar

pkill -f "mstar serve.*--port $PORT" 2>/dev/null || true
sleep 2

echo "=== starting mstar serve bagel on port $PORT, gpu $GPU at $(date) ===" > "$LOG"
nohup "$BIN" serve bagel --gpus "$GPU" --port "$PORT" >> "$LOG" 2>&1 &
echo $! > ${SIMIT_ROOT}/speed_experiments/server.pid
echo "started, pid=$(cat ${SIMIT_ROOT}/speed_experiments/server.pid), log=$LOG"
