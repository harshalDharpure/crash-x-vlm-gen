#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/architecture_v4
PIDFILE=results/architecture_v4/worker.pid
LOGFILE=results/architecture_v4/worker.log
exec 9>results/architecture_v4/launcher.lock
flock -n 9 || { echo 'Launcher already active'; exit 1; }
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Worker already running: $(cat "$PIDFILE")"
    exit 0
fi
GPU="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2,2nr -k3,3n | head -1 | cut -d, -f1 | tr -d ' ')"
nohup setsid env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false \
    "$ROOT/.venv/bin/python" -u -m crashx.run_architecture_suite >>"$LOGFILE" 2>&1 9>&- </dev/null &
echo $! >"$PIDFILE"
echo "GPU=$GPU PID=$(cat "$PIDFILE") log=$ROOT/$LOGFILE"
