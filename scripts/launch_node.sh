#!/bin/sh
# usage: launch.sh <shard-spec> [extra retrain_sweep args...]
set -e
cd ~
[ -d rcc ] || git clone -q --depth 1 -b physics-corrections https://github.com/Imamatdin/radiative-cooling-control rcc
cd rcc
PIP="python3 -m pip install -q --break-system-packages"
python3 -c "import torch" 2>/dev/null || $PIP torch --index-url https://download.pytorch.org/whl/cpu
python3 -c "import pandas, scipy, numpy" 2>/dev/null || $PIP pandas scipy numpy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
SHARD="$1"; shift
W=${GMN_CPU_LIMIT:-$(nproc)}
echo "node $(hostname) cpus=$W shard=$SHARD start $(date -u)"
python3 -u scripts/retrain_sweep.py --shard "$SHARD" --workers "$W" "$@"
echo "finished $(date -u)"
