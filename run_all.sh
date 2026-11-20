#!/usr/bin/env bash
# Re-runs every experiment reported in the paper (12 training runs), then builds tables and figures.
# Usage:  bash run_all.sh        (about 1-2 hours on a normal laptop CPU)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
run() { echo ">>> python src/qwe.py $*"; python src/qwe.py "$@" > "logs/$(echo "$*" | tr ' ' '_' | tr -d '-').log" 2>&1; tail -1 "logs/$(echo "$*" | tr ' ' '_' | tr -d '-').log"; }
for s in 13 42 2026; do
  extra=""; [ "$s" = "42" ] && extra="--save"
  run --model fact  --rank 4 --seed $s $extra
  run --model dense --seed $s $extra
  run --model real  --seed $s $extra
done
run --model fact --rank 2 --seed 42
run --model fact --rank 8 --seed 42
run --model fact_noimag --rank 4 --seed 42
for m in dense fact real; do python src/profile_res.py $m; done
python src/analyze.py
echo "All done. See results/ and figures/"
