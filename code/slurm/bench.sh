#!/usr/bin/env bash
#SBATCH --account=def-grandha8-ab
#SBATCH --job-name=bench
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-01:00
#SBATCH --output=logs/%x_%j.out
# Inference-time benchmark of every trained model (analysis/bench_inference.py) on the test windows of one fold.
# Usage:  sbatch slurm/bench.sh        (OUT default ~/scratch/burn/runs; DATA default ~/scratch/burn/data_postfire)
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 gdal/3.9.1 2>/dev/null || module load StdEnv/2023 gcc python/3.11 cuda gdal
for m in opencv arrow; do
  v=$(module -t avail "$m" 2>&1 | grep -E "^$m/[0-9]" | sed 's/[( ].*//' | sort -V | tail -1 || true); [ -n "$v" ] && module load "$v"
done
source ~/venv-burn/bin/activate
DATA="${DATA:-$HOME/scratch/burn/data_postfire}"; OUT="${OUT:-$HOME/scratch/burn/runs}"; mkdir -p logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python analysis/bench_inference.py --data "$DATA" --runs "$OUT" --out "$OUT/benchmark_inference_${FOLD:-F-AB}.csv" --fold "${FOLD:-F-AB}" --n-windows "${NW:-5}"
