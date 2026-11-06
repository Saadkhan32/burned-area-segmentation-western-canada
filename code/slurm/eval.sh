#!/usr/bin/env bash
#SBATCH --account=def-grandha8-ab
#SBATCH --job-name=eval
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00
#SBATCH --output=logs/%x_%j.out
# Re-run only the final test evaluation of an already trained run (eval.py), e.g. after an evaluation bug.
# Usage:  RUN=mask2former_F-AB sbatch --export=ALL --job-name=eval_mask2former_F-AB slurm/eval.sh
#         (RUN = folder name under $OUT, default OUT=~/scratch/burn/runs; DATA default ~/scratch/burn/data_postfire)
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 gdal/3.9.1 2>/dev/null || module load StdEnv/2023 gcc python/3.11 cuda gdal
for m in opencv arrow; do   # dummy-wheel packages (cv2, pyarrow) come from modules; load the newest before the venv (terratorch needs both)
  v=$(module -t avail "$m" 2>&1 | grep -E "^$m/[0-9]" | sed 's/[( ].*//' | sort -V | tail -1 || true); [ -n "$v" ] && module load "$v"
done
source ~/venv-burn/bin/activate
DATA="${DATA:-$HOME/scratch/burn/data_postfire}"; OUT="${OUT:-$HOME/scratch/burn/runs}"; mkdir -p logs "$OUT/results"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f "$OUT/$RUN/best.pt" ] || { echo "no best.pt in $OUT/$RUN"; exit 1; }
python eval.py --data "$DATA" --run "$OUT/$RUN" --out "$OUT/results" --save-maps
