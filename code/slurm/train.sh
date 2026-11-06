#!/usr/bin/env bash
#SBATCH --account=def-grandha8-ab
#SBATCH --job-name=burn
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --output=logs/%x_%j.out
# Usage:  MODEL=unet FOLD=F-AB sbatch --export=ALL slurm/train.sh      (set DATA to the dataset folder, default ~/scratch/burn/data_postfire)
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 gdal/3.9.1 2>/dev/null || module load StdEnv/2023 gcc python/3.11 cuda gdal
for m in opencv arrow; do   # dummy-wheel packages (cv2, pyarrow) come from modules; load the newest before the venv (terratorch needs both)
  v=$(module -t avail "$m" 2>&1 | grep -E "^$m/[0-9]" | sed 's/[( ].*//' | sort -V | tail -1 || true); [ -n "$v" ] && module load "$v"
done
source ~/venv-burn/bin/activate
DATA="${DATA:-$HOME/scratch/burn/data_postfire}"; OUT="${OUT:-$HOME/scratch/burn/runs}"; mkdir -p logs "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cp -r "$DATA" "$SLURM_TMPDIR/data"      # local disk for fast reads
RUN="${MODEL}_${FOLD}${TAG:+_$TAG}"      # TAG=C2 -> runs/<model>_<fold>_C2 (loss-contrast set)
python train.py --data "$SLURM_TMPDIR/data" --fold "$FOLD" --model "$MODEL" --out "$OUT" --epochs "${EPOCHS:-120}" --patience "${PATIENCE:-10}" \
  --crops-per-epoch "${CROPS:-2000}" --batch "${BATCH:-16}" --sam-checkpoint "$HOME/weights/sam_vit_b_01ec64.pth" ${TAG:+--tag "$TAG"} ${EXTRA:-}
python eval.py --data "$SLURM_TMPDIR/data" --run "$OUT/$RUN" --out "$OUT/results" --save-maps
