#!/usr/bin/env bash
# Submit the full experiment (42 jobs) from the drac/ folder:
#   Set 1  burn1_<model>_<fold>   pspnet deeplabv3 unet samlora mask2former                     (original architectures)
#   Set 3  burn3_<model>_<fold>   segformer deeplabv3plus upernet_convnext prithvi_lora terramind prithvi   (newer models; dinov3_lora excluded: gated weights)
#   Set 2  burnC2_<model>_<fold>  pspnet deeplabv3 unet with plain cross-entropy (TAG=C2, --no-class-balancing --dice-fraction 0)
# x folds F-AB F-BC F-SK. Time limits: 3 h for the CNN / light transformer runs, 8 h for samlora, mask2former, prithvi, prithvi_lora, terramind.
# Logs: logs/<jobname>_<jobid>.out; results: $OUT/results/<run>_test_windows.csv (OUT default ~/scratch/burn/runs, see slurm/train.sh).
batch() { case "$1" in samlora) echo 4;; mask2former|prithvi|prithvi_lora|terramind|dinov3_lora) echo 8;; *) echo 16;; esac; }
hours() { case "$1" in samlora|mask2former|prithvi|prithvi_lora|terramind|dinov3_lora) echo 0-08:00;; *) echo 0-03:00;; esac; }
mkdir -p logs
for FOLD in F-AB F-BC F-SK; do
  for MODEL in pspnet deeplabv3 unet samlora mask2former; do
    MODEL=$MODEL FOLD=$FOLD BATCH=$(batch $MODEL) sbatch --export=ALL --time=$(hours $MODEL) --job-name="burn1_${MODEL}_${FOLD}" slurm/train.sh
  done
  for MODEL in segformer deeplabv3plus upernet_convnext prithvi_lora terramind prithvi; do
    MODEL=$MODEL FOLD=$FOLD BATCH=$(batch $MODEL) sbatch --export=ALL --time=$(hours $MODEL) --job-name="burn3_${MODEL}_${FOLD}" slurm/train.sh
  done
  for MODEL in pspnet deeplabv3 unet; do
    MODEL=$MODEL FOLD=$FOLD BATCH=16 TAG=C2 EXTRA="--no-class-balancing --dice-fraction 0" sbatch --export=ALL --time=0-03:00 --job-name="burnC2_${MODEL}_${FOLD}" slurm/train.sh
  done
done
echo "submitted: $(squeue -u "$USER" -h | grep -c burn) burn jobs in the queue"
