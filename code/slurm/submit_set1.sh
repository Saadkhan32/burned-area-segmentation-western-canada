#!/usr/bin/env bash
# Set 1: five original architectures x three leave-one-province-out folds (15 jobs). Fast CNNs first. Run from the drac/ folder.
batch() { case "$1" in samlora) echo 4;; mask2former|prithvi|prithvi_lora|terramind|dinov3_lora) echo 8;; *) echo 16;; esac; }
for MODEL in pspnet deeplabv3 unet samlora mask2former; do
  for FOLD in F-AB F-BC F-SK; do
    MODEL=$MODEL FOLD=$FOLD BATCH=$(batch $MODEL) sbatch --export=ALL --job-name="burn_${MODEL}_${FOLD}" slurm/train.sh
  done
done
# Set 2 (loss contrast, CNNs only, plain CE without balancing):  bash slurm/submit_set2.sh
# Set 3 (newer models):  bash slurm/submit_set3.sh
