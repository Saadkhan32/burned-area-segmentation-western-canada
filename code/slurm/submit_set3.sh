#!/usr/bin/env bash
# Set 3: newer models from the 2024-2026 literature (03_literature/newer_models_2026-10-13.md). Order = expected value per GPU-hour.
# dinov3_lora needs the gated Hugging Face weights (see env_setup.sh); drop it from the list if the licence was not accepted.
batch() { case "$1" in prithvi|prithvi_lora|terramind|dinov3_lora) echo 8;; *) echo 16;; esac; }
for MODEL in segformer deeplabv3plus prithvi_lora upernet_convnext terramind dinov3_lora prithvi; do
  for FOLD in F-AB F-BC F-SK; do
    MODEL=$MODEL FOLD=$FOLD BATCH=$(batch $MODEL) sbatch --export=ALL --job-name="burn3_${MODEL}_${FOLD}" slurm/train.sh
  done
done
