#!/usr/bin/env bash
# Set 2: loss / class-balance contrast (R1.5, R2.e, R3.5) - the three CNNs with plain cross-entropy and no class weights (configuration C2).
for MODEL in pspnet deeplabv3 unet; do
  for FOLD in F-AB F-BC F-SK; do
    MODEL=$MODEL FOLD=$FOLD TAG=C2 EXTRA="--no-class-balancing --dice-fraction 0" sbatch --export=ALL --time=0-03:00 --job-name="burnC2_${MODEL}_${FOLD}" slurm/train.sh
  done
done
