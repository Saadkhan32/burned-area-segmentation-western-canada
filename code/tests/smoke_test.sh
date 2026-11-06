#!/usr/bin/env bash
# CPU smoke test of the whole pipeline on synthetic data (about 2-5 minutes). Run from the drac/ folder:  bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=/tmp/burn_smoke; rm -rf "$D"; mkdir -p "$D"
python tests/make_synthetic.py "$D/data"
for M in unet pspnet; do
  python train.py --data "$D/data" --fold F-AB --model $M --out "$D/runs" --epochs 3 --patience 2 --crops-per-epoch 64 --crop 64 --batch 8 --stride 32 --no-pretrained --tiny
  python eval.py --data "$D/data" --run "$D/runs/${M}_F-AB" --out "$D/results" --save-maps
done
python train.py --data "$D/data" --fold F-AB --model mask2former --out "$D/runs" --epochs 1 --crops-per-epoch 16 --crop 64 --batch 4 --stride 32 --no-pretrained --tiny
python eval.py --data "$D/data" --run "$D/runs/mask2former_F-AB" --out "$D/results"
python train.py --data "$D/data" --fold F-AB --model prithvi_lora --out "$D/runs" --epochs 1 --crops-per-epoch 4 --crop 256 --batch 2 --stride 128 --no-pretrained
python eval.py --data "$D/data" --run "$D/runs/prithvi_lora_F-AB" --out "$D/results" --batch 2
python summarize.py --results "$D/results" --out "$D/results/summary"
echo "SMOKE TEST PASSED"
