# Run package for the Alliance (DRAC) clusters
The training / evaluation / summary code was smoke-tested end to end on CPU with synthetic windows
(`tests/smoke_test.sh`); the catalogue and NBAC download code could not be run from the preparation environment (no access to
the satellite catalogues) and must be tried on a login node first (`--dry-run`, then `--only Betts`).
Account/cluster assumptions: Alliance group def-grandha8-ab (opportunistic allocation), Narval/Fir GPUs; adjust `slurm/train.sh`.

Order of commands (login node unless stated):
1. `bash env_setup.sh` - venv with CUDA PyTorch, segmentation_models_pytorch, transformers, segment-anything, rasterio; caches the pretrained weights (SAM ViT-B, Mask2Former Swin-L ADE20K, ImageNet ResNet-50 / MiT-B2).
2. `bash fetch_nbac.sh ~/scratch/burn/nbac` - NBAC 2020-2024 shapefiles from the CWFIS datamart (public).
3. `python verify_acquisitions.py` (optional, ~10 min) - pixel comparison that tells which Sentinel-2 acquisition each of the 32 ORIGINAL composites came from (`inputs/acquisition_table_verified.csv`); needed for Table S1 of the paper.
4. `python build_dataset.py --nbac ~/scratch/burn/nbac --out ~/scratch/burn/data_postfire --per-province 40 --negatives-per-province 5 --dry-run` - check `manifest_dryrun.csv` (NBAC dates, search ranges, 40 fire windows per province = the originals topped up with further NBAC fires >= 2,000 ha of 2022-2024, size-stratified, >= ~25 km apart, seed 42; plus 5 fire-free windows per province); then `--only Betts`; then the full build in 4 parallel processes (`--batch k/4`, k = 0..3, each writing `manifest_part<k>of4.csv`) and `python build_dataset.py --out ~/scratch/burn/data_postfire --merge-parts` to join them into `manifest.csv`.
   Scene rule: clearest Sentinel-2 L2A scene after the end of burning (last NBAC hotspot date + 7 days) within the snow-free season (Jun 1 - Oct 15); if burning ended too late, the following season. Label: NBAC polygons of the fire year = 1; scars of the two previous years, and (for next-season scenes) fires of that year that started before the acquisition = 255 (ignored); manifest columns `scene_year_offset`, `later_fires_ignored`, `ignored_ha`.
   Validation windows per fold = the three pre-registered reserved windows + 2 seeded extra-NBAC windows per development province (`train.py --val-extra-per-province 2`, recorded in each run's `config.json`).
5. `bash tests/smoke_test.sh` (CPU, ~3 min) and a 3-epoch GPU pilot: `MODEL=pspnet FOLD=F-AB EPOCHS=3 CROPS=200 sbatch --export=ALL slurm/train.sh`.
6. `bash slurm/submit_set1.sh` - 15 jobs (5 architectures x 3 folds); results land in `runs/results/*_test_windows.csv`, ledger in `runs/run_ledger.csv`.
   Then `bash slurm/submit_set2.sh` (loss contrast, 9 jobs) and `bash slurm/submit_set3.sh` (newer models: SegFormer-B2, DeepLabV3+, Prithvi-EO-2.0 LoRA and full, UPerNet-ConvNeXt, TerraMind, DINOv3-SAT LoRA; up to 21 jobs).
7. `python summarize.py --results ~/scratch/burn/runs/results --out ~/scratch/burn/runs/summary` - tables with bootstrap intervals and paired tests.
8. Copy `runs/` (without the checkpoints if large) back to the analysis computer; `analysis/make_figures_planA.py` reads it from there.

Files: `common.py` (I/O, grid, metrics, sliding inference), `build_dataset.py` (post-fire scenes, harmonised bands, NBAC labels),
`models.py` (U-Net / DeepLabV3 / PSPNet ResNet-50, SAM-LoRA ViT-B, Mask2Former Swin-L; Set 3: SegFormer-B2, DeepLabV3+, UPerNet-ConvNeXt, Prithvi-EO-2.0-300M (full / LoRA), TerraMind-1.0-base, DINOv3 ViT-L SAT-493M LoRA),
`train.py`, `eval.py`, `summarize.py`, `verify_acquisitions.py`, `inputs/` (event table, reconstructed acquisition table, composite blocks).
Design rules implemented: whole windows as units; held-out province never read during a fold; three reserved validation windows per fold
(early stopping on their mean burned F1); normalisation and class weights from training windows only; class-weighted CE + 0.5 Dice for the
CNNs and SAM-LoRA, native loss for Mask2Former; every test pixel scored once (overlapping tiles averaged); prior-year burn scars ignored (255).
