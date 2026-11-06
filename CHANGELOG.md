# Changelog

## 2.0 - 2026-11-06 (revised study)

The study was redesigned in response to peer review. Nothing of the submitted experiment is reused in the results of the revised paper.

- Reference: National Burned Area Composite (NBAC) perimeters rasterised at 10 m replace the manually digitised masks of the submitted study.
- Unit and imagery: 135 windows of about 20 x 20 km (120 fire windows of 2022-2024, 15 fire-free windows), each a single post-fire Sentinel-2 Level-2A scene selected by a stated rule, replace the 30 Google Earth Engine composites of the submitted study (+/-30-day window, 2023-2024). Cloud, shadow and snow are masked with the scene classification layer; the 1,000-DN offset of processing baseline 04.00 is removed.
- Split: three leave-one-province-out folds of whole windows, with seven validation windows per fold reserved before any result, replace the random 80/20 chip split. The test chips of the submitted Table 7 were found to be copies of training-export chips (`archive_v1_submitted/audit/test_chip_overlap.csv`), so that evaluation is withdrawn.
- Models and training: fourteen configurations in plain PyTorch under one protocol (class-weighted cross-entropy with a Dice term, AdamW, early stopping on the validation windows; Mask2Former with its native loss; a plain-cross-entropy contrast of three CNNs) replace the five ArcGIS Pro trainings of the submitted study. Nine configurations are new (DeepLabV3+, UPerNet-ConvNeXt, SegFormer, Prithvi-EO-2.0 full and LoRA, TerraMind, and the three contrast runs).
- Evaluation: whole-window scoring with per-window confusion counts, bootstrap intervals over windows, paired sign-flip tests with Holm correction, burned-area error, fire-free windows, label-free NBR and dNBR baselines, and an inference-time benchmark.
- Hardware: all runs on NVIDIA A100-SXM4-40GB GPUs of the Narval cluster (Digital Research Alliance of Canada); see `environment/`.
- Repository: `code/` (the pipeline), `data/` (manifests and fold file), `results/` (per-window results, summaries, tables, logs), `environment/`; the earlier scripts moved to `archive_v1_submitted/`; dataset, reference rasters, predicted maps and checkpoints published on Zenodo (DOIs in `README.md`).

## 1.0 - 2026-07-05 (submitted study)

Scripts for Sentinel-2 compositing in Google Earth Engine, binary labelling, batch inference on validation tiles and plotting, with the derived data on Google Drive. Kept in `archive_v1_submitted/`.
