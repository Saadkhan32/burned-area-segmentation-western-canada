# burned-area-segmentation-western-canada

Code, manifests and results of the study

**Remote sensing of burned-area delineation with Sentinel-2: a leave-one-province-out multi-model semantic segmentation benchmark across western Canada**

Muhammad S. Khan, Hassan Afzaal, Gurjit S. Randhawa, Aitazaz A. Farooque (manuscript in revision at *Computers & Geosciences*).

Version 2.0 (November 2026) replaces the repository of the submitted version of the study (July 2026). The study was redesigned during peer review; `CHANGELOG.md` lists the differences and `archive_v1_submitted/` keeps the earlier scripts and the audit of the earlier chips.

## What the study does

Fourteen segmentation configurations (U-Net, DeepLabV3, PSPNet and DeepLabV3+ with ResNet-50 encoders; UPerNet with a ConvNeXt-T encoder; SegFormer MiT-B2; SAM-LoRA ViT-B; Mask2Former Swin-L; Prithvi-EO-2.0 300M, fully fine-tuned and with LoRA adapters; TerraMind-1.0 base; and a plain-cross-entropy contrast of U-Net, DeepLabV3 and PSPNet) are trained under one protocol on 135 windows of about 20 x 20 km at 10 m in Alberta, British Columbia and Saskatchewan: 120 windows centred on large fires of 2022-2024 and 15 fire-free windows, each a single post-fire Sentinel-2 Level-2A scene stacked as nine layers (blue, green, red, NIR, SWIR1, SWIR2, NDVI, NBR, NDMI) with a reference raster derived from the National Burned Area Composite (NBAC). Three leave-one-province-out folds hold out one province at a time; every window is scored once, whole, in the fold that holds out its province. Label-free NBR and dNBR rules are scored on the same windows.

## Repository layout

| Folder | Content |
|---|---|
| `code/` | The pipeline as run on the Narval cluster: `build_dataset.py` (windows, scenes, NBAC reference), `build_prefire.py` (pre-fire scenes for the dNBR baselines), `models.py`, `train.py`, `eval.py`, `eval_dnbr.py`, `summarize.py` (bootstrap intervals, paired tests), `verify_acquisitions.py`, `slurm/` job scripts, `analysis/` (figures, tables, inference benchmark, checks), `tests/` (CPU smoke test on synthetic windows). `code/README.md` gives the order of commands. |
| `data/` | `window_manifest.csv` (the 135 windows: fire identifiers, NBAC dates, acquisition, product identifiers, processing baseline, clear and masked fractions, burned and ignored area), `fold_membership.csv` (the role of every window in each fold), `prefire_scene_manifest.csv` (the pre-fire scene of every window), `submitted_composites_acquisitions_verified.csv` (the acquisitions behind the 32 composites of the submitted study). The rasters themselves are on Zenodo (below). |
| `results/` | `runs/<configuration>_<fold>/` (run configuration incl. the training, validation and test windows, normalisation statistics, per-epoch log, checkpoint SHA-256), `runs/run_ledger.csv`, `per_window/` (per-window confusion counts and scores of every configuration and rule), `summary/` (summary tables with bootstrap intervals and paired tests), `tables/` (the tables of the paper and its supplement), `inference_benchmark/` (seconds per window, memory), `slurm_logs/` (the job output of the 42 training runs). |
| `environment/` | Software environment of the runs (modules, package versions, hardware). |
| `archive_v1_submitted/` | The scripts and README of the submitted version of the study, the manifests of its chip exports and the audit scripts that re-examined them. Kept for the record; not used by the revised study. |

## Data and trained models (Zenodo)

| Record | DOI | Content |
|---|---|---|
| Dataset, reference rasters, fold file, per-window results, predicted maps | [10.5281/zenodo.23996015](https://doi.org/10.5281/zenodo.23996015) | For every window `<window_id>_S2.tif` (nine layers, float32, masked pixels NaN), `<window_id>_label.tif` (0 unburned, 1 burned, 255 ignored), `<window_id>_scl.tif` (scene classification); the manifests; the predicted maps of the 1,890 window-model pairs. |
| Trained models | [10.5281/zenodo.23996017](https://doi.org/10.5281/zenodo.23996017) | The 42 checkpoints (`best.pt`, best validation epoch) with their run folders; SHA-256 of every checkpoint in `results/runs/run_ledger.csv`. |

## Reproducing the runs

1. `bash code/env_setup.sh` on a Digital Research Alliance of Canada cluster (modules `StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 gdal/3.9.1`; package versions in `environment/`).
2. Download the dataset from Zenodo (or rebuild it with `code/build_dataset.py`, which needs access to the Sentinel-2 catalogue and the NBAC shapefiles from `code/fetch_nbac.sh`).
3. `MODEL=pspnet FOLD=F-AB sbatch --export=ALL code/slurm/train.sh` for one run, or `code/slurm/submit_set1.sh`, `submit_set2.sh`, `submit_set3.sh` for all 42.
4. `python code/summarize.py --results runs/results --out runs/summary`, then `python code/analysis/make_figures_planA.py --runs runs --out figures`.

Every configuration was trained once per fold with seed 42; `results/runs/<run>/config.json` records the exact arguments and window lists of each run.

## Licences and sources

- Code in this repository: MIT License (`LICENSE`).
- Dataset, reference rasters, predicted maps and trained models on Zenodo: Creative Commons Attribution 4.0 International.
- Sentinel-2 Level-2A data: Copernicus Sentinel data (ESA), used under the Copernicus data licence; the scenes were read from the Earth Search STAC catalogue of the Sentinel-2 Cloud-Optimized GeoTIFF archive on AWS Open Data (Element 84).
- National Burned Area Composite: Natural Resources Canada, Canadian Forest Service (Open Government Licence - Canada).
- Pretrained weights of the backbones belong to their authors (ImageNet ResNet-50 and MiT-B2 via segmentation_models_pytorch, Swin-L/Mask2Former via Hugging Face, SAM ViT-B by Meta, Prithvi-EO-2.0 by IBM/NASA, TerraMind by IBM/ESA) and are downloaded by `code/env_setup.sh`.

## Citation

Please cite the paper when it is published and this repository (`CITATION.cff`). The Zenodo records carry their own DOIs.

## Disclaimer

The materials are provided "as is", without warranty of any kind, for research and reproducibility purposes. Results may vary with software versions, hardware and random seeds.
