# Software and hardware environment of the runs

All 42 training runs, the evaluations, the dNBR baselines and the inference benchmark ran on the Narval cluster of the Digital Research Alliance of Canada in October 2026 (the 42 training runs on 14 October 2026).

- Hardware: one NVIDIA A100-SXM4-40GB per job (driver 580.178.04); data staged to the node's local scratch at job start.
- Modules: `StdEnv/2023`, `gcc/12.3`, `python/3.11`, `cuda/12.2`, `gdal/3.9.1`, `opencv/4.14.0`, `arrow/25.0.0`.
- Python 3.11 virtual environment built by `code/env_setup.sh` from the Alliance wheelhouse: `torch 2.14.1`, `segmentation_models_pytorch 0.5.0`, `transformers 5.18.0`, `terratorch 1.2.11`, `peft 0.21.2`, `rasterio 1.4.4`, `timm`, `segment-anything`, `pyproj`, `shapely 2`, `pystac-client`, `numpy`, `pandas`, `scikit-learn`.
- Mixed precision (fp16 autocast) for training and inference; seed 42 for every run; `results/runs/<run>/config.json` records the arguments of each run and the `torch` version it saw.
- Pretrained weights downloaded by `code/env_setup.sh`: ImageNet ResNet-50 and MiT-B2 (segmentation_models_pytorch), `facebook/mask2former-swin-large-ade-semantic` (Hugging Face), SAM ViT-B `sam_vit_b_01ec64.pth` (Meta), Prithvi-EO-2.0-300M and TerraMind-1.0-base (TerraTorch backbone registry).
- Submitted study (archived, not used by the revised results): ArcGIS Pro 3.5 deep-learning framework (`arcgis.learn 2.4.2`, CUDA build of PyTorch) on a workstation with an NVIDIA GeForce RTX GPU; see `archive_v1_submitted/`.
