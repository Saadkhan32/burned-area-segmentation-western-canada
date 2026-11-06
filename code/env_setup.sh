#!/usr/bin/env bash
# One-time environment on an Alliance (DRAC) cluster (Narval / Fir / Rorqual / Nibi). Run on a login node:  bash env_setup.sh
# Creates ~/venv-burn with PyTorch (CUDA), segmentation_models_pytorch, transformers, segment-anything, rasterio, pystac-client.
# Re-runnable: an existing ~/venv-burn is reused. The OpenCV module must be loaded BEFORE the venv is activated (terratorch ->
# albumentations -> cv2; on Alliance clusters opencv-python-headless is a dummy wheel that points to the module).
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 gdal/3.9.1 2>/dev/null || module load StdEnv/2023 gcc python/3.11 cuda gdal
# opencv (cv2) and arrow (pyarrow) are "dummy wheels" on Alliance clusters: their modules must be loaded before the venv
load_newest() { local m; m=$(module -t avail "$1" 2>&1 | grep -E "^$1/[0-9]" | sed 's/[( ].*//' | sort -V | tail -1 || true)
  if [ -n "$m" ]; then module load "$m" && echo "loaded $m"; else echo "WARNING: no $1 module found (module -t avail $1)"; fi; }
load_newest opencv; load_newest arrow
module list 2>&1 | tr -s ' ' | head -20
[ -d ~/venv-burn ] || python -m venv ~/venv-burn
source ~/venv-burn/bin/activate
pip install --no-index --upgrade pip
# wheels from the Alliance wheelhouse first (fast, no internet needed, no source builds), PyPI for the rest (login nodes have internet)
pip install --no-index torch torchvision numpy pandas scipy scikit-learn pillow rasterio pyproj shapely tabulate || true
for p in pydantic pydantic-core timm transformers huggingface-hub safetensors tokenizers lightning torchmetrics geopandas pyogrio rioxarray xarray scikit-image \
         tensorboard rich einops pycocotools tifffile jsonschema requests tqdm aiohttp multidict yarl matplotlib kornia accelerate omegaconf hydra-core jsonargparse \
         segmentation-models-pytorch segment-anything pystac-client planetary-computer pyshp; do
  pip install --no-index "$p" >/dev/null 2>&1 && echo "wheelhouse: $p" || echo "not in wheelhouse (PyPI later): $p"
done
pip install segmentation-models-pytorch timm transformers segment-anything pystac-client planetary-computer pyshp tabulate peft "terratorch[peft]"
# the hdf5 / netcdf modules pulled in by gdal put NumPy-1 builds of numexpr, cftime and h5py on the path (pandas / xarray / terratorch
# then fail to import); NumPy-2 builds from the wheelhouse installed in the venv take precedence
for p in numexpr cftime h5py; do
  pip install --no-index --ignore-installed --no-deps "$p" >/dev/null 2>&1 && echo "venv copy (wheelhouse): $p" || { pip install --ignore-installed --no-deps "$p" && echo "venv copy (PyPI): $p"; }
done
python - <<'PY'
import numexpr, cftime, h5py, pandas, xarray
import torch, segmentation_models_pytorch, transformers, rasterio, pystac_client, shapefile, cv2, pyarrow, terratorch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| smp', segmentation_models_pytorch.__version__, '| transformers', transformers.__version__,
      '| rasterio', rasterio.__version__, '| cv2', cv2.__version__, '| terratorch', getattr(terratorch, '__version__', '?'))
PY
# pretrained weights (login node, internet): SAM ViT-B, Mask2Former Swin-L (ADE20K semantic), ImageNet ResNet-50 via smp/timm
mkdir -p ~/weights && cd ~/weights
[ -f sam_vit_b_01ec64.pth ] || wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python - <<'PY'
import h5py   # first: see the note in common.py (HDF5 1.14.6 vs the gdal module's 1.14.2)
import segmentation_models_pytorch as smp
from transformers import Mask2FormerForUniversalSegmentation
smp.Unet('resnet50', encoder_weights='imagenet', in_channels=9, classes=2); smp.Segformer('mit_b2', encoder_weights='imagenet', in_channels=9, classes=2)
smp.UPerNet('tu-convnext_tiny', encoder_weights='imagenet', in_channels=9, classes=2)
Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-ade-semantic')
from terratorch.registry import BACKBONE_REGISTRY
BACKBONE_REGISTRY.build('prithvi_eo_v2_300', pretrained=True)            # NASA/IBM Prithvi-EO-2.0-300M (Hugging Face, open)
BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=True, modalities=['S2L2A'])   # ESA/IBM TerraMind-1.0-base (open)
print('weights cached')
PY
# DINOv3 (SAT-493M) is gated: accept the licence at https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m, then
#   hf auth login   (paste a read token)   and run:  python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-sat493m')"

echo "environment ready: source ~/venv-burn/bin/activate"
