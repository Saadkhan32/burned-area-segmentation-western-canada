# burned-area-segmentation-western-canada

Repository associated with the study:

**Remote Sensing of Burned-Area Delineation with Sentinel-2: A Geographically Independent Multi-Model Semantic Segmentation Benchmark Across Western Canada**

## Overview
This repository contains the supporting materials and any shared scripts associated with a binary burned-unburned wildfire segmentation workflow for Western Canada.

The study focused on wildfire events in:
- British Columbia
- Alberta
- Saskatchewan

## Study workflow
The overall workflow used:
- Sentinel-2A/B MSI Level-2A surface reflectance imagery
- Google Earth Engine for image acquisition and preprocessing
- ArcGIS Pro 3.5.4 for annotation, chip generation, model training, and inference
- Binary burned-unburned semantic segmentation
- Geographically independent testing across 30 wildfire events

## Models evaluated
- UNet-ResNet50
- DeepLabV3-ResNet50
- PSPNet-ResNet50
- SAM-LoRA (ViT-B)
- Mask2Former

## Data availability
Large derived data files, including shared materials related to masks, outputs, and other repository-linked resources, are available at:

[Google Drive folder](https://drive.google.com/drive/folders/19xmRAmy2QL1v9HuNxYrYrftOHhErpYtk?usp=sharing)

## Notes
This repository includes only the materials and scripts that are actually being shared by the authors. Some workflow steps were performed through the ArcGIS Pro graphical user interface rather than standalone code scripts.

## Disclaimer
This repository is provided for research and reproducibility purposes only. The materials are provided “as is,” without warranty of any kind. Results may vary depending on software versions, hardware, preprocessing choices, random seeds, data access conditions, and execution environment. The authors are not responsible for any errors, omissions, or outcomes arising from reuse of these materials.

## Citation
Please cite the associated manuscript and this repository where appropriate.
