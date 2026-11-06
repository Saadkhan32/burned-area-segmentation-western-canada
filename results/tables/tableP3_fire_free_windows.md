| Model                                   |   Fire-free windows |   Mean mapped burn (ha) |   Largest mapped burn (ha) | Mean mapped fraction of the window   |
|:----------------------------------------|--------------------:|------------------------:|---------------------------:|:-------------------------------------|
| Mask2Former (Swin-L)                    |                  15 |                      51 |                        692 | 0.12 %                               |
| Prithvi-EO-2.0, LoRA                    |                  15 |                      79 |                       1147 | 0.19 %                               |
| SegFormer (MiT-B2)                      |                  15 |                       6 |                         48 | 0.01 %                               |
| U-Net, plain CE                         |                  15 |                      88 |                        488 | 0.21 %                               |
| UPerNet (ConvNeXt-T)                    |                  15 |                     217 |                       3144 | 0.52 %                               |
| DeepLabV3 (ResNet-50)                   |                  15 |                      95 |                       1248 | 0.23 %                               |
| U-Net (ResNet-50)                       |                  15 |                     862 |                       8475 | 2.08 %                               |
| PSPNet (ResNet-50)                      |                  15 |                     149 |                       1083 | 0.36 %                               |
| DeepLabV3+ (ResNet-50)                  |                  15 |                     134 |                       1905 | 0.33 %                               |
| SAM-LoRA (ViT-B)                        |                  15 |                      45 |                        514 | 0.11 %                               |
| DeepLabV3, plain CE                     |                  15 |                      92 |                        624 | 0.22 %                               |
| PSPNet, plain CE                        |                  15 |                     349 |                       1800 | 0.83 %                               |
| Prithvi-EO-2.0, full fine-tuning        |                  15 |                     187 |                       1542 | 0.44 %                               |
| TerraMind, full fine-tuning             |                  15 |                    1396 |                      13101 | 3.36 %                               |
| Otsu threshold on dNBR (pre-fire scene) |                  15 |                   17687 |                      42650 | 43.33 %                              |
| dNBR > 0.1 rule (pre-fire scene)        |                  15 |                   10046 |                      36702 | 24.83 %                              |
| Otsu threshold on NBR                   |                  15 |                   15054 |                      26460 | 35.8 %                               |
| NBR < 0 rule                            |                  15 |                    3872 |                      17923 | 9.14 %                               |
