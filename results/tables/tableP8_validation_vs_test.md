| Model                            | Fold   |   Validation windows |   Best validation mean F1 |   Epoch of best |   Test mean F1 (held-out province) |   Test median F1 |   Drop (validation - test) |
|:---------------------------------|:-------|---------------------:|--------------------------:|----------------:|-----------------------------------:|-----------------:|---------------------------:|
| Mask2Former (Swin-L)             | F-AB   |                    7 |                     0.941 |              14 |                              0.914 |            0.933 |                      0.026 |
| Mask2Former (Swin-L)             | F-BC   |                    7 |                     0.951 |              16 |                              0.873 |            0.916 |                      0.078 |
| Mask2Former (Swin-L)             | F-SK   |                    7 |                     0.91  |              10 |                              0.94  |            0.948 |                     -0.031 |
| Prithvi-EO-2.0, LoRA             | F-AB   |                    7 |                     0.936 |               8 |                              0.922 |            0.936 |                      0.015 |
| Prithvi-EO-2.0, LoRA             | F-BC   |                    7 |                     0.948 |              25 |                              0.87  |            0.931 |                      0.078 |
| Prithvi-EO-2.0, LoRA             | F-SK   |                    7 |                     0.912 |              16 |                              0.929 |            0.94  |                     -0.017 |
| SegFormer (MiT-B2)               | F-AB   |                    7 |                     0.94  |              19 |                              0.916 |            0.934 |                      0.024 |
| SegFormer (MiT-B2)               | F-BC   |                    7 |                     0.942 |              14 |                              0.875 |            0.932 |                      0.067 |
| SegFormer (MiT-B2)               | F-SK   |                    7 |                     0.906 |              20 |                              0.93  |            0.941 |                     -0.024 |
| U-Net, plain CE                  | F-AB   |                    7 |                     0.938 |               9 |                              0.905 |            0.933 |                      0.033 |
| U-Net, plain CE                  | F-BC   |                    7 |                     0.945 |              18 |                              0.874 |            0.926 |                      0.071 |
| U-Net, plain CE                  | F-SK   |                    7 |                     0.916 |              20 |                              0.93  |            0.941 |                     -0.014 |
| UPerNet (ConvNeXt-T)             | F-AB   |                    7 |                     0.935 |               8 |                              0.897 |            0.934 |                      0.038 |
| UPerNet (ConvNeXt-T)             | F-BC   |                    7 |                     0.945 |              20 |                              0.878 |            0.931 |                      0.068 |
| UPerNet (ConvNeXt-T)             | F-SK   |                    7 |                     0.918 |              37 |                              0.927 |            0.939 |                     -0.009 |
| DeepLabV3 (ResNet-50)            | F-AB   |                    7 |                     0.934 |               8 |                              0.908 |            0.931 |                      0.026 |
| DeepLabV3 (ResNet-50)            | F-BC   |                    7 |                     0.94  |              14 |                              0.883 |            0.929 |                      0.057 |
| DeepLabV3 (ResNet-50)            | F-SK   |                    7 |                     0.884 |               5 |                              0.909 |            0.919 |                     -0.026 |
| U-Net (ResNet-50)                | F-AB   |                    7 |                     0.932 |               9 |                              0.917 |            0.933 |                      0.015 |
| U-Net (ResNet-50)                | F-BC   |                    7 |                     0.934 |               9 |                              0.876 |            0.924 |                      0.058 |
| U-Net (ResNet-50)                | F-SK   |                    7 |                     0.898 |              13 |                              0.894 |            0.914 |                      0.003 |
| PSPNet (ResNet-50)               | F-AB   |                    7 |                     0.932 |               8 |                              0.901 |            0.927 |                      0.031 |
| PSPNet (ResNet-50)               | F-BC   |                    7 |                     0.934 |              25 |                              0.875 |            0.93  |                      0.059 |
| PSPNet (ResNet-50)               | F-SK   |                    7 |                     0.901 |              13 |                              0.904 |            0.924 |                     -0.004 |
| DeepLabV3+ (ResNet-50)           | F-AB   |                    7 |                     0.936 |              21 |                              0.899 |            0.93  |                      0.037 |
| DeepLabV3+ (ResNet-50)           | F-BC   |                    7 |                     0.939 |              14 |                              0.869 |            0.922 |                      0.069 |
| DeepLabV3+ (ResNet-50)           | F-SK   |                    7 |                     0.888 |               7 |                              0.91  |            0.926 |                     -0.022 |
| SAM-LoRA (ViT-B)                 | F-AB   |                    7 |                     0.936 |              13 |                              0.91  |            0.929 |                      0.026 |
| SAM-LoRA (ViT-B)                 | F-BC   |                    7 |                     0.948 |              20 |                              0.854 |            0.929 |                      0.094 |
| SAM-LoRA (ViT-B)                 | F-SK   |                    7 |                     0.913 |              11 |                              0.909 |            0.932 |                      0.004 |
| DeepLabV3, plain CE              | F-AB   |                    7 |                     0.938 |              19 |                              0.895 |            0.928 |                      0.044 |
| DeepLabV3, plain CE              | F-BC   |                    7 |                     0.94  |              14 |                              0.845 |            0.926 |                      0.094 |
| DeepLabV3, plain CE              | F-SK   |                    7 |                     0.903 |              17 |                              0.917 |            0.94  |                     -0.013 |
| PSPNet, plain CE                 | F-AB   |                    7 |                     0.935 |              11 |                              0.895 |            0.926 |                      0.04  |
| PSPNet, plain CE                 | F-BC   |                    7 |                     0.931 |               6 |                              0.837 |            0.91  |                      0.094 |
| PSPNet, plain CE                 | F-SK   |                    7 |                     0.902 |              13 |                              0.909 |            0.93  |                     -0.007 |
| Prithvi-EO-2.0, full fine-tuning | F-AB   |                    7 |                     0.923 |              15 |                              0.877 |            0.908 |                      0.046 |
| Prithvi-EO-2.0, full fine-tuning | F-BC   |                    7 |                     0.929 |              39 |                              0.866 |            0.889 |                      0.062 |
| Prithvi-EO-2.0, full fine-tuning | F-SK   |                    7 |                     0.89  |               9 |                              0.861 |            0.911 |                      0.029 |
| TerraMind, full fine-tuning      | F-AB   |                    7 |                     0.927 |               8 |                              0.866 |            0.912 |                      0.061 |
| TerraMind, full fine-tuning      | F-BC   |                    7 |                     0.903 |               4 |                              0.843 |            0.871 |                      0.059 |
| TerraMind, full fine-tuning      | F-SK   |                    7 |                     0.89  |               9 |                              0.834 |            0.898 |                      0.056 |
