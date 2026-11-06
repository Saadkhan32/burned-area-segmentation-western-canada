#!/usr/bin/env python3
"""Table P10 - inference benchmark per configuration (Supplementary Table S8).
Reads runs/benchmark_inference_F-AB.csv (analysis/bench_inference.py, Narval job 5615762: five Alberta test windows of fold
F-AB, 256-px tiles, batch 16, fp16 autocast, one A100-SXM4-40GB) and writes one row per configuration in the order of Table 4
with the total parameter count, seconds per window at stride 128 (the protocol of Section 2.7) and 256, throughput at stride 128
and peak GPU memory; the NBR < 0 rule (CPU, numpy) is the last row.
Usage: python analysis/make_table_inference.py <benchmark_inference_F-AB.csv> <out_dir>"""
import csv, sys, os

LABEL = [('unet', 'U-Net (ResNet-50)'), ('deeplabv3', 'DeepLabV3 (ResNet-50)'), ('pspnet', 'PSPNet (ResNet-50)'),
         ('deeplabv3plus', 'DeepLabV3+ (ResNet-50)'), ('upernet_convnext', 'UPerNet (ConvNeXt-T)'), ('samlora', 'SAM-LoRA (ViT-B)'),
         ('mask2former', 'Mask2Former (Swin-L)'), ('segformer', 'SegFormer (MiT-B2)'), ('prithvi', 'Prithvi-EO-2.0, full fine-tuning'),
         ('prithvi_lora', 'Prithvi-EO-2.0, LoRA'), ('terramind', 'TerraMind, full fine-tuning'), ('unet_C2', 'U-Net, plain CE'),
         ('deeplabv3_C2', 'DeepLabV3, plain CE'), ('pspnet_C2', 'PSPNet, plain CE')]
COLS = ['Model', 'Parameters (M)', 'Time per window, stride 128 (s)', 'Time per window, stride 256 (s)', 'Throughput, stride 128 (Mpx/s)', 'Peak GPU memory (GB)']

def main(src, out_dir):
    rows = list(csv.DictReader(open(src)))
    by = {}
    for r in rows: by[(r['model'], r['stride'])] = r
    out = []
    for key, label in LABEL:
        a, b = by[(key, '128')], by[(key, '256')]
        assert a['n_windows'] == b['n_windows'] == '5' and a['batch'] == '16' and a['precision'] == 'fp16 autocast' and 'A100' in a['device']
        out.append([label, a['params_M'], f"{float(a['s_per_window']):.2f}", f"{float(b['s_per_window']):.2f}", f"{float(a['Mpx_per_s']):.1f}", f"{float(a['gpu_mem_GB']):.1f}"])
    n = by[('NBR<0', '-')]
    out.append(['NBR < 0 rule (CPU, NumPy)', '-', f"{float(n['s_per_window']):.4f}", '-', f"{float(n['Mpx_per_s']):.0f}", '-'])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'tableP10_inference_benchmark.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(out)
    with open(os.path.join(out_dir, 'tableP10_inference_benchmark.md'), 'w') as f:
        f.write('| ' + ' | '.join(COLS) + ' |\n|' + '---|' * len(COLS) + '\n')
        for r in out: f.write('| ' + ' | '.join(r) + ' |\n')
    print('wrote tableP10_inference_benchmark.csv/.md with', len(out), 'rows')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
