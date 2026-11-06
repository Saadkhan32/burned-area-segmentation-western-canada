#!/usr/bin/env python3
"""Inference-time benchmark for the benchmark models (Supplementary Table S8 of the paper).
For every run folder given (default: the F-AB run of every model), loads best.pt, runs the same sliding-window inference as
eval.py over a few test windows of that fold (whole 20 x 20 km windows, 256-px tiles) and reports seconds per window,
tiles per second, pixels per second, GPU memory, parameter count, batch size and precision. Also times the NBR < 0 rule.
Usage (GPU job, see slurm/bench.sh):  python analysis/bench_inference.py --data data --runs ~/scratch/burn/runs --out bench.csv
"""
import os, sys, csv, json, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_manifest, read_window, sliding_logits, IGNORE   # noqa: E402 (h5py / rasterio order)
import numpy as np, torch
from models import build_model, N_CLASSES
from train import normalise

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fold', default='F-AB'); ap.add_argument('--n-windows', type=int, default=5); ap.add_argument('--batch', type=int, default=16); ap.add_argument('--strides', default='128,256')
    a = ap.parse_args(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, f'*_{a.fold}')) + glob.glob(os.path.join(a.runs, f'*_{a.fold}_C2')) if os.path.exists(os.path.join(d, 'best.pt')))
    rows = [r for r in read_manifest(os.path.join(a.data, 'manifest.csv')) if r.get('status', 'OK') == 'OK']
    cfg0 = json.load(open(os.path.join(runs[0], 'config.json')))
    wids = [w for w in cfg0['test_windows'] if not w.startswith('NEG')][:a.n_windows]
    windows = {w: read_window(a.data, w) for w in wids}
    px = sum(v[0].shape[1] * v[0].shape[2] for v in windows.values())
    out = []
    # the index rule
    t0 = time.perf_counter()
    for w, (img, lab, prof) in windows.items(): _ = (img[7] < 0)
    t = time.perf_counter() - t0
    out.append(dict(model='NBR<0', run='-', params_M='', batch='', precision='-', stride='-', n_windows=len(wids), s_per_window=round(t / len(wids), 4), tiles_per_s='', Mpx_per_s=round(px / t / 1e6, 1), gpu_mem_GB='', device='cpu (numpy)'))
    for run in runs:
        cfg = json.load(open(os.path.join(run, 'config.json'))); ns = json.load(open(os.path.join(run, 'norm_stats.json')))
        mean = np.array(ns['mean'], np.float32); std = np.array(ns['std'], np.float32)
        model = build_model(cfg['model'], 9, pretrained=False, tiny=cfg.get('tiny', False)).to(dev)
        ck = torch.load(os.path.join(run, 'best.pt'), map_location=dev, weights_only=False); model.load_state_dict(ck['model']); model.eval()
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        def predict(tiles):
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):
                return model(torch.from_numpy(normalise(tiles, mean, std)).to(dev)).float().cpu().numpy()
        for stride in [int(s) for s in a.strides.split(',')]:
            # warm-up on one window, then timed pass over all windows (GPU work synchronised by the .cpu() copies)
            img0 = next(iter(windows.values()))[0]; sliding_logits(predict, img0, cfg['crop'], stride, N_CLASSES, a.batch)
            if dev.type == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter(); n_tiles = 0
            for w, (img, lab, prof) in windows.items():
                H, W = img.shape[1:]; ny = len(range(0, max(H - cfg['crop'], 0) + 1, stride)) + 1; nx = len(range(0, max(W - cfg['crop'], 0) + 1, stride)) + 1; n_tiles += ny * nx
                sliding_logits(predict, img, cfg['crop'], stride, N_CLASSES, a.batch)
            if dev.type == 'cuda': torch.cuda.synchronize()
            t = time.perf_counter() - t0
            out.append(dict(model=cfg['model'] + ('_' + cfg['tag'] if cfg.get('tag') else ''), run=os.path.basename(run), params_M=round(n_par, 1), batch=a.batch, precision='fp16 autocast' if dev.type == 'cuda' else 'fp32',
                            stride=stride, n_windows=len(wids), s_per_window=round(t / len(wids), 2), tiles_per_s=round(n_tiles / t, 1), Mpx_per_s=round(px / t / 1e6, 2),
                            gpu_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == 'cuda' else '', device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'))
            print(out[-1], flush=True)
        del model; torch.cuda.empty_cache() if dev.type == 'cuda' else None
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print('wrote', a.out)

if __name__ == '__main__':
    main()
