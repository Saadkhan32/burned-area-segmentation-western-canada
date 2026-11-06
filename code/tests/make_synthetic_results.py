#!/usr/bin/env python3
"""Synthetic stand-in for a finished run folder, used only to exercise analysis/make_figures_planA.py before the real
results exist. Writes results/<run>_test_windows.csv (same columns as eval.py), summary/ (via summarize.py), manifest.csv,
run_ledger.csv and <run>/log.csv + ledger.json + config.json. Numbers are random and mean nothing.
Usage: python tests/make_synthetic_results.py --manifest manifest.csv --out /tmp/fake_runs"""
import os, csv, json, argparse, subprocess, sys
import numpy as np

MODELS = {'unet': .885, 'deeplabv3': .885, 'pspnet': .880, 'samlora': .87, 'mask2former': .878, 'unet_C2': .89, 'deeplabv3_C2': .88, 'pspnet_C2': .888,
          'deeplabv3plus': .883, 'segformer': .902, 'upernet_convnext': .90, 'prithvi': .858, 'prithvi_lora': .904, 'terramind': .845}
BASE = {'NBR<0': .63, 'Otsu(NBR)': .65}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    rows = [r for r in csv.DictReader(open(a.manifest, encoding='utf-8')) if r['status'] == 'OK']
    os.makedirs(os.path.join(a.out, 'results'), exist_ok=True)
    import shutil; shutil.copy(a.manifest, os.path.join(a.out, 'manifest.csv'))
    hard = {r['window_id']: rng.uniform(0.3, 0.7) for r in rng.choice(rows, 6, replace=False)}   # a few hard windows shared by all models
    ledger = []
    for model, mu in MODELS.items():
        for fold in ['F-AB', 'F-BC', 'F-SK']:
            prov = fold.split('-')[1]; run = f"{model.replace('_C2', '')}_{fold}" + ('_C2' if model.endswith('_C2') else '')
            out = []
            for r in rows:
                if r['province'] != prov: continue
                neg = r['is_negative'] in ('True', 'true', '1'); valid = int(r['width']) * int(r['height'])
                ref = 0 if neg else int(float(r['burned_ha']) * 100)
                for name, m in [(model, mu)] + list(BASE.items()):
                    if neg:
                        fp = int(rng.exponential(2000 if name in BASE else 30 * (1 + 5 * (name in ('terramind', 'segformer'))))); tp = fn = 0; tn = valid - fp
                        P = R = F1 = I = 0.0
                    else:
                        f1 = hard.get(r['window_id'], 1.0) * np.clip(rng.normal(m, 0.05), 0.02, 0.99)
                        rec = np.clip(f1 + rng.normal(0, 0.03), 0.02, 0.995); prec = f1 * rec / max(2 * rec - f1, 1e-6); prec = float(np.clip(prec, 0.02, 0.995))
                        tp = int(ref * rec); fn = ref - tp; fp = int(tp / prec - tp); tn = valid - tp - fn - fp
                        P, R = tp / (tp + fp), tp / (tp + fn); F1 = 2 * P * R / (P + R); I = tp / (tp + fp + fn)
                    out.append(dict(run=run, model=name, fold=fold, role='TEST', window_id=r['window_id'], province=prov, valid_px=valid, tp=tp, fp=fp, fn=fn, tn=tn,
                                    P=round(P, 6), R=round(R, 6), F1=round(F1, 6), IoU=round(I, 6), ref_ha=round((tp + fn) / 100, 1), pred_ha=round((tp + fp) / 100, 1),
                                    otsu_threshold=-0.1 if name == 'Otsu(NBR)' else '', best_epoch=int(rng.integers(5, 40))))
            with open(os.path.join(a.out, 'results', run + '_test_windows.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
            rd = os.path.join(a.out, run); os.makedirs(rd, exist_ok=True); ep = int(rng.integers(12, 50))
            with open(os.path.join(rd, 'log.csv'), 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['epoch', 'loss', 'val_F1_mean', 'val_F1_pooled', 'lr', 'seconds', 'best'])
                for e in range(1, ep + 1): w.writerow([e, round(0.4 * np.exp(-e / 8) + 0.08, 4), round(min(0.94, 0.8 + 0.01 * e), 4), 0.9, 3e-4, 15, 0.9])
            json.dump(dict(model=model.replace('_C2', ''), fold=fold, tag='C2' if model.endswith('_C2') else '', crop=256, stride=128, batch=16, epochs=120, patience=10), open(os.path.join(rd, 'config.json'), 'w'))
            json.dump(dict(run=run, best_val_F1_mean_window=0.9, epochs_run=ep, wall_seconds=int(ep * 20), gpu='NVIDIA A100-SXM4-40GB'), open(os.path.join(rd, 'ledger.json'), 'w'))
            ledger.append([run, 0.9, ep, 'sha', '2026-10-14 03:00:00', int(ep * 20), 'NVIDIA A100-SXM4-40GB'])
    with open(os.path.join(a.out, 'run_ledger.csv'), 'w', newline='') as f: csv.writer(f).writerows(ledger)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(here, 'summarize.py'), '--results', os.path.join(a.out, 'results'), '--out', os.path.join(a.out, 'summary')], check=True, stdout=subprocess.DEVNULL)
    print('synthetic run folder ->', a.out)

if __name__ == '__main__':
    main()
