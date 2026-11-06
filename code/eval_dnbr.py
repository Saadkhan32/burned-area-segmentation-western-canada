#!/usr/bin/env python3
"""dNBR baselines on every window with a pre-fire stack (build_prefire.py): dNBR = NBR_pre - NBR_post on the pre-fire and
post-fire scenes of the window; rules 'dNBR>0.1' (the unburned / low-severity boundary of Key & Benson 2006, dNBR x 1000 = 100)
and 'Otsu(dNBR)' (per-window Otsu threshold on dNBR, label-free). Scored exactly like eval.py (valid = both scenes unmasked and
label != 255; burned = 1) and written in the same per-window format, so summarize.py and the figures pick them up as baselines.
Usage: python eval_dnbr.py --data ~/scratch/burn/data_postfire --out ~/scratch/burn/runs/results [--save-maps]"""
import os, sys, csv, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_manifest, read_window, confusion, prf, otsu_threshold, write_gtiff, IGNORE
import numpy as np, rasterio

THRESHOLD = 0.1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--save-maps', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = {r['window_id']: r for r in read_manifest(os.path.join(a.data, 'manifest.csv')) if r.get('status', 'OK') == 'OK'}
    pre = {r['window_id']: r for r in csv.DictReader(open(os.path.join(a.data, 'manifest_prefire.csv'), encoding='utf-8')) if r['pre_status'] in ('OK', 'EXISTS')}
    out = []; missing = [w for w in rows if w not in pre]
    for wid, r in rows.items():
        if wid not in pre: continue
        img, lab, profile = read_window(a.data, wid)
        with rasterio.open(os.path.join(a.data, wid, wid + '_S2pre.tif')) as src: nbr_pre = src.read(8).astype(np.float32)   # band 8 = NBR (1-based)
        nbr_post = img[7]; dnbr = nbr_pre - nbr_post
        valid = ~np.isnan(nbr_pre) & ~np.isnan(nbr_post) & (lab != IGNORE)
        thr = otsu_threshold(dnbr[valid]) if valid.sum() > 1000 else float('nan')
        preds = {f'dNBR>{THRESHOLD}': (dnbr > THRESHOLD).astype(np.uint8), 'Otsu(dNBR)': (dnbr > thr).astype(np.uint8) if not math.isnan(thr) else np.zeros_like(lab)}
        for name, p in preds.items():
            tp, fp, fn, tn = confusion(p, lab, valid); P, R, F1, I = prf(tp, fp, fn)
            out.append(dict(run='dnbr_rules', model=name, fold='-', role='TEST', window_id=wid, province=r['province'], valid_px=int(valid.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                            P=P, R=R, F1=F1, IoU=I, ref_ha=round((tp + fn) / 100, 1), pred_ha=round((tp + fp) / 100, 1), otsu_threshold=thr if name.startswith('Otsu') else '', best_epoch=''))
            if a.save_maps:
                m = p.copy(); m[~valid] = IGNORE
                write_gtiff(os.path.join(a.out, 'maps', 'dnbr_rules', f'{wid}_{name.replace(">", "gt")}.tif'), m, profile, 'uint8', nodata=IGNORE)
        print(wid, {k: round(v, 3) for k, v in zip(('P', 'R', 'F1'), prf(*confusion(preds[f'dNBR>{THRESHOLD}'], lab, valid)[:3])[:3])}, 'pre', pre[wid]['pre_acquisition'][:10], 'valid %.2f' % valid.mean(), flush=True)
    if not out:
        raise SystemExit('no window has a pre-fire stack yet (manifest_prefire.csv empty?) - run build_prefire.py --merge-parts after the builds finish')
    path = os.path.join(a.out, 'dnbr_rules_test_windows.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print('wrote', path, '|', len(out) // 2, 'windows scored;', len(missing), 'without pre-fire stack:', ', '.join(missing[:10]))

if __name__ == '__main__':
    main()
