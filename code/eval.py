#!/usr/bin/env python3
"""Score a trained run on the held-out province windows of its fold (whole windows, every pixel once).
Also scores the two label-free index rules on the same windows (NBR < 0 and per-window Otsu on NBR; with harmonised
reflectances these are the standard NBR values). Writes results/<run>_windows.csv and, with --save-maps, the predicted
class raster per window (GeoTIFF, 0/1, 255 where the image is masked).
Usage: python eval.py --data data_postfire --run runs/unet_F-AB --out results [--save-maps]
"""
import os, csv, json, argparse, math
import numpy as np, torch
from common import read_manifest, fold_roles, read_window, confusion, prf, sliding_logits, otsu_threshold, write_gtiff, IGNORE
from models import build_model, N_CLASSES
from train import DEFAULT_VALIDATION, normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--manifest', default=None); ap.add_argument('--run', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--save-maps', action='store_true'); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--validation-json', default=None); ap.add_argument('--which', default='TEST', choices=['TEST', 'VALIDATION'])
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, 'config.json'))); ns = json.load(open(os.path.join(a.run, 'norm_stats.json')))
    mean = np.array(ns['mean'], np.float32); std = np.array(ns['std'], np.float32)
    rows = [r for r in read_manifest(a.manifest or os.path.join(a.data, 'manifest.csv')) if r.get('status', 'OK') == 'OK']
    val_map = json.load(open(a.validation_json)) if a.validation_json else DEFAULT_VALIDATION
    roles = fold_roles(rows, cfg['fold'], cfg['validation_windows'] if (not a.validation_json and cfg.get('validation_windows')) else val_map)   # the run's own validation list
    ids = [w for w, r in roles.items() if r == a.which]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model'], 9, pretrained=False, tiny=cfg.get('tiny', False)).to(device)
    ck = torch.load(os.path.join(a.run, 'best.pt'), map_location=device, weights_only=False); model.load_state_dict(ck['model']); model.eval()
    def predict(tiles):
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            return model(torch.from_numpy(normalise(tiles, mean, std)).to(device)).float().cpu().numpy()
    os.makedirs(a.out, exist_ok=True); out_rows = []
    for wid in ids:
        img, lab, profile = read_window(a.data, wid)
        valid = ~np.isnan(img).any(0) & (lab != IGNORE)
        lg = sliding_logits(predict, img, cfg['crop'], cfg['stride'], N_CLASSES, a.batch)
        pred = lg.argmax(0).astype(np.uint8)
        nbr = img[7]
        label = cfg['model'] + (f"_{cfg['tag']}" if cfg.get('tag') else '')   # e.g. pspnet_C2 for the loss-contrast runs
        preds = {label: pred, 'NBR<0': (nbr < 0).astype(np.uint8)}
        thr = otsu_threshold(nbr[valid]); preds['Otsu(NBR)'] = (nbr < thr).astype(np.uint8) if not math.isnan(thr) else np.zeros_like(pred)
        for name, p in preds.items():
            tp, fp, fn, tn = confusion(p, lab, valid); P, R, F1, I = prf(tp, fp, fn)
            out_rows.append(dict(run=os.path.basename(a.run.rstrip('/')), model=name, fold=cfg['fold'], role=a.which, window_id=wid, province=next(r['province'] for r in rows if r['window_id'] == wid),
                                 valid_px=int(valid.sum()), tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F1=F1, IoU=I, ref_ha=round((tp + fn) / 100, 1), pred_ha=round((tp + fp) / 100, 1),
                                 otsu_threshold=thr if name == 'Otsu(NBR)' else '', best_epoch=ck.get('epoch', '')))
        if a.save_maps:
            m = pred.copy(); m[~(~np.isnan(img).any(0))] = IGNORE
            write_gtiff(os.path.join(a.out, 'maps', os.path.basename(a.run.rstrip('/')), wid + '_pred.tif'), m, profile, 'uint8', nodata=IGNORE)
        print(wid, {k: round(v, 3) for k, v in zip(('P', 'R', 'F1', 'IoU'), prf(*confusion(pred, lab, valid)[:3]))}, flush=True)
    path = os.path.join(a.out, os.path.basename(a.run.rstrip('/')) + f'_{a.which.lower()}_windows.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print('wrote', path)


if __name__ == '__main__':
    main()
