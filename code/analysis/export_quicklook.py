#!/usr/bin/env python3
"""Export compact quick-look rasters of selected windows for the qualitative figure (runs on Narval, next to the data).
For every selected window: <wid>_rgb.png (SWIR2-NIR-Red, 2-98 % stretch, masked pixels light grey), <wid>_label.png (0 unburned,
1 burned, 255 ignored) and <wid>_pred_<model>.png (0/1/255) for every requested model, plus quicklook_index.json with the
acquisition, burned area and per-model scores from the results. About 5 MB per window instead of 150 MB for the float stack.
Usage (Narval, venv):
  python analysis/export_quicklook.py --data ~/scratch/burn/data_postfire --results ~/scratch/burn/runs/results \
      --windows auto --models segformer,upernet_convnext,prithvi_lora,unet,deeplabv3,mask2former --out ~/burn/drac/quicklook
--windows auto = per province the best, the median and the worst window of the first model listed, plus the five weakest
windows of that model overall and two fire-free windows (the one with the largest and one with no mapped burn)."""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: F401  (h5py before rasterio, GDAL environment)
import numpy as np, pandas as pd, rasterio
from PIL import Image

def rgb_quicklook(img):
    rgb = np.zeros(img.shape[1:] + (3,), np.float32)
    for k, b in enumerate((5, 3, 2)):    # SWIR2, NIR, Red
        v = img[b]; lo, hi = np.nanpercentile(v, [2, 98]); rgb[..., k] = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    rgb[np.isnan(img).any(0)] = 0.85
    return (rgb * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--results', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--windows', default='auto'); ap.add_argument('--models', default='segformer,upernet_convnext,prithvi_lora,unet,deeplabv3,mask2former')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    models = a.models.split(',')
    res = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.results, '*_test_windows.csv')))], ignore_index=True).drop_duplicates(['model', 'window_id'], keep='last')
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv')).set_index('window_id')
    if a.windows == 'auto':
        m0 = res[(res.model == models[0]) & (res.ref_ha > 0)].sort_values('F1'); sel = []
        for p, g in m0.groupby('province'):
            g = g.sort_values('F1'); sel += [g.window_id.iloc[-1], g.window_id.iloc[len(g) // 2], g.window_id.iloc[0]]
        sel += list(m0.window_id.iloc[:5])
        neg = res[(res.model == models[0]) & (res.ref_ha <= 0)].sort_values('pred_ha'); sel += [neg.window_id.iloc[-1], neg.window_id.iloc[0]]
        wins = list(dict.fromkeys(sel))
    else:
        wins = a.windows.split(',')
    index = {}
    for wid in wins:
        with rasterio.open(os.path.join(a.data, wid, wid + '_S2.tif')) as s: img = s.read().astype(np.float32)
        with rasterio.open(os.path.join(a.data, wid, wid + '_label.tif')) as s: lab = s.read(1)
        Image.fromarray(rgb_quicklook(img)).save(os.path.join(a.out, wid + '_rgb.png'), optimize=True)
        Image.fromarray(lab.astype(np.uint8)).save(os.path.join(a.out, wid + '_label.png'), optimize=True)
        nan = np.isnan(img).any(0); Image.fromarray((nan * 255).astype(np.uint8)).save(os.path.join(a.out, wid + '_nodata.png'), optimize=True)
        entry = dict(province=str(man.loc[wid, 'province']), acquisition=str(man.loc[wid, 'acquisition']), burned_ha=float(man.loc[wid, 'burned_ha']), fire_year=int(man.loc[wid, 'fire_year']),
                     source=str(man.loc[wid, 'source']), agency_fire_name=str(man.loc[wid].get('agency_fire_name', '')), scene_year_offset=int(man.loc[wid].get('scene_year_offset', 0)), models={})
        for m in models + ['NBR<0', 'Otsu(NBR)']:
            r = res[(res.model == m) & (res.window_id == wid)]
            if len(r): entry['models'][m] = dict(P=float(r.P.iloc[0]), R=float(r.R.iloc[0]), F1=float(r.F1.iloc[0]), pred_ha=float(r.pred_ha.iloc[0]), ref_ha=float(r.ref_ha.iloc[0]))
        for m in models:
            base, tag = (m[:-3], '_C2') if m.endswith('_C2') else (m, '')
            cands = glob.glob(os.path.join(a.results, 'maps', f'{base}_F-*{tag}', wid + '_pred.tif'))
            cands = [c for c in cands if (tag or not os.path.basename(os.path.dirname(c)).endswith('_C2'))]
            if not cands:
                print('no map for', m, wid); continue
            with rasterio.open(cands[0]) as s: pred = s.read(1)
            Image.fromarray(pred.astype(np.uint8)).save(os.path.join(a.out, f'{wid}_pred_{m}.png'), optimize=True)
        for name, arr in (('NBR<0', (img[7] < 0)),):
            Image.fromarray(np.where(nan, 255, arr.astype(np.uint8)).astype(np.uint8)).save(os.path.join(a.out, f'{wid}_pred_NBR.png'), optimize=True)
        index[wid] = entry; print('exported', wid, {m: round(v['F1'], 3) for m, v in entry['models'].items()})
    json.dump(index, open(os.path.join(a.out, 'quicklook_index.json'), 'w'), indent=1)
    print('windows:', ', '.join(wins)); print('->', a.out)

if __name__ == '__main__':
    main()
