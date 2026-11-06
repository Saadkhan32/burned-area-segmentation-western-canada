#!/usr/bin/env python3
"""Synthetic windows for the smoke test: 14 windows (3-4 fire windows + extra-NBAC windows per province, one negative window), 320 x 320 px, with burn blobs (low NBR), a prior-year scar (255),
and masked pixels (NaN). Writes <out>/manifest.csv in the dataset format."""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_gtiff, indices
from rasterio.transform import from_origin
out = sys.argv[1]; rng = np.random.default_rng(0); rows = []
names = {'AB': ['Betts', 'Pony_Creek', 'Rabbit_Lake', 'NEG2023_AB_01', 'NBAC2023_AB_301'], 'BC': ['Greer_Creek', 'Tatuk_Lake', 'Murtle_Lake', 'NBAC2023_BC_101', 'NBAC2023_BC_102'],
         'SK': ['HEINZ', 'COWIE', 'SMITH', 'NBAC2023_SK_201', 'NBAC2023_SK_202']}
for pv, ids in names.items():
    for wid in ids:
        H = W = 320; yy, xx = np.mgrid[0:H, 0:W]
        lab = np.zeros((H, W), np.uint8)
        for _ in range(0 if wid.startswith('NEG') else 3):
            cy, cx, r = rng.integers(40, H - 40), rng.integers(40, W - 40), rng.integers(20, 60)
            lab[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 1
        prior = ((yy - 290) ** 2 + (xx - 30) ** 2 < 20 ** 2) & (not wid.startswith('NEG')); lab[prior] = 255
        refl = np.stack([rng.normal(m, 0.01, (H, W)) for m in (0.03, 0.05, 0.04, 0.30, 0.20, 0.12)]).astype(np.float32)
        burned = lab == 1
        refl[3][burned] = rng.normal(0.12, 0.01, burned.sum()); refl[5][burned] = rng.normal(0.22, 0.01, burned.sum())   # NIR down, SWIR2 up
        refl[3][prior] = 0.16; refl[5][prior] = 0.18
        img = indices(np.clip(refl, 0, 1)); img[:, :15, :] = np.nan   # masked strip
        prof = dict(crs='EPSG:32612', transform=from_origin(500000, 6200000, 10, 10), width=W, height=H)
        write_gtiff(os.path.join(out, wid, wid + '_S2.tif'), img, prof, 'float32'); write_gtiff(os.path.join(out, wid, wid + '_label.tif'), lab, prof, 'uint8')
        source = 'negative window' if wid.startswith('NEG') else ('extra NBAC fire' if wid.startswith('NBAC') else 'original 32 windows')
        rows.append(dict(window_id=wid, province=pv, source=source, status='OK', fire_year=2023, burned_ha=round(float(burned.sum()) / 100, 1), is_negative=wid.startswith('NEG')))
with open(os.path.join(out, 'manifest.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('synthetic windows written to', out)
