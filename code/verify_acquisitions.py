#!/usr/bin/env python3
"""Verify which Sentinel-2 acquisition each ORIGINAL composite came from (needs catalogue access:
run on a DRAC login node or any computer with internet; not runnable from the preparation environment).

The archived composites of the submitted study were exported by the GEE script as DN * 1e-4 (no offset correction). For each of the 32
windows a 64 x 64 block of the composite (all 9 bands, taken at the window centre; inputs/composite_blocks_64px.npz/.json)
is compared with the same block read from every candidate acquisition within --days of the reconstructed date
(inputs/acquisition_table_reconstructed.csv): Blue band DN of the candidate vs composite * 1e4. The candidate with the
highest fraction of exactly equal pixels (and Pearson r) is the acquisition actually used. Writes
inputs/acquisition_table_verified.csv with VERIFIED / UNRESOLVED per window.
Status: VERIFIED (> 90 % of pixels equal within 1 DN), VERIFIED_RESAMPLED (Pearson r > 0.99: same acquisition, but the export was
resampled when GEE reprojected a neighbouring UTM zone), BEST_MATCH_ONLY, UNRESOLVED. --only "A,B" re-runs selected windows
(e.g. with --days 150) and updates their rows in the verified table.
Usage: python verify_acquisitions.py [--catalogue es|pc] [--days 45] [--only "owl creek,Anahim"]
"""
import os, csv, json, argparse, datetime
import numpy as np
from common import BANDS  # noqa: F401 - sets the GDAL environment and loads h5py before rasterio (see common.py)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--catalogue', default='es'); ap.add_argument('--days', type=int, default=45)
    ap.add_argument('--inputs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs'))
    ap.add_argument('--only', default=None, help='comma-separated event names to (re)run; other rows of an existing verified table are kept')
    ap.add_argument('--min-valid', type=int, default=100)
    a = ap.parse_args()
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject, Resampling
    from build_dataset import open_catalogue, tile_of, item_baseline
    client, assets = open_catalogue(a.catalogue)
    blocks = np.load(os.path.join(a.inputs, 'composite_blocks_64px.npz')); meta = json.load(open(os.path.join(a.inputs, 'composite_blocks_64px.json')))
    table = list(csv.DictReader(open(os.path.join(a.inputs, 'acquisition_table_reconstructed.csv'))))
    out_path = os.path.join(a.inputs, 'acquisition_table_verified.csv')
    previous = {r['event']: r for r in csv.DictReader(open(out_path))} if os.path.exists(out_path) else {}
    only = [x.strip() for x in a.only.split(',')] if a.only else None
    out = []
    for r in table:
        ev = r['event']
        if only is not None and ev not in only:
            if ev in previous: out.append(previous[ev])
            continue
        m = meta[ev]; comp = blocks[ev.replace(' ', '_')]      # (64, 64, 9)
        comp_dn = np.round(comp[..., 0] * 1e4)                                       # Blue band as stored (DN with offset)
        ok = np.isfinite(comp_dn) & (comp_dn > 0)
        epsg = m['epsg']; x0, y1 = m['block_x0'], m['block_y0']; res = m['scale']
        bounds = (x0, y1 - 64 * res, x0 + 64 * res, y1)
        from pyproj import Transformer
        tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
        lons, lats = tr.transform([bounds[0], bounds[2], bounds[0], bounds[2]], [bounds[1], bounds[1], bounds[3], bounds[3]])
        bbox = (min(lons), min(lats), max(lons), max(lats))
        d = datetime.date.fromisoformat(r['acquisition_date'])
        search = client.search(collections=['sentinel-2-l2a'], bbox=bbox, datetime=f'{d - datetime.timedelta(days=a.days)}/{d + datetime.timedelta(days=a.days)}', max_items=400)
        best = None; tested = 0
        for it in search.items():
            try:
                href = it.assets[assets['Blue']].href
                with rasterio.open(href) as src:
                    arr = np.zeros((64, 64), np.float32)
                    if src.crs.to_epsg() == epsg:
                        w = from_bounds(*bounds, transform=src.transform)
                        arr = src.read(1, window=w, out_shape=(64, 64), boundless=True, fill_value=0).astype(np.float32)
                    else:
                        from rasterio.transform import from_origin
                        reproject(rasterio.band(src, 1), arr, dst_transform=from_origin(x0, y1, res, res), dst_crs=f'EPSG:{epsg}', resampling=Resampling.nearest, dst_nodata=0)
            except Exception as e:
                print(ev, it.id, 'read failed', e); continue
            tested += 1
            if bool(it.properties.get('earthsearch:boa_offset_applied', False)) and item_baseline(it) >= 4.0:
                arr = arr + 1000   # catalogue removed the offset; the composite still has it
            v = ok & (arr > 0)
            if v.sum() < a.min_valid or arr[v].std() == 0 or comp_dn[v].std() == 0: continue
            exact = float((np.abs(arr[v] - comp_dn[v]) <= 1).mean()); rr = float(np.corrcoef(arr[v], comp_dn[v])[0, 1])
            cand = dict(id=it.id, date=it.datetime.strftime('%Y-%m-%d'), tile=tile_of(it), exact=exact, r=rr, n_valid=int(v.sum()))
            score = (1 if exact > 0.9 else 0, exact if exact > 0.9 else rr)   # exact matches first, then the best-correlated candidate
            if best is None or score > best['_score']:
                cand['_score'] = score; best = cand
        if best is None: status = 'UNRESOLVED'
        elif best['exact'] > 0.9: status = 'VERIFIED'
        elif best['r'] > 0.99: status = 'VERIFIED_RESAMPLED'
        else: status = 'BEST_MATCH_ONLY'
        row = dict(r); row.update(verified_status=status, candidates_tested=tested, search_days=a.days, composite_valid_px=int(ok.sum()),
                                  best_product=best['id'] if best else '', best_date=best['date'] if best else '', best_tile=best['tile'] if best else '',
                                  exact_fraction=round(best['exact'], 4) if best else '', pearson_r=round(best['r'], 4) if best else '',
                                  agrees_with_reconstruction=(best['date'] == r['acquisition_date']) if best else '')
        out.append(row); print(ev, status, {k: v for k, v in (best or {}).items() if k != '_score'}, flush=True)
    keys = []
    for row in out:
        for k in row:
            if k not in keys: keys.append(k)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(out)
    print('wrote acquisition_table_verified.csv:', {s: sum(1 for x in out if x.get('verified_status') == s) for s in ('VERIFIED', 'VERIFIED_RESAMPLED', 'BEST_MATCH_ONLY', 'UNRESOLVED')})


if __name__ == '__main__':
    main()
