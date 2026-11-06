#!/usr/bin/env python3
"""Pre-fire scene for every window of the dataset -> the dNBR baseline (the more-information baseline of the paper).
For each window in <out>/manifest.csv (status OK) the clearest Sentinel-2 L2A scene of the snow-free season
(--season-start/--season-end) of the year BEFORE the fire year is chosen, preferring the acquisition closest in day-of-year to the
window's post-fire scene (phenology match: the "anniversary" pre-fire image of the dNBR literature) among scenes with clear fraction
>= --min-clear; if none reaches --min-clear the clearest one is taken. Fire-free windows get the season before their scene year.
The same grid, bands, harmonisation (BOA offset check), SCL masking and index layers as build_dataset.py are used, so NBR_pre and
NBR_post are directly comparable. Writes <out>/<wid>/<wid>_S2pre.tif, <wid>_scl_pre.tif and manifest_prefire[_part<k>of<n>].csv.
Usage:  python build_prefire.py --out ~/scratch/burn/data_postfire --batch 0/4      (one process per batch; then --merge-parts)
        python build_prefire.py --out ~/scratch/burn/data_postfire --merge-parts
"""
import os, sys, csv, glob, argparse, datetime, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BANDS, indices, harmonise, offset_in_dn, write_gtiff, window_grid, read_manifest
from build_dataset import open_catalogue, search_items, item_baseline, read_into_grid, clear_fraction, tile_of, log, MASK_SCL, in_batch

PRE_COLUMNS = ['window_id', 'province', 'fire_year', 'post_acquisition', 'pre_status', 'pre_search_start', 'pre_search_end', 'pre_acquisition', 'pre_product_ids', 'pre_tiles',
               'pre_processing_baseline', 'pre_boa_offset_applied_by_catalogue', 'pre_boa_offset_subtracted', 'pre_blue_dn_p1', 'pre_offset_note', 'pre_clear_fraction', 'pre_coverage',
               'pre_masked_fraction', 'doy_difference_days', 'pre_candidates_checked']


def choose_prefire(client, assets, bbox, grid, start, end, target_doy, min_clear, max_candidates=25):
    """Candidates ordered by |day-of-year - target|; the first with clear >= min_clear (coverage >= 0.99) wins; else the clearest."""
    epsg, transform, width, height = grid
    groups = search_items(client, bbox, start, end)
    def doy(k): return datetime.date.fromisoformat(k[:10]).timetuple().tm_yday
    keys = sorted(groups, key=lambda k: (abs(doy(k) - target_doy), min(it.properties.get('eo:cloud_cover', 100) for it in groups[k])))[:max_candidates]
    best = None; checked = 0
    for k in keys:
        if min(it.properties.get('eo:cloud_cover', 100) for it in groups[k]) > 80:
            continue
        try:
            clear, cov, scl = clear_fraction(groups[k], assets, epsg, transform, width, height); checked += 1
        except Exception as e:
            log('  skip', k, e); continue
        if cov < 0.99:
            continue
        if clear >= min_clear:
            return (k, groups[k], clear, cov, scl), checked
        if best is None or clear > best[2]:
            best = (k, groups[k], clear, cov, scl)
    return best, checked


def build_prefire(client, assets, r, args):
    wid = r['window_id']; row = dict(window_id=wid, province=r['province'], fire_year=r['fire_year'], post_acquisition=r['acquisition'])
    post = r['acquisition'][:10]; post_doy = datetime.date.fromisoformat(post).timetuple().tm_yday
    is_neg = str(r.get('is_negative', '')).lower() in ('true', '1')
    year = int(post[:4]) - 1 if is_neg else int(r['fire_year']) - 1   # the season before the fire (fire-free windows: before their scene)
    start, end = f'{year}-{args.season_start}', f'{year}-{args.season_end}'
    if not is_neg and r.get('nbac_sdate') and r['nbac_sdate'][:4] == str(int(r['fire_year'])) and r['nbac_sdate'] < f'{int(r["fire_year"])}-{args.season_start}':
        pass   # a fire that started before the season start is still after the previous season: nothing to change
    row.update(pre_search_start=start, pre_search_end=end)
    out_dir = os.path.join(args.out, wid); dst = os.path.join(out_dir, wid + '_S2pre.tif')
    if os.path.exists(dst) and not args.overwrite:
        log(f'{wid}: pre-fire stack exists, skipped'); row['pre_status'] = 'EXISTS'; return row
    lon, lat = float(r['lon']), float(r['lat'])
    epsg, transform, width, height = window_grid(lon, lat, res=10.0)
    assert epsg == int(r['epsg']) and width == int(r['width']) and height == int(r['height']), f'{wid}: grid mismatch with the manifest'
    from rasterio.transform import array_bounds
    from pyproj import Transformer
    b = array_bounds(height, width, transform); tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lons, lats = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]]); bbox = (min(lons), min(lats), max(lons), max(lats))
    grid = (epsg, transform, width, height); failed = set()
    for attempt in range(3):
        best, checked = choose_prefire(client, assets, bbox, grid, start, end, post_doy, args.min_clear)
        if best is None:
            log(f'{wid}: no pre-fire scene in {start}..{end}'); row['pre_status'] = 'NO_SCENE'; return row
        key, items, clear, cov, scl = best
        if key in failed:
            row['pre_status'] = 'READ_FAILED'; return row
        baseline = max(item_baseline(it) for it in items)
        offset_applied = all(bool(it.properties.get('earthsearch:boa_offset_applied', False)) for it in items)
        dns = []
        try:
            for band in BANDS[:6]:
                dn, _ = read_into_grid(items, assets[band], epsg, transform, width, height, 'bilinear' if band in ('SWIR1', 'SWIR2') else 'nearest', np.float32, 0); dns.append(dn)
            break
        except Exception as e:
            log(f'{wid}: band read failed for {key} ({e}); retrying'); failed.add(key)
    else:
        row['pre_status'] = 'READ_FAILED'; return row
    subtract = baseline >= 4.0 and not offset_applied
    dn_p1, present = offset_in_dn(dns[0], ~np.isin(scl, MASK_SCL)); note = ''
    if subtract and present is False:
        subtract = False; note = f'catalogue flag says offset not applied, but blue DN p1 = {dn_p1:.0f} shows no offset: not subtracted'
    elif not subtract and present is True:
        note = f'blue DN p1 = {dn_p1:.0f}: offset may still be in the DN (not subtracted; check this window)'
    if note: log(f'{wid}: {note}')
    stack = indices(np.stack([harmonise(dn, baseline, offset_applied, subtract=subtract) for dn in dns])); del dns
    bad = np.isin(scl, MASK_SCL); stack[:, bad] = np.nan
    profile = dict(crs=f'EPSG:{epsg}', transform=transform, width=width, height=height)
    write_gtiff(dst, stack, profile, 'float32', nodata=None); write_gtiff(os.path.join(out_dir, wid + '_scl_pre.tif'), scl, profile, 'uint8', nodata=0)
    row.update(pre_status='OK', pre_acquisition=key, pre_product_ids=';'.join(it.id for it in items), pre_tiles=';'.join(sorted({tile_of(it) for it in items})), pre_processing_baseline=baseline,
               pre_boa_offset_applied_by_catalogue=offset_applied, pre_boa_offset_subtracted=subtract, pre_blue_dn_p1=round(dn_p1, 1) if dn_p1 == dn_p1 else '', pre_offset_note=note,
               pre_clear_fraction=round(clear, 4), pre_coverage=round(cov, 4), pre_masked_fraction=round(float(bad.mean()), 4),
               doy_difference_days=datetime.date.fromisoformat(key[:10]).timetuple().tm_yday - post_doy, pre_candidates_checked=checked)
    log(f'{wid}: pre-fire {key} clear {clear:.3f} (post {post}, DOY diff {row["doy_difference_days"]:+d} d)')
    return row


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=PRE_COLUMNS, restval=''); w.writeheader(); w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, help='dataset folder (with manifest.csv)'); ap.add_argument('--catalogue', default='es', choices=['es', 'pc'])
    ap.add_argument('--season-start', default='06-01'); ap.add_argument('--season-end', default='10-15'); ap.add_argument('--min-clear', type=float, default=0.8)
    ap.add_argument('--only', default=None); ap.add_argument('--batch', default=None, help='k/n'); ap.add_argument('--merge-parts', action='store_true'); ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.merge_parts:
        parts = sorted(glob.glob(os.path.join(args.out, 'manifest_prefire_part*of*.csv'))); merged = {}
        if not parts:
            raise SystemExit('no manifest_prefire_part*.csv yet - the batch builds have not finished (each writes its part file at the end)')
        for p in parts:
            for r in csv.DictReader(open(p, encoding='utf-8')): merged[r['window_id']] = r
        path = write_rows(os.path.join(args.out, 'manifest_prefire.csv'), list(merged.values()))
        st = {}
        for r in merged.values(): st[r['pre_status']] = st.get(r['pre_status'], 0) + 1
        log(f'merged {len(parts)} parts: {len(merged)} windows ->', path, st); return
    rows = [r for r in read_manifest(os.path.join(args.out, 'manifest.csv')) if r.get('status', 'OK') == 'OK']
    if args.only: rows = [r for r in rows if r['window_id'] in set(args.only.split(','))]
    client, assets = open_catalogue(args.catalogue); out = []; t0 = time.time()
    for r in rows:
        if not in_batch(args): continue
        try:
            out.append(build_prefire(client, assets, r, args))
        except Exception as e:
            log(f'{r["window_id"]}: ERROR {e}'); out.append(dict(window_id=r['window_id'], province=r['province'], fire_year=r['fire_year'], post_acquisition=r['acquisition'], pre_status=f'ERROR {str(e)[:80]}'))
    name = f'manifest_prefire_part{args.batch.replace("/", "of")}.csv' if args.batch else ('manifest_prefire_only.csv' if args.only else 'manifest_prefire.csv')
    path = write_rows(os.path.join(args.out, name), out); log(f'done: {len(out)} windows in {(time.time() - t0) / 60:.0f} min ->', path)


if __name__ == '__main__':
    main()
