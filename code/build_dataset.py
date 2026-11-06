#!/usr/bin/env python3
"""Build the post-fire, NBAC-referenced window dataset .
STATUS: written against pystac-client 0.9 / rasterio 1.5 / shapely 2; the network parts could not be run from the
preparation environment (no access to the catalogues). Run the --dry-run first, then one window, then all.

For every fire window:
  1. the window grid is the bounds of a 10 km circle around the agency fire location (as in the original GEE script),
     10 m, in the UTM zone of the centre (option --zone-from-tile keeps the original composite's zone);
  2. NBAC polygons of the fire year inside the window are the burned reference (burned = 1); polygons of the
     --prior-years before it are set to 255 (ignored in training and scoring) because a single post-fire image cannot
     separate a one-year-old scar from a new one; everything else is unburned (0);
  3. the scene is the clearest Sentinel-2 L2A acquisition in the window AFTER the fire's NBAC end date (+ --settle-days)
     and before --season-end of that year (first snow-free season); if none qualifies, the next year's --season-start..--season-end;
     clearness = fraction of the window with SCL in {3,8,9,10,11} or nodata, computed from the SCL asset; same-datetime
     tiles are mosaicked so that a window crossing a tile boundary is complete;
  4. bands B2,B3,B4,B8 (10 m) and B11,B12 (20 m, bilinear to 10 m) are harmonised to surface reflectance (BOA offset of
     processing baseline >= 04.00 subtracted unless the catalogue already applied it), scaled to [0,1], NDVI/NBR/NDMI appended,
     masked pixels set to NaN; written as <window_id>_S2.tif (9 x float32), <window_id>_label.tif (uint8), <window_id>_scl.tif.
  5. manifest.csv gets one row per window (ids, dates, product ids, cloud fraction, burned ha, prior-year ha).

Catalogue: Earth Search v1 (AWS, no account) by default; --catalogue pc uses the Planetary Computer with SAS signing
(pip install planetary-computer). Inputs: inputs/event_table_recovered.csv (32 fires) and the NBAC shapefiles
(fetch_nbac.sh). --per-province N tops every province up to N fire windows with further NBAC fires (>= --min-ha,
2022-2024, size-stratified, >= ~25 km from every other window; seed 42); --negatives-per-province M adds M fire-free windows
per province (R1.8). With --batch k/n each process writes manifest_part<k>of<n>.csv; --merge-parts joins them into manifest.csv.
Usage:
  python build_dataset.py --nbac nbac/ --out data_postfire/ --per-province 40 --negatives-per-province 5 --dry-run
  python build_dataset.py --nbac nbac/ --out data_postfire/ --only Betts
  python build_dataset.py --nbac nbac/ --out data_postfire/ --per-province 40 --negatives-per-province 5 --batch 0/4
  python build_dataset.py --nbac nbac/ --out data_postfire/ --per-province 40 --negatives-per-province 5 --only NBAC2023_AB_685,Levellers_Creek   (rebuild)
  python build_dataset.py --out data_postfire/ --merge-parts        (joins the parts and every manifest_only*.csv into manifest.csv)
A --only rebuild also replaces its windows' rows in the existing manifest.csv directly.
"""
import os, sys, csv, json, math, argparse, datetime, time
import numpy as np
from common import BANDS, IGNORE, indices, harmonise, offset_in_dn, write_gtiff, window_grid

EARTH_SEARCH = 'https://earth-search.aws.element84.com/v1'
PLANETARY = 'https://planetarycomputer.microsoft.com/api/stac/v1'
ES_ASSETS = {'Blue': 'blue', 'Green': 'green', 'Red': 'red', 'NIR': 'nir', 'SWIR1': 'swir16', 'SWIR2': 'swir22', 'SCL': 'scl'}
PC_ASSETS = {'Blue': 'B02', 'Green': 'B03', 'Red': 'B04', 'NIR': 'B08', 'SWIR1': 'B11', 'SWIR2': 'B12', 'SCL': 'SCL'}
MASK_SCL = (0, 1, 3, 8, 9, 10, 11)   # nodata, saturated/defective, cloud shadow, cloud medium/high, cirrus, snow (0 and 1 added vs the original script)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------- NBAC
def load_nbac(nbac_dir, years, required=()):
    """Read the NBAC shapefiles (one per year, unzipped by fetch_nbac.sh) and reproject them to EPSG:4326 with GDAL
    (rasterio.warp.transform on the whole coordinate array - no per-vertex Python, no pyproj). The result is cached as
    NBAC_<year>_4326.pkl next to the shapefile. Returns [(year, attributes dict, shapely geometry)]."""
    import glob, pickle
    import shapely
    recs = []; fields_logged = False
    for y in years:
        cands = sorted(glob.glob(os.path.join(nbac_dir, '**', f'*{y}*.shp'), recursive=True))
        if not cands:
            if y in required:
                raise SystemExit(f'NBAC shapefile for {y} not found under {nbac_dir} (run fetch_nbac.sh)')
            log(f'NBAC {y}: not present (prior-year scars of {y} will not be masked)'); continue
        shp = cands[0]; cache = os.path.join(os.path.dirname(shp), f'NBAC_{y}_4326.pkl')
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(shp):
            with open(cache, 'rb') as f:
                fields, attrs, wkb = pickle.load(f)
            geoms = shapely.from_wkb(wkb)
        else:
            fields, attrs, geoms = _read_shapefile_4326(shp)
            with open(cache, 'wb') as f:
                pickle.dump((fields, attrs, shapely.to_wkb(geoms)), f)
        if not fields_logged and fields:
            log('NBAC fields:', ', '.join(fields)); fields_logged = True
        n = 0
        for a, g in zip(attrs, geoms):
            for k in ('NFIREID', 'GID', 'YEAR'):   # integer ids read as floats by some drivers
                v = a.get(k)
                if isinstance(v, float) and v == int(v):
                    a[k] = int(v)
            if g is None or g.is_empty:
                continue
            recs.append((y, a, g)); n += 1
        if n:
            b = shapely.total_bounds(geoms); prov = {}
            for a in attrs:
                k = province_of(a); prov[k] = prov.get(k, 0) + 1
            inside = int(np.sum(shapely.intersects(geoms, shapely.box(-141, 48, -88, 62))))
            log(f'NBAC {y}: {n} polygons from {os.path.basename(shp)}; lon {b[0]:.1f}..{b[2]:.1f} lat {b[1]:.1f}..{b[3]:.1f}; '
                f'{inside} inside western Canada; by ADMIN_NAME: ' + ', '.join(f'{k}={v}' for k, v in sorted(prov.items(), key=lambda kv: -kv[1])[:8]))
            if inside == 0:
                raise SystemExit(f'NBAC {y}: no polygon falls inside western Canada after reprojection - the .prj / GDAL PROJ setup is wrong')
    return recs


def _read_shapefile_4326(shp):
    """Shapefile -> (field names, list of attribute dicts, array of shapely geometries in EPSG:4326)."""
    import shapely
    from rasterio.crs import CRS as RCRS
    from rasterio.warp import transform as rtransform
    prj = os.path.splitext(shp)[0] + '.prj'
    src_crs = RCRS.from_wkt(open(prj).read()) if os.path.exists(prj) else RCRS.from_epsg(3978)
    try:
        from pyogrio.raw import read as raw_read
        meta, _, geom_wkb, field_data = raw_read(shp, datetime_as_string=True)   # NBAC has empty dates ('year 0') that break datetime parsing
        fields = list(meta['fields']); attrs = [dict(zip(fields, vals)) for vals in zip(*field_data)] if fields else [{} for _ in geom_wkb]
        geoms = shapely.from_wkb(geom_wkb)
    except Exception as e:
        log(f'pyogrio could not read {os.path.basename(shp)} ({e}); falling back to pyshp')
        import shapefile  # pyshp fallback
        from shapely.geometry import shape
        with shapefile.Reader(shp) as src:
            fields = [f[0] for f in src.fields[1:]]; attrs = []; gl = []
            for sr in src.iterShapeRecords():
                attrs.append(dict(zip(fields, sr.record))); gl.append(shape(sr.shape.__geo_interface__))
        geoms = np.array(gl, dtype=object)
    geoms = np.asarray(geoms, dtype=object)
    if src_crs.to_epsg() != 4326:
        coords = shapely.get_coordinates(geoms)
        xs, ys = rtransform(src_crs, RCRS.from_epsg(4326), coords[:, 0].tolist(), coords[:, 1].tolist())
        geoms = shapely.set_coordinates(geoms, np.column_stack([xs, ys]))
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms[bad] = shapely.make_valid(geoms[bad])
    return fields, attrs, geoms


def province_of(p):
    """Two-letter province / agency code of an NBAC record (ADMIN_NAME in the 2026 release; AGENCY in older ones)."""
    v = str(p.get('ADMIN_NAME') or p.get('AGENCY') or p.get('ADMIN_AREA') or '').strip().upper()
    names = {'ALBERTA': 'AB', 'BRITISH COLUMBIA': 'BC', 'SASKATCHEWAN': 'SK', 'MANITOBA': 'MB', 'ONTARIO': 'ON', 'QUEBEC': 'QC',
             'YUKON': 'YT', 'NORTHWEST TERRITORIES': 'NT'}
    return names.get(v, v[:2])


def tile_of(it):
    p = it.properties
    if 'grid:code' in p:
        return p['grid:code'].replace('MGRS-', '')
    if 's2:mgrs_tile' in p:
        return p['s2:mgrs_tile']
    parts = it.id.split('_')
    return parts[5][1:] if len(parts) > 5 else parts[1]


_TREES = {}


def nbac_in_window(recs, year, bbox):
    """Records of `year` whose polygon intersects bbox (lon/lat); one STRtree per year is built on first use."""
    import shapely
    key = (id(recs), year)
    if key not in _TREES:
        idx = [i for i, r in enumerate(recs) if r[0] == year]
        _TREES[key] = (idx, shapely.STRtree([recs[i][2] for i in idx]) if idx else None)
    idx, tree = _TREES[key]
    if tree is None:
        return []
    hits = tree.query(shapely.box(*bbox), predicate='intersects')
    return [recs[idx[i]] for i in sorted(hits)]


def nbac_dates(polys):
    """Start/end of burning from the NBAC attributes. 2026 release: AG_SDATE/AG_EDATE (agency) and HS_SDATE/HS_EDATE (hotspots);
    older releases: SDATE/EDATE or AFSDATE/AFEDATE. Start = earliest agency (else hotspot) start; end = latest hotspot end
    (the last detected burning - the agency end date is often the much later 'out' date), else the agency end."""
    def norm(v):
        if v is None or isinstance(v, (bytes, bytearray)) or str(v).strip() in ('', 'NaT', 'nan', 'None'):
            return None
        t = str(v).strip().replace('/', '-')
        if len(t) == 8 and t.isdigit():          # YYYYMMDD
            t = f'{t[:4]}-{t[4:6]}-{t[6:]}'
        if len(t) < 10 or t[4] != '-' or not t[:4].isdigit() or int(t[:4]) < 1990:   # '0000-00-00' and other fillers
            return None
        return t[:10]
    s_ag, s_hs, e_hs, e_ag = [], [], [], []
    for _, p, _ in polys:
        for k in ('AG_SDATE', 'SDATE', 'AFSDATE'):
            v = norm(p.get(k))
            if v: s_ag.append(v); break
        v = norm(p.get('HS_SDATE'))
        if v: s_hs.append(v)
        v = norm(p.get('HS_EDATE'))
        if v: e_hs.append(v)
        for k in ('EDATE', 'AFEDATE', 'AG_EDATE'):
            v = norm(p.get(k))
            if v: e_ag.append(v); break
    start = min(s_ag) if s_ag else (min(s_hs) if s_hs else None)
    end = max(e_hs) if e_hs else (max(e_ag) if e_ag else None)   # hotspot ends preferred over agency 'out' dates whenever any exist
    return start, end


# ------------------------------------------------------------------------------------------------------------ catalogue
def open_catalogue(name):
    from pystac_client import Client
    if name == 'pc':
        import planetary_computer
        return Client.open(PLANETARY, modifier=planetary_computer.sign_inplace), PC_ASSETS
    return Client.open(EARTH_SEARCH), ES_ASSETS


def search_items(client, bbox, start, end, max_cloud=80):
    """All L2A items touching bbox in [start, end] with scene cloud cover <= max_cloud, grouped by acquisition datetime."""
    search = client.search(collections=['sentinel-2-l2a'], bbox=bbox, datetime=f'{start}/{end}',
                           query={'eo:cloud_cover': {'lte': max_cloud}}, max_items=500)
    items = list(search.items())
    groups = {}
    for it in items:
        key = it.datetime.strftime('%Y-%m-%dT%H:%M')
        groups.setdefault(key, []).append(it)
    return dict(sorted(groups.items()))


def item_baseline(it):
    p = it.properties
    for k in ('s2:processing_baseline', 'processing:baseline'):
        if k in p:
            try:
                return float(p[k])
            except ValueError:
                pass
    # fall back to the product id: ..._N0509_...
    import re
    m = re.search(r'_N(\d{4})_', it.id)
    return float(m.group(1)) / 100 if m else 0.0


def read_into_grid(items, asset_key, epsg, transform, width, height, resampling, dtype=np.float32, nodata_fill=0):
    """Mosaic one asset of same-datetime items onto the target grid (first valid value wins)."""
    import rasterio
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    out = np.zeros((height, width), dtype); filled = np.zeros((height, width), bool)
    for it in items:
        href = it.assets[asset_key].href
        with rasterio.open(href) as src:
            tmp = np.zeros((height, width), dtype)
            reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=f'EPSG:{epsg}',
                      resampling=Resampling.bilinear if resampling == 'bilinear' else Resampling.nearest, dst_nodata=nodata_fill)
            m = (tmp != nodata_fill) & ~filled
            out[m] = tmp[m]; filled |= m
    return out, filled


def clear_fraction(items, assets, epsg, transform, width, height):
    scl, filled = read_into_grid(items, assets['SCL'], epsg, transform, width, height, 'nearest', np.uint8, 0)
    bad = np.isin(scl, MASK_SCL) | ~filled
    return 1.0 - bad.mean(), filled.mean(), scl


def choose_scene(client, assets, bbox, grid, start, end, min_coverage=0.99, max_candidates=40, exclude=()):
    """Return (datetime_key, items, clear_fraction, coverage, scl) of the clearest acquisition in [start, end], or None.
    exclude: datetime keys whose bands could not be read (the caller retries without them)."""
    epsg, transform, width, height = grid
    groups = search_items(client, bbox, start, end)
    best = None
    # cheap pre-sort: scene-level cloud cover of the group's least cloudy tile
    keys = sorted(groups, key=lambda k: min(it.properties.get('eo:cloud_cover', 100) for it in groups[k]))[:max_candidates]
    for k in keys:
        if k in exclude:
            continue
        try:
            clear, cov, scl = clear_fraction(groups[k], assets, epsg, transform, width, height)
        except Exception as e:  # unreadable asset: skip
            log('  skip', k, e); continue
        if cov < min_coverage:
            continue
        if best is None or clear > best[2] or (clear == best[2] and k < best[0]):
            best = (k, groups[k], clear, cov, scl)
        if best[2] >= 0.999:
            break
    return best


# ---------------------------------------------------------------------------------------------------------------- build
_COUNTER = {'i': -1}


def in_batch(args):
    _COUNTER['i'] += 1
    if not args.batch:
        return True
    k, n = (int(v) for v in args.batch.split('/'))
    return _COUNTER['i'] % n == k


def event_has_reference(r, nbac, forced_dates=None):
    """True when the original event window contains an NBAC polygon of its fire year (the same test build_window applies)."""
    if forced_dates:
        return True
    epsg, transform, width, height = window_grid(float(r['agency_lon']), float(r['agency_lat']), res=10.0)
    from rasterio.transform import array_bounds
    from pyproj import Transformer
    b = array_bounds(height, width, transform)
    tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lons, lats = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    return bool(nbac_in_window(nbac, int(r['fire_year']), (min(lons), min(lats), max(lons), max(lats))))


def build_window(client, assets, wid, lon, lat, year, nbac, args, forced_dates=None, meta=None):
    if not in_batch(args):
        return None
    epsg, transform, width, height = window_grid(lon, lat, res=10.0)
    from rasterio.transform import array_bounds
    from pyproj import Transformer
    b = array_bounds(height, width, transform)  # (left, bottom, right, top) in UTM
    tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lons, lats = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    bbox = (min(lons), min(lats), max(lons), max(lats))
    polys = nbac_in_window(nbac, year, bbox)
    sdate, edate = nbac_dates(polys)
    prior = []
    for py in range(year - args.prior_years, year):
        prior += nbac_in_window(nbac, py, bbox)
    if forced_dates:
        start, end = forced_dates
        if not polys and not (meta or {}).get('is_negative'):
            log(f'{wid}: no NBAC polygon of {year} in the window (forced dates) - built with an all-unburned reference')
    else:
        if edate is None:
            log(f'{wid}: no NBAC polygon of {year} in the window -> skipped'); return None
        e = datetime.date.fromisoformat(edate) + datetime.timedelta(days=args.settle_days)
        start = e.isoformat(); end = f'{year}-{args.season_end}'
        if start > end:
            start, end = f'{year + 1}-{args.season_start}', f'{year + 1}-{args.season_end}'
    row = dict(window_id=wid, fire_year=year, lon=lon, lat=lat, epsg=epsg, width=width, height=height, bounds_utm=';'.join(f'{v:.0f}' for v in b),
               nbac_fire_ids=';'.join(sorted({str(p.get('NFIREID', p.get('FIRE_ID', ''))) for _, p, _ in polys})), nbac_sdate=sdate, nbac_edate=edate,
               nbac_ag_dates=';'.join(sorted({f"{p.get('AG_SDATE', '')}/{p.get('AG_EDATE', '')}" for _, p, _ in polys}))[:200],
               nbac_hs_dates=';'.join(sorted({f"{p.get('HS_SDATE', '')}/{p.get('HS_EDATE', '')}" for _, p, _ in polys}))[:200],
               search_start=start, search_end=end, prior_year_polygons=len(prior))
    if meta:
        row.update(meta)
    if args.dry_run:
        log(f'{wid}: {len(polys)} NBAC polygons of {year}, fire {sdate}..{edate}, search {start}..{end}, grid {width}x{height} EPSG:{epsg}')
        return row
    grid = (epsg, transform, width, height); failed = set()
    for attempt in range(4):   # a scene whose SCL is readable but a band is not (missing object, 403) -> try the next-best scene
        best = choose_scene(client, assets, bbox, grid, start, end, exclude=failed)
        if attempt == 0 and not forced_dates and start[:4] == str(year) and (best is None or best[2] < args.min_clear):
            # no usable scene, or only a cloudy one, left in the fire season -> the following season is searched as well
            s2, e2 = f'{year + 1}-{args.season_start}', f'{year + 1}-{args.season_end}'
            alt = choose_scene(client, assets, bbox, grid, s2, e2, exclude=failed)
            if alt is not None and (best is None or alt[2] > best[2]):
                row.update(search_start=s2, search_end=e2, fallback_reason='no scene in the fire season' if best is None else f'best same-season scene only {best[2]:.2f} clear')
                best = alt; start, end = s2, e2
        if best is None:
            log(f'{wid}: no usable scene'); row['status'] = 'NO_SCENE'; return row
        key, items, clear, cov, scl = best
        baseline = max(item_baseline(it) for it in items)
        offset_applied = all(bool(it.properties.get('earthsearch:boa_offset_applied', False)) for it in items)
        dns = []
        try:
            for band in BANDS[:6]:
                dn, _ = read_into_grid(items, assets[band], epsg, transform, width, height, 'bilinear' if band in ('SWIR1', 'SWIR2') else 'nearest', np.float32, 0)
                dns.append(dn)
            break
        except Exception as e:
            log(f'{wid}: band read failed for {key} ({e}); trying the next scene'); failed.add(key)
    else:
        log(f'{wid}: bands unreadable for every candidate'); row['status'] = 'READ_FAILED'; return row
    # BOA offset: the catalogue rule (baseline >= 04.00 and not yet applied -> subtract 1000) is checked against the data
    # (offset_in_dn), because for three 2022 scenes the catalogue flag was wrong and the subtraction destroyed the dark pixels.
    subtract = baseline >= 4.0 and not offset_applied
    dn_p1, present = offset_in_dn(dns[0], ~np.isin(scl, MASK_SCL))
    offset_note = ''
    if subtract and present is False:
        subtract = False; offset_note = f'catalogue flag says offset not applied, but blue DN p1 = {dn_p1:.0f} shows no offset: not subtracted'
    elif not subtract and present is True:
        offset_note = f'blue DN p1 = {dn_p1:.0f}: offset may still be in the DN (not subtracted; check this window)'
    if offset_note:
        log(f'{wid}: {offset_note}')
    refl = [harmonise(dn, baseline, offset_applied, subtract=subtract) for dn in dns]
    del dns
    stack = indices(np.stack(refl))
    bad = np.isin(scl, MASK_SCL)
    stack[:, bad] = np.nan
    # labels
    import rasterio
    from rasterio.features import rasterize
    from pyproj import Transformer as T
    from shapely.ops import transform as stransform
    to_utm = T.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    def utm(g): return stransform(lambda x, y, z=None: to_utm.transform(x, y), g)
    label = np.zeros((height, width), np.uint8)
    if prior:
        label = rasterize([(utm(g), IGNORE) for _, _, g in prior], out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
    if polys:
        burned = rasterize([(utm(g), 1) for _, _, g in polys], out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
        label[burned == 1] = 1
    # scene from the following season: fires of that year that had already started by the acquisition date are burned in the
    # image but are not part of the reference year -> ignored (255), like the prior-year scars
    scene_year, scene_date = int(key[:4]), key[:10]; later = []
    if scene_year > year:
        for (_, p, g) in nbac_in_window(nbac, scene_year, bbox):
            s0, _ = nbac_dates([(scene_year, p, g)])
            if s0 is None or s0 <= scene_date:
                later.append(g)
        if later:
            lab2 = rasterize([(utm(g), 1) for g in later], out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
            label[lab2 == 1] = IGNORE
    profile = dict(crs=f'EPSG:{epsg}', transform=transform, width=width, height=height)
    out_dir = os.path.join(args.out, wid)
    write_gtiff(os.path.join(out_dir, wid + '_S2.tif'), stack, profile, 'float32', nodata=None)
    write_gtiff(os.path.join(out_dir, wid + '_label.tif'), label, profile, 'uint8', nodata=None)
    write_gtiff(os.path.join(out_dir, wid + '_scl.tif'), scl, profile, 'uint8', nodata=0)
    row.update(status='OK', acquisition=key, product_ids=';'.join(it.id for it in items), tiles=';'.join(sorted({tile_of(it) for it in items})),
               processing_baseline=baseline, boa_offset_applied_by_catalogue=offset_applied, boa_offset_subtracted=subtract,
               blue_dn_p1=round(dn_p1, 1) if dn_p1 == dn_p1 else '', offset_note=offset_note, clear_fraction=round(clear, 4), coverage=round(cov, 4),
               burned_ha=round(float((label == 1).sum()) / 100, 1), ignored_ha=round(float((label == IGNORE).sum()) / 100, 1),
               scene_year_offset=scene_year - year, later_fires_ignored=len(later),
               masked_fraction=round(float(bad.mean()), 4))
    log(f"{wid}: {key} clear {clear:.3f} burned {row['burned_ha']} ha (NBAC {sdate}..{edate})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=os.path.join(os.path.dirname(__file__), 'inputs', 'event_table_recovered.csv'))
    ap.add_argument('--nbac', default=None, help='folder with the unzipped NBAC shapefiles (fetch_nbac.sh)')
    ap.add_argument('--merge-parts', action='store_true', help='merge the manifest_part*.csv files written by --batch runs into manifest.csv and exit')
    ap.add_argument('--out', required=True)
    ap.add_argument('--catalogue', default='es', choices=['es', 'pc'])
    ap.add_argument('--only', default=None, help='build only these windows (event names, comma-separated)')
    ap.add_argument('--batch', default=None, help='k/n: build only every n-th window starting at k (0-based), for several processes in parallel')
    ap.add_argument('--extra-fires', type=int, default=0, help='total extra NBAC fires, split equally over AB/BC/SK (ignored when --per-province is set)')
    ap.add_argument('--per-province', type=int, default=0, help='target number of fire windows per province (originals + extra NBAC fires), e.g. 40 -> AB/BC/SK each get 40')
    ap.add_argument('--negatives-per-province', type=int, default=0, help='fire-free windows per province (R1.8), sampled around that province\'s fires')
    ap.add_argument('--negative-windows', type=int, default=0, help='windows with no NBAC burn of the fire year or the prior years (R1.8 false-positive test)')
    ap.add_argument('--min-ha', type=float, default=2000)
    ap.add_argument('--min-clear', type=float, default=0.8, help='if the clearest same-season scene is below this clear fraction, the next season is searched too and the clearer scene is used')
    ap.add_argument('--prior-years', type=int, default=2)
    ap.add_argument('--settle-days', type=int, default=7)
    ap.add_argument('--season-start', default='06-01')
    ap.add_argument('--season-end', default='10-15')
    ap.add_argument('--dates-from', default=None, help='CSV with window_id,start,end to force the search range (e.g. to rebuild the original early scenes)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.merge_parts:
        return merge_parts(args.out)
    if not args.nbac:
        ap.error('--nbac is required (unless --merge-parts)')
    events = list(csv.DictReader(open(args.events, encoding='utf-8')))
    fire_years = {int(r['fire_year']) for r in events} | ({2022, 2023, 2024} if (args.extra_fires or args.per_province) else set())
    years = list(range(min(fire_years) - args.prior_years, max(fire_years) + 2))   # +1: scenes from the following season need that year's fires
    nbac = load_nbac(args.nbac, years, required=fire_years)
    client, assets = (None, ES_ASSETS) if args.dry_run else open_catalogue(args.catalogue)
    forced = {}
    if args.dates_from:
        for r in csv.DictReader(open(args.dates_from)):
            forced[r['window_id']] = (r['start'], r['end'])
    rows = []
    for r in events:
        wid = r['event'].replace(' ', '_')
        if args.only and r['event'] not in args.only.split(',') and wid not in args.only.split(','):
            continue
        meta = dict(province=r['province'], agency_fire_id=r['agency_fire_id'], agency_fire_name=r['agency_fire_name'], source='original 32 windows')
        row = build_window(client, assets, wid, float(r['agency_lon']), float(r['agency_lat']), int(r['fire_year']), nbac, args, forced.get(wid), meta)
        if row: rows.append(row)
    if args.extra_fires or args.per_province:
        from shapely.geometry import Point
        rng = np.random.default_rng(args.seed)
        existing = [Point(float(r['agency_lon']), float(r['agency_lat'])) for r in events]
        n_existing = {pv: sum(1 for r in events if r['province'] == pv and event_has_reference(r, nbac, forced.get(r['event'].replace(' ', '_'))))
                      for pv in ('AB', 'BC', 'SK')}   # originals that build_window will keep (NBAC polygon of the fire year inside the window)
        cands = []
        for (y, p, g) in nbac:
            if y not in (2022, 2023, 2024): continue
            if province_of(p) not in ('AB', 'BC', 'SK'): continue
            ha = float(p.get('POLY_HA', p.get('AREA_HA', 0)) or 0)
            if ha < args.min_ha: continue
            c = g.centroid
            if any(c.distance(e) < 0.35 for e in existing): continue   # ~25 km from the original windows
            cands.append((y, p, c, ha))
        prov = {}
        for c in cands: prov.setdefault(province_of(c[1]), []).append(c)
        for pv in ('AB', 'BC', 'SK'):
            need = (args.per_province - n_existing[pv]) if args.per_province else max(1, args.extra_fires // 3)
            pool = sorted(prov.get(pv, []), key=lambda c: c[3])          # by size: sample equally from small / medium / large eligible fires
            if need <= 0 or not pool:
                log(f'{pv}: {n_existing[pv]} original windows, {len(pool)} eligible NBAC fires, {max(need, 0)} to add'); continue
            bins = [pool[i::3] for i in range(3)] if len(pool) >= 3 else [pool]
            chosen = []; i = 0
            while len(chosen) < need and any(bins):
                b = bins[i % len(bins)]; i += 1
                if not b:
                    continue
                cand = b.pop(int(rng.integers(len(b))))
                if any(cand[2].distance(e) < 0.35 for e in existing):   # keep >= ~25 km between all windows
                    continue
                chosen.append(cand); existing.append(cand[2])
            log(f'{pv}: {n_existing[pv]} original windows + {len(chosen)} extra NBAC fires (eligible {len(pool)}) -> {n_existing[pv] + len(chosen)}')
            for (y, p, c, ha) in chosen:
                wid = f"NBAC{y}_{pv}_{int(p.get('NFIREID', 0) or 0)}"
                if args.only and wid not in args.only.split(','):
                    continue
                meta = dict(province=pv, agency_fire_id=str(p.get('NFIREID', '')), agency_fire_name='', source='extra NBAC fire', nbac_poly_ha=ha)
                row = build_window(client, assets, wid, c.x, c.y, y, nbac, args, None, meta)
                if row: rows.append(row)
    if args.negative_windows or args.negatives_per_province:
        import math as _m
        from shapely.geometry import Point
        rng = np.random.default_rng(args.seed + 1)
        plan = {pv: args.negatives_per_province for pv in ('AB', 'BC', 'SK')} if args.negatives_per_province else {pv: args.negative_windows // 3 for pv in ('AB', 'BC', 'SK')}
        for pv, n_neg in plan.items():
            srcs = [r for r in events if r['province'] == pv]; made = 0; tries = 0
            while made < n_neg and tries < 60 * max(1, n_neg):
                tries += 1
                r = srcs[rng.integers(len(srcs))]; year = int(r['fire_year'])
                dist = rng.uniform(40000, 90000); ang = rng.uniform(0, 2 * _m.pi)
                lat0 = float(r['agency_lat']); lon0 = float(r['agency_lon'])
                lat = lat0 + dist * _m.cos(ang) / 110574.0; lon = lon0 + dist * _m.sin(ang) / (111320.0 * _m.cos(_m.radians(lat0)))
                if not (48.9 < lat < 60.1):
                    continue
                dlat = 10500 / 110574.0; dlon = 10500 / (111320.0 * _m.cos(_m.radians(lat)))
                bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
                if any(nbac_in_window(nbac, y, bbox) for y in range(year - args.prior_years, year + 1)):
                    continue
                made += 1   # counted before the batch filter so that every parallel process sees the same sequence of negative windows
                wid = f'NEG{year}_{pv}_{made:02d}'
                if args.only and wid not in args.only.split(','):
                    continue
                meta = dict(province=pv, agency_fire_id='', agency_fire_name='', source=f'negative window (no NBAC burn {year - args.prior_years}-{year}), offset from {r["event"]}', is_negative=True)
                row = build_window(client, assets, wid, lon, lat, year, nbac, args, (f'{year}-08-01', f'{year}-09-30'), meta)
                if row:
                    rows.append(row)
    if args.batch:
        k, n = (int(v) for v in args.batch.split('/')); name = f'manifest{"_dryrun" if args.dry_run else ""}_part{k}of{n}.csv'   # merged later by --merge-parts
    elif args.dry_run:
        name = 'manifest_dryrun.csv'
    elif args.only:
        name = f'manifest_only_{datetime.datetime.now():%Y%m%dT%H%M%S}.csv'   # one record per rebuild; all of them are merged by --merge-parts
    else:
        name = 'manifest.csv'
    path = write_rows(os.path.join(args.out, name), rows)
    log('done:', len(rows), 'windows ->', path)
    main_manifest = os.path.join(args.out, 'manifest.csv')
    if args.only and not args.dry_run and os.path.exists(main_manifest):   # a rebuild replaces its windows' rows in the current manifest
        merged = {r['window_id']: r for r in csv.DictReader(open(main_manifest, encoding='utf-8'))}
        for r in rows:
            merged[r['window_id']] = {k: ('' if v is None else v) for k, v in r.items()}
        write_rows(main_manifest, list(merged.values()))
        log(f'updated {main_manifest}: rows replaced for', ', '.join(r['window_id'] for r in rows))


MANIFEST_COLUMNS = ['window_id', 'province', 'source', 'status', 'fire_year', 'lon', 'lat', 'epsg', 'width', 'height', 'bounds_utm', 'nbac_fire_ids',
                    'nbac_sdate', 'nbac_edate', 'nbac_ag_dates', 'nbac_hs_dates', 'search_start', 'search_end', 'acquisition', 'product_ids', 'tiles', 'processing_baseline',
                    'boa_offset_applied_by_catalogue', 'boa_offset_subtracted', 'blue_dn_p1', 'offset_note', 'clear_fraction', 'coverage', 'burned_ha', 'ignored_ha', 'prior_year_polygons',
                    'masked_fraction', 'scene_year_offset', 'later_fires_ignored', 'fallback_reason', 'agency_fire_id', 'agency_fire_name', 'nbac_poly_ha', 'is_negative']


def write_rows(path, rows):
    """Write rows with one fixed column order (every row gets every column) so that part files can be concatenated safely."""
    keys = list(MANIFEST_COLUMNS)
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader(); w.writerows(rows)
    return path


def merge_parts(out_dir):
    """Join manifest_part<k>of<n>.csv (one per --batch process) into manifest.csv; later parts win on a duplicated window_id."""
    import glob
    parts = sorted(glob.glob(os.path.join(out_dir, 'manifest_part*of*.csv')))
    if not parts:
        raise SystemExit(f'no manifest_part*of*.csv in {out_dir}')
    parts += sorted(glob.glob(os.path.join(out_dir, 'manifest_only*.csv')))   # windows rebuilt with --only replace their earlier rows (oldest first)
    merged = {}
    for p in parts:
        for r in csv.DictReader(open(p, encoding='utf-8')):
            merged[r['window_id']] = r
    rows = list(merged.values())
    ok = sum(1 for r in rows if r.get('status') == 'OK')
    by_pv = {}
    for r in rows:
        if r.get('status') == 'OK':
            key = (r['province'], 'negative' if r.get('is_negative') in ('True', 'true', '1') else 'fire')
            by_pv[key] = by_pv.get(key, 0) + 1
    path = write_rows(os.path.join(out_dir, 'manifest.csv'), rows)
    log(f'merged {len(parts)} part files: {len(rows)} windows, {ok} with status OK ->', path)
    for key in sorted(by_pv): log(f'  {key[0]} {key[1]} windows OK: {by_pv[key]}')
    missing = [r['window_id'] for r in rows if r.get('status') != 'OK']
    if missing: log('  not built (NO_SCENE etc.):', ', '.join(missing))
    return path


if __name__ == '__main__':
    main()
