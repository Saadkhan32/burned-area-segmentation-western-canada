#!/usr/bin/env python3
"""Export the NBAC polygons that intersect the dataset windows as one GeoJSON (EPSG:4326) with the attributes used by
build_dataset.py, so that the manual burned polygons of the submitted study can be compared with the agency perimeters
off-cluster (reference audit of the paper). Runs on Narval next to the NBAC caches (load_nbac).
Usage: python analysis/export_nbac_geojson.py --nbac ~/scratch/burn/nbac --manifest ~/scratch/burn/data_postfire/manifest.csv --out ~/burn/drac/nbac_windows_2020-2025.geojson"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import window_grid
from build_dataset import load_nbac, nbac_in_window
from shapely.geometry import mapping

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nbac', required=True); ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--years', default='2020-2025'); a = ap.parse_args()
    y0, y1 = (int(v) for v in a.years.split('-')); years = list(range(y0, y1 + 1))
    nbac = load_nbac(a.nbac, years)
    rows = [r for r in csv.DictReader(open(a.manifest, encoding='utf-8')) if r.get('status', 'OK') == 'OK']
    from rasterio.transform import array_bounds
    from pyproj import Transformer
    feats = {}
    for r in rows:
        lon, lat = float(r['lon']), float(r['lat']); epsg, transform, width, height = window_grid(lon, lat, res=10.0)
        b = array_bounds(height, width, transform); tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
        lons, lats = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]]); bbox = (min(lons), min(lats), max(lons), max(lats))
        for y in years:
            for (yy, p, g) in nbac_in_window(nbac, y, bbox):
                key = (y, str(p.get('NFIREID', p.get('GID', ''))), str(p.get('ADMIN_NAME', '')))
                if key in feats:
                    feats[key]['properties']['windows'] = sorted(set(feats[key]['properties']['windows'] + [r['window_id']])); continue
                props = {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in p.items()}
                props.update(year=y, windows=[r['window_id']])
                feats[key] = dict(type='Feature', properties=props, geometry=mapping(g))
    json.dump(dict(type='FeatureCollection', name='nbac_windows', crs=dict(type='name', properties=dict(name='EPSG:4326')), features=list(feats.values())), open(a.out, 'w'))
    print('wrote', a.out, len(feats), 'polygons for', len(rows), 'windows;', os.path.getsize(a.out) / 1e6, 'MB')

if __name__ == '__main__':
    main()
