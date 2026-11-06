#!/usr/bin/env python3
"""Reference audit: the burned polygons digitised for the submitted study against the NBAC agency perimeters.
For each of the 32 original windows: area of the manual burned polygons inside the window; share of it that
lies inside the NBAC perimeters of the fire year (agreement) and inside NBAC perimeters of any year 2020-2025 (catches earlier
scars digitised as burned); the remainder is digitised burn that no agency perimeter contains (commission against NBAC, or an
NBAC omission). Coverage of the NBAC perimeter by the manual polygons is reported only as context: the original composites
were acquired during many of the fires and the polygons cover a fraction of each window, so low coverage is expected.
Inputs: the unpacked manual_burned_polygons_32events.zip (BurnedOnly.shp per event), the NBAC GeoJSON exported on Narval
(analysis/export_nbac_geojson.py) and the dataset manifest (window centres, fire years).
Usage: python analysis/manual_vs_nbac.py --polys manual_polys/ --nbac nbac_windows_2020-2025.geojson --manifest manifest.csv --out audit_labels/"""
import os, sys, argparse, json
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import window_grid

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--polys', required=True); ap.add_argument('--nbac', required=True); ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    man = pd.read_csv(a.manifest); man = man[(man.status == 'OK') & man.source.str.contains('original', case=False)]
    nb = gpd.read_file(a.nbac); nb['year'] = nb['year'].astype(int)
    rows = []
    for _, r in man.iterrows():
        wid = r.window_id; folder = os.path.join(a.polys, wid)
        if not os.path.exists(os.path.join(folder, 'BurnedOnly.shp')):
            print('no polygons for', wid); continue
        epsg, transform, width, height = window_grid(float(r.lon), float(r.lat), res=10.0)
        from rasterio.transform import array_bounds
        wb = box(*array_bounds(height, width, transform)); crs = f'EPSG:{epsg}'
        g = gpd.read_file(os.path.join(folder, 'BurnedOnly.shp')).to_crs(crs)
        manual = unary_union([geom.buffer(0) for geom in g.geometry if geom is not None]).intersection(wb)
        nbw = nb.to_crs(crs); nbw = nbw[nbw.intersects(wb)]
        fy = unary_union([geom.buffer(0) for geom in nbw[nbw.year == int(r.fire_year)].geometry]).intersection(wb) if (nbw.year == int(r.fire_year)).any() else wb.buffer(0).intersection(box(0, 0, 0, 0))
        anyy = unary_union([geom.buffer(0) for geom in nbw.geometry]).intersection(wb) if len(nbw) else fy
        prior = unary_union([geom.buffer(0) for geom in nbw[nbw.year < int(r.fire_year)].geometry]).intersection(wb) if (nbw.year < int(r.fire_year)).any() else None
        ha = lambda geom: (geom.area / 1e4) if geom is not None and not geom.is_empty else 0.0
        m_ha = ha(manual); in_fy = ha(manual.intersection(fy)); in_any = ha(manual.intersection(anyy)); in_prior = ha(manual.intersection(prior)) if prior is not None else 0.0
        rows.append({'window_id': wid, 'province': r.province, 'fire_year': int(r.fire_year), 'manual_burned_ha': round(m_ha, 1), 'nbac_fire_year_ha_in_window': round(ha(fy), 1),
                     'manual_inside_nbac_fire_year_ha': round(in_fy, 1), 'share_inside_fire_year': round(in_fy / m_ha, 3) if m_ha else np.nan,
                     'share_inside_any_year_2020_2025': round(in_any / m_ha, 3) if m_ha else np.nan, 'share_inside_prior_year_scar': round(in_prior / m_ha, 3) if m_ha else np.nan,
                     'manual_outside_all_nbac_ha': round(m_ha - in_any, 1), 'coverage_of_nbac_fire_year': round(in_fy / ha(fy), 3) if ha(fy) else np.nan})
        print(rows[-1])
    t = pd.DataFrame(rows); t.to_csv(os.path.join(a.out, 'manual_polygons_vs_nbac.csv'), index=False)
    tot_m = t.manual_burned_ha.sum(); tot_in = t.manual_inside_nbac_fire_year_ha.sum(); tot_any = tot_m - t.manual_outside_all_nbac_ha.sum()
    summary = [f'# Manual burned polygons (submitted study) against NBAC perimeters - {len(t)} windows',
               f'- Digitised burned area inside the windows: {tot_m:,.0f} ha; inside the NBAC perimeter of the fire year: {tot_in:,.0f} ha ({100 * tot_in / tot_m:.1f} %); inside any NBAC perimeter 2020-2025: {tot_any:,.0f} ha ({100 * tot_any / tot_m:.1f} %); outside every NBAC perimeter: {tot_m - tot_any:,.0f} ha ({100 * (tot_m - tot_any) / tot_m:.1f} %).',
               f'- Per window, share of the digitised burn inside the fire-year perimeter: median {t.share_inside_fire_year.median():.3f}, min {t.share_inside_fire_year.min():.3f} ({t.loc[t.share_inside_fire_year.idxmin(), "window_id"]}); windows below 0.9: {int((t.share_inside_fire_year < 0.9).sum())}.',
               f'- Earlier (2020-{int(t.fire_year.min()) - 1}) scars digitised as burned: windows with > 5 % of the digitised area in a prior-year perimeter: {int((t.share_inside_prior_year_scar > 0.05).sum())}.',
               f'- Coverage of the NBAC fire-year perimeter by the digitised polygons (context only; composites during the fire, partial digitising): median {t.coverage_of_nbac_fire_year.median():.3f}.', '',
               t.to_markdown(index=False)]
    open(os.path.join(a.out, 'manual_polygons_vs_nbac.md'), 'w').write('\n'.join(summary) + '\n'); print('\n'.join(summary[:5]))

if __name__ == '__main__':
    main()
