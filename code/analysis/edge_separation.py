#!/usr/bin/env python3
"""Closest distance between the EDGES of test and development windows per leave-one-province-out fold, and all
overlapping window pairs (with their roles in each fold). Complements tableP7 (centre-to-centre).
Window footprints are the manifest bounds (UTM of each window); distances are geodesic (WGS84) between densified
window boundaries, so windows in different UTM zones are handled exactly.
Usage: python analysis/edge_separation.py --manifest manifest.csv --folds tableP6_fold_membership.csv --out <dir>"""
import argparse, os, itertools
import pandas as pd
from pyproj import Transformer, Geod
from shapely.geometry import box

def boundary_ll(r, step_m=10.0):
    """Densified window boundary (every step_m metres) and the four corners, in lon/lat."""
    x0, y0, x1, y1 = [float(v) for v in r.bounds_utm.split(';')]
    t = Transformer.from_crs(int(r.epsg), 4326, always_xy=True)
    pts = []
    for (ax, ay), (bx, by) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        n = max(2, int(max(abs(bx - ax), abs(by - ay)) / step_m))
        pts += [t.transform(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n)]
    corners = [t.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return pts, corners

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--folds', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    man = pd.read_csv(a.manifest).set_index('window_id'); folds = pd.read_csv(a.folds).set_index('window_id')
    g = Geod(ellps='WGS84'); B = {w: boundary_ll(r) for w, r in man.iterrows()}
    cen = {w: (r.lon, r.lat) for w, r in man.iterrows()}
    def edge_km(u, v):
        # the minimum distance between two convex quadrilaterals is reached at a corner of one of them:
        # corners of u against the dense boundary of v, and corners of v against the dense boundary of u
        import numpy as np
        best = 1e12
        for (a, b) in ((u, v), (v, u)):
            lons = np.array([q[0] for q in B[b][0]]); lats = np.array([q[1] for q in B[b][0]])
            for lo, la in B[a][1]:
                best = min(best, g.inv(np.full(lons.shape, lo), np.full(lats.shape, la), lons, lats)[2].min())
        return best / 1000.0
    rows = []
    for prov, fold in (('AB', 'F-AB'), ('BC', 'F-BC'), ('SK', 'F-SK')):
        test = [w for w in man.index if man.loc[w, 'province'] == prov]; dev = [w for w in man.index if man.loc[w, 'province'] != prov]
        cand = sorted(((g.inv(*cen[u], *cen[v])[2] / 1000.0, u, v) for u in test for v in dev))[:60]
        best = min((edge_km(u, v), u, v, c) for c, u, v in cand)
        rows.append(dict(fold=fold, closest_edges_km=round(best[0], 3), test_window=best[1], development_window=best[2], their_centre_distance_km=round(best[3], 1)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'tableP7b_edge_separation.csv'), index=False)
    ov = []
    for u, v in itertools.combinations(man.index, 2):
        ru, rv = man.loc[u], man.loc[v]
        if g.inv(*cen[u], *cen[v])[2] > 32000: continue
        if int(ru.epsg) == int(rv.epsg):
            area = box(*[float(x) for x in ru.bounds_utm.split(';')]).intersection(box(*[float(x) for x in rv.bounds_utm.split(';')])).area / 1e4
            if area > 0:
                ov.append(dict(window_a=u, window_b=v, province_a=ru.province, province_b=rv.province, overlap_ha=round(area), centre_distance_km=round(g.inv(*cen[u], *cen[v])[2] / 1000.0, 2),
                               same_role_in_every_fold=all(folds.loc[u, c] == folds.loc[v, c] for c in ('role_F-AB', 'role_F-BC', 'role_F-SK'))))
        else:
            ov.append(dict(window_a=u, window_b=v, province_a=ru.province, province_b=rv.province, overlap_ha=None, centre_distance_km=round(g.inv(*cen[u], *cen[v])[2] / 1000.0, 2), same_role_in_every_fold=None))
    pd.DataFrame(ov).to_csv(os.path.join(a.out, 'window_overlaps.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(ov).to_string(index=False))

if __name__ == '__main__':
    main()
