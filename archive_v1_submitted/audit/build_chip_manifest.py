#!/usr/bin/env python3
"""Build chip and event manifests from the ArcGIS 'Export Training Data' folders.

Reads, per chip: the .tfw world file (footprint), the label .tif (class pixel counts),
and per event: esri_model_definition.emd (CRS wkid, tile size).
Writes: chip_manifest.csv, event_manifest.csv under <AUDIT_DIR>/splits/.
"""
import os, sys, json, csv, glob, hashlib, time
import numpy as np
from pyproj import Transformer

ROOT = os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study')   # root of the archived study folder (chip exports, model files, test predictions)
TRAIN = os.path.join(ROOT, 'Burned and Unburned Data/Burned and Unburned Training Data')
OUT = os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'splits')
os.makedirs(OUT, exist_ok=True)
READ_LABELS = '--labels' in sys.argv

def read_tfw(p):
    v = [float(x) for x in open(p).read().split()]
    # A, D, B, E, C, F  (x-scale, rot, rot, y-scale, x-origin(center of UL px), y-origin)
    return v

def chip_footprint(tfw, w=256, h=256):
    A, D, B, E, C, F = tfw
    # tfw gives centre of upper-left pixel; convert to edges
    x0 = C - A/2.0; y0 = F - E/2.0
    x1 = x0 + A*w;  y1 = y0 + E*h
    return min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)

chip_rows = []; event_rows = []
t0 = time.time()
for prov in ['AB','BC','SK']:
    for ev_top in sorted(glob.glob(os.path.join(TRAIN, f'{prov} Chips', '*'))):
        ev = os.path.basename(ev_top)
        ev_dir = ev_top
        if not os.path.exists(os.path.join(ev_dir, 'esri_model_definition.emd')):
            nested = os.path.join(ev_top, ev)          # e.g. 'SK Chips/SMITH/SMITH' (exported 2026-02-11)
            if os.path.exists(os.path.join(nested, 'esri_model_definition.emd')):
                ev_dir = nested
            else:
                event_rows.append(dict(province_folder=prov, event=ev, status='EMPTY_FOLDER'))
                continue
        emd_p = os.path.join(ev_dir, 'esri_model_definition.emd')
        export_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(emd_p)))
        emd = json.load(open(emd_p))
        wkid = emd['MinCellSize']['spatialReference']['wkid']
        cell = emd['MinCellSize']['x']
        stats = json.load(open(os.path.join(ev_dir,'esri_accumulated_stats.json')))
        ntiles = stats['NumTiles']
        tr = Transformer.from_crs(f'EPSG:{wkid}', 'EPSG:4326', always_xy=True)
        xs=[]; ys=[]; origins=[]
        img_dir = os.path.join(ev_dir,'images'); lab_dir = os.path.join(ev_dir,'labels')
        tfws = sorted(glob.glob(os.path.join(img_dir,'*.tfw')))
        for tfw_p in tfws:
            cid = os.path.splitext(os.path.basename(tfw_p))[0]
            tfw = read_tfw(tfw_p)
            x0,y0,x1,y1 = chip_footprint(tfw)
            origins.append((tfw[4], tfw[5]))
            xs += [x0,x1]; ys += [y0,y1]
            row = dict(province_folder=prov, event=ev, export_folder=os.path.relpath(ev_dir, TRAIN), chip_id=cid, epsg=wkid, cell=cell,
                       xmin=x0, ymin=y0, xmax=x1, ymax=y1, x_origin=tfw[4], y_origin=tfw[5])
            if READ_LABELS:
                import tifffile
                lab = tifffile.imread(os.path.join(lab_dir, cid + '.tif'))
                lab = np.asarray(lab)
                if lab.ndim == 3: lab = lab[0] if lab.shape[0] <= 4 else lab[...,0]
                vals, cnts = np.unique(lab, return_counts=True)
                d = dict(zip(vals.tolist(), cnts.tolist()))
                row.update(n0_nodata=d.get(0,0), n1_unburned=d.get(1,0), n2_burned=d.get(2,0),
                           n_other=int(lab.size - d.get(0,0) - d.get(1,0) - d.get(2,0)))
                with open(os.path.join(img_dir, cid + '.tif'),'rb') as fh:
                    row['img_sha256_16'] = hashlib.sha256(fh.read()).hexdigest()[:16]
            chip_rows.append(row)
        # stride: unique sorted x origins
        ox = sorted(set(round(o[0],3) for o in origins)); oy = sorted(set(round(o[1],3) for o in origins))
        dx = sorted(set(round(b-a,3) for a,b in zip(ox[:-1], ox[1:]))); dy = sorted(set(round(b-a,3) for a,b in zip(oy[:-1], oy[1:])))
        cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
        lon, lat = tr.transform(cx, cy)
        event_rows.append(dict(province_folder=prov, event=ev, status='OK', export_folder=os.path.relpath(ev_dir, TRAIN), export_date=export_date,
                               in_training_pool=(export_date < '2026-01-06'), epsg=wkid, cell_m=cell, n_chips=len(tfws),
                               n_tiles_stats=ntiles, n_cols=len(ox), n_rows=len(oy), stride_x_m=dx, stride_y_m=dy,
                               xmin=min(xs), ymin=min(ys), xmax=max(xs), ymax=max(ys),
                               extent_x_km=round((max(xs)-min(xs))/1000,3), extent_y_km=round((max(ys)-min(ys))/1000,3),
                               centroid_lon=round(lon,5), centroid_lat=round(lat,5),
                               burned_px_total=stats['ClassPixelStats']['NumPixelsPerClass'][1],
                               unburned_px_total=stats['ClassPixelStats']['NumPixelsPerClass'][0],
                               chips_with_burned=stats['ClassPixelStats']['NumImagesPerClass'][1]))
        print(f'{prov} {ev}: {len(tfws)} chips, EPSG:{wkid}, stride {dx}/{dy}, centroid {lon:.3f},{lat:.3f}  [{time.time()-t0:.0f}s]', flush=True)

with open(os.path.join(OUT,'chip_manifest_trainingexports.csv'),'w',newline='') as fh:
    keys = sorted({k for r in chip_rows for k in r}); w=csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(chip_rows)
with open(os.path.join(OUT,'event_manifest.csv'),'w',newline='') as fh:
    keys = sorted({k for r in event_rows for k in r}); w=csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(event_rows)
print('wrote', len(chip_rows), 'chips,', len(event_rows), 'events')
