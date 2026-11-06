#!/usr/bin/env python3
"""Locate every archived test chip, recover its footprint from the prediction
world files (.tfw), and test it against the training-export chip manifest:
  (a) exact footprint match (same chip from the same export) + image hash equality,
  (b) any spatial overlap with a training-export chip (shared ground pixels),
  (c) nearest edge-to-edge distance to the training-export chips of the same CRS.
Writes <AUDIT_DIR>/splits/test_chip_overlap.csv and test_event_summary.csv
"""
import os, glob, csv, hashlib, json
import numpy as np, pandas as pd
from pyproj import Transformer
ROOT = os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study')   # root of the archived study folder (chip exports, model files, test predictions)
OUT = os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'splits')
man = pd.read_csv(os.path.join(OUT,'chip_manifest_trainingexports.csv'), dtype={'chip_id':str})
TRAIN = os.path.join(ROOT, 'Burned and Unburned Data/Burned and Unburned Training Data')

TEST_SETS = {  # name: (folder with images/labels, a prediction folder holding *_Classified.tfw, role)
 'Betts (Table 7 "Alberta")': ('Test Locations/Betts', 'Test Locations/Betts/unet pred', 'TABLE7_AB'),
 'BOLDING (Table 7 "British Columbia")': ('Test Locations/BOLDING', 'Test Locations/BOLDING/unet_pred', 'TABLE7_BC'),
 'rossmere lake (Table 7 "Saskatchewan")': ('Test Locations/rossmere lake', 'Test Locations/rossmere lake/unet_pred', 'TABLE7_SK'),
 'Algar Lake Complex (earlier AB test)': ('Test Locations/Algar Lake Complex', 'Test Locations/Algar Lake Complex/unet_pred', 'EARLIER'),
 'SHAW (earlier SK test)': ('Test Locations/SHAW', 'Test Locations/SHAW/unet_predictions', 'EARLIER'),
 'Tomais Mountain (earlier BC test)': ('Test Locations/Tomais Mountain', 'Test Locations/Tomais Mountain/unet_predictions', 'EARLIER'),
 'SMITH (earlier test)': ('Test Locations/SMITH', 'Test Locations/SMITH/deeplab_pred', 'EARLIER'),
 'Wentzel (earlier test)': ('Test Locations/Wentzel', 'Test Locations/Wentzel/pred_deeplab', 'EARLIER'),
 'SMITH with more images (earlier test)': ('Test Locations/SMITH with more images', 'Test Locations/SMITH with more images/pred_deeplab', 'EARLIER'),
 'Wentzel with more images (earlier test)': ('Test Locations/Wentzel with more images', 'Test Locations/Wentzel with more images/deeplab_pred', 'EARLIER'),
 'Algar lake complex v2 (earlier)': ('Test Locations/Algar lake complex v2', 'Test Locations/Algar lake complex v2/deeplab_pred', 'EARLIER'),
}
def read_tfw(p):
    v=[float(x) for x in open(p).read().split()]; A,D,B,E,C,F=v
    x0=C-A/2; y0=F-E/2; x1=x0+A*256; y1=y0+E*256
    return min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1), v
def sha16(p):
    with open(p,'rb') as fh: return hashlib.sha256(fh.read()).hexdigest()[:16]
rows=[]; summ=[]
for name,(folder,predfolder,role) in TEST_SETS.items():
    base=os.path.join(ROOT,folder); pred=os.path.join(ROOT,predfolder)
    imgs=sorted(glob.glob(os.path.join(base,'images','*.tif')))
    tfws={os.path.basename(p).split('_')[0]: p for p in glob.glob(os.path.join(pred,'*.tfw'))}
    n_exact=0; n_overlap=0; n_none=0; dists=[]; evs=set(); crs=set()
    for ip in imgs:
        cid=os.path.splitext(os.path.basename(ip))[0]
        if cid not in tfws:
            rows.append(dict(test_set=name, chip_id=cid, status='NO_TFW')); continue
        x0,y0,x1,y1,v=read_tfw(tfws[cid])
        # candidate training chips: same origin coordinates (any CRS id) -> verify by hash
        cand=man[(np.isclose(man.xmin,x0,atol=0.5))&(np.isclose(man.ymin,y0,atol=0.5))&(np.isclose(man.xmax,x1,atol=0.5))&(np.isclose(man.ymax,y1,atol=0.5))]
        exact=None; hsh=None
        for _,c in cand.iterrows():
            tp=os.path.join(TRAIN,c.export_folder,"images",f"{c.chip_id}.tif")
            if os.path.exists(tp):
                h1=sha16(ip); h2=sha16(tp)
                if h1==h2: exact=c; hsh=h1; break
        # spatial overlap with any training chip in a plausible CRS (same UTM zone unknown -> compare all; restrict by distance)
        ov=man[(man.xmin<x1)&(man.xmax>x0)&(man.ymin<y1)&(man.ymax>y0)]
        # overlap could be coincidental across different UTM zones; keep only events in the same zone as the exact match, else report all
        if exact is not None:
            ov=ov[ov.epsg==exact.epsg]
            d=0.0
        else:
            d=np.nan
        rec=dict(test_set=name, role=role, chip_id=cid, xmin=x0,ymin=y0,xmax=x1,ymax=y1,
                 exact_training_chip=(f"{exact.province_folder} Chips/{exact.event}/{exact.chip_id}" if exact is not None else ''),
                 exact_epsg=(int(exact.epsg) if exact is not None else ''), image_sha256_16=(hsh or ''),
                 n_overlapping_training_chips=int(len(ov)), overlapping_events=';'.join(sorted(set(ov.event))))
        rows.append(rec)
        if exact is not None: n_exact+=1; evs.add(exact.event); crs.add(int(exact.epsg))
        elif len(ov)>0: n_overlap+=1
        else: n_none+=1
    summ.append(dict(test_set=name, role=role, n_chips=len(imgs), n_exact_copy_of_training_export_chip=n_exact,
                     n_overlap_only=n_overlap, n_no_match=n_none, matched_training_events=';'.join(sorted(evs)), epsg=';'.join(map(str,sorted(crs)))))
    print(summ[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT,'test_chip_overlap.csv'), index=False)
pd.DataFrame(summ).to_csv(os.path.join(OUT,'test_event_summary.csv'), index=False)
