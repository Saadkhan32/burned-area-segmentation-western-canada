#!/usr/bin/env python3
"""Held-out check (added 2026-10-12 after the folder search).
SMITH (SK) and Wentzel Fire (AB) were exported on 2026-02-11 (ArcGIS geoprocessing history), after all five final models had
been trained (4-23 Jan 2026), so they are the only archived events the original models never saw. The archive holds DeepLabV3
predictions for four selected-chip test sets (30 or 60 chips each). This script re-scores them on unique ground pixels
(nearest-chip-centre mosaic, as unique_pixel_recount.py) and adds a spectral check: mean NBR / NDVI of TP, FP, FN, TN pixels,
to see whether 'false positives' look burned (possible label omission) or unburned (model error).
Output: <AUDIT_DIR>/results/heldout_events_SMITH_Wentzel_deeplab.csv and heldout_SMITH_Wentzel_spectral_check.csv"""
import os, glob, numpy as np, tifffile, pandas as pd
ROOT=os.path.join(os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study'), 'Test Locations')
OUT=os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'results')
SETS={'SMITH':'deeplab_pred','SMITH with more images':'pred_deeplab','Wentzel':'pred_deeplab','Wentzel with more images':'deeplab_pred'}
def tfw_origin(p):
    v=[float(x) for x in open(p).read().split()]; return v[4]-v[0]/2, v[5]-v[3]/2
rows=[]; spec=[]
for ev,sub in SETS.items():
    base=os.path.join(ROOT,ev); chips={}
    for lp in sorted(glob.glob(os.path.join(base,'labels','*.tif'))):
        cid=os.path.splitext(os.path.basename(lp))[0]
        t=glob.glob(os.path.join(base,sub,cid+'*.tfw'))
        if t: chips[cid]=tfw_origin(t[0])
    ids=list(chips); xs=[o[0] for o in chips.values()]; ys=[o[1] for o in chips.values()]
    x0=min(xs); y0=max(ys); W=int((max(xs)-x0)/10)+256; H=int((y0-min(ys))/10)+256
    best=np.full((H,W),np.inf); owner=np.full((H,W),-1,dtype=np.int32)
    yy,xx=np.mgrid[0:256,0:256]; dc=np.sqrt((yy-127.5)**2+(xx-127.5)**2)
    for k,cid in enumerate(ids):
        c=int((chips[cid][0]-x0)/10); r=int((y0-chips[cid][1])/10)
        s=best[r:r+256,c:c+256]; m=dc<s; s[m]=dc[m]; owner[r:r+256,c:c+256][m]=k
    def mosaic(reader,dtype=np.int16,fill=0,bands=None):
        M=np.full((H,W) if bands is None else (H,W,bands),fill,dtype=dtype)
        for k,cid in enumerate(ids):
            a=reader(cid)
            if a is None: continue
            c=int((chips[cid][0]-x0)/10); r=int((y0-chips[cid][1])/10)
            sel=(owner[r:r+256,c:c+256]==k); M[r:r+256,c:c+256][sel]=a[sel]
        return M
    def rd_label(cid):
        a=tifffile.imread(os.path.join(base,'labels',cid+'.tif')); return (a[0] if a.ndim==3 else a).astype(np.int16)
    def rd_img(cid):
        im=tifffile.imread(os.path.join(base,'images',cid+'.tif')); return (np.transpose(im,(1,2,0)) if im.shape[0]==9 else im).astype(np.float32)
    def rd_pred(cid):
        pp=glob.glob(os.path.join(base,sub,cid+'_Classified.tif'))
        if not pp: return None
        a=tifffile.imread(pp[0]); return (a[0] if a.ndim==3 else a).astype(np.int16)
    L=mosaic(rd_label); P=mosaic(rd_pred); I=mosaic(rd_img,dtype=np.float32,fill=np.nan,bands=9)
    valid=((L==1)|(L==2))&~np.isnan(I).any(2)
    pb=(P==2); lb=(L==2)
    grp={'TP':pb&lb&valid,'FP':pb&~lb&valid,'FN':~pb&lb&valid,'TN':~pb&~lb&valid}
    tp,fp,fn,tn=(int(grp[g].sum()) for g in ('TP','FP','FN','TN'))
    P_=tp/(tp+fp) if tp+fp else np.nan; R_=tp/(tp+fn) if tp+fn else np.nan
    rows.append(dict(test_set=ev,model='DeepLabV3 (original)',n_chips=len(ids),unique_labelled_valid_px=int(valid.sum()),burned_share=round((tp+fn)/valid.sum(),4),
                     tp=tp,fp=fp,fn=fn,tn=tn,P=round(P_,4),R=round(R_,4),F1=round(2*tp/(2*tp+fp+fn),4),IoU=round(tp/(tp+fp+fn),4),
                     ref_burned_ha=round((tp+fn)/100,1),pred_burned_ha=round((tp+fp)/100,1)))
    for g,m in grp.items():
        if m.sum()==0: continue
        spec.append(dict(test_set=ev,group=g,n_px=int(m.sum()),NBR_mean=round(float(np.nanmean(I[...,7][m])),3),NBR_p25=round(float(np.nanpercentile(I[...,7][m],25)),3),
                         NBR_p75=round(float(np.nanpercentile(I[...,7][m],75)),3),NDVI_mean=round(float(np.nanmean(I[...,6][m])),3),SWIR2_mean=round(float(np.nanmean(I[...,5][m])),3)))
pd.DataFrame(rows).to_csv(os.path.join(OUT,'heldout_events_SMITH_Wentzel_deeplab.csv'),index=False)
pd.DataFrame(spec).to_csv(os.path.join(OUT,'heldout_SMITH_Wentzel_spectral_check.csv'),index=False)
pd.set_option('display.width',220)
print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(spec).to_string())
