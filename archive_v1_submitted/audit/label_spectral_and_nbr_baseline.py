#!/usr/bin/env python3
"""Label spectral-consistency diagnostic and same-information NBR baseline on the submitted chips.
Reads the NON-OVERLAPPING subset of export chips (even grid rows/cols = every other chip) for all 30 events,
so each ground pixel is counted once. Per event: NBR/NDVI distributions by label, Otsu threshold on NBR
(label-free, per event), burned-class metrics of 'NBR < Otsu' against the manual labels, and the same on the
Table-7 test chips. Outputs in <AUDIT_DIR>/results/: label_spectral_consistency.csv, nbr_otsu_baseline_events.csv,
nbr_otsu_baseline_testchips.csv. Band order: 0 B2,1 B3,2 B4,3 B8,4 B11,5 B12,6 NDVI,7 NBR,8 NDMI (GEE script).
"""
import os, glob, json, csv, time, sys
import numpy as np, tifffile, pandas as pd
ROOT=os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study')   # root of the archived study folder (chip exports, model files, test predictions)
TRAIN=os.path.join(ROOT,'Burned and Unburned Data/Burned and Unburned Training Data')
OUT=os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'results'); os.makedirs(OUT,exist_ok=True)
man=pd.read_csv(os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'splits', 'chip_manifest_trainingexports.csv'),dtype={'chip_id':str})
def otsu(x,bins=256):
    h,edges=np.histogram(x,bins=bins); c=(edges[:-1]+edges[1:])/2
    w0=np.cumsum(h); w1=w0[-1]-w0; m0=np.cumsum(h*c)/np.maximum(w0,1); m1=(np.cumsum((h*c)[::-1])[::-1]/np.maximum(w1,1))
    var=w0[:-1]*w1[:-1]*(m0[:-1]-m1[1:])**2; k=np.argmax(var); return c[k]
def metrics(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else np.nan; R=tp/(tp+fn) if tp+fn else np.nan; F=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan; I=tp/(tp+fp+fn) if tp+fp+fn else np.nan
    return P,R,F,I
spec_rows=[]; base_rows=[]; t0=time.time()
CACHE=os.path.join(OUT,'cache_nbr'); os.makedirs(CACHE,exist_ok=True)
BUDGET=float(os.environ.get('BUDGET_S','150'))
pending=False
for (prov,ev),g in man.groupby(['province_folder','event']):
    cf=os.path.join(CACHE,f'{prov}__{ev}.json')
    if os.path.exists(cf):
        d=json.load(open(cf)); spec_rows.append(d['spec']); base_rows.append(d['base']); continue
    if time.time()-t0>BUDGET:
        pending=True; print('BUDGET reached; rerun for remaining events',flush=True); break
    # non-overlapping subset: chips whose origin index (x,y) are both even
    xs=sorted(g.x_origin.unique()); ys=sorted(g.y_origin.unique())
    xi={v:i for i,v in enumerate(xs)}; yi={v:i for i,v in enumerate(ys)}
    sub=g[[ (xi[r.x_origin]%2==0 and yi[r.y_origin]%2==0) for r in g.itertuples()]]
    nbr_b=[]; nbr_u=[]; ndvi_b=[]; ndvi_u=[]; allnbr=[]; labs=[]
    for r in sub.itertuples():
        im=tifffile.imread(os.path.join(TRAIN,f'{prov} Chips',ev,'images',r.chip_id+'.tif')); im=np.transpose(im,(1,2,0)) if im.shape[0]==9 else im
        lb=tifffile.imread(os.path.join(TRAIN,f'{prov} Chips',ev,'labels',r.chip_id+'.tif')); lb=lb[0] if lb.ndim==3 else lb
        valid=~np.isnan(im).any(2)&((lb==1)|(lb==2))
        nbr=im[...,7]; ndvi=im[...,6]
        nbr_b.append(nbr[valid&(lb==2)]); nbr_u.append(nbr[valid&(lb==1)]); ndvi_b.append(ndvi[valid&(lb==2)]); ndvi_u.append(ndvi[valid&(lb==1)])
        allnbr.append(nbr[valid]); labs.append(lb[valid])
    nb=np.concatenate(nbr_b) if nbr_b else np.array([]); nu=np.concatenate(nbr_u); db=np.concatenate(ndvi_b) if ndvi_b else np.array([]); du=np.concatenate(ndvi_u)
    A=np.concatenate(allnbr); L=np.concatenate(labs)
    thr=float(otsu(A[np.isfinite(A)]))
    pred=(A<thr)
    tp=int((pred&(L==2)).sum()); fp=int((pred&(L==1)).sum()); fn=int((~pred&(L==2)).sum())
    P,R,F,I=metrics(tp,fp,fn)
    med_u=float(np.median(nu)) if nu.size else np.nan
    spec_rows.append(dict(province_folder=prov,event=ev,n_chips_nonoverlap=len(sub),n_burned_px=int(nb.size),n_unburned_px=int(nu.size),
        nbr_burned_median=float(np.median(nb)) if nb.size else np.nan, nbr_burned_p90=float(np.percentile(nb,90)) if nb.size else np.nan,
        nbr_unburned_median=med_u, nbr_unburned_p10=float(np.percentile(nu,10)) if nu.size else np.nan,
        ndvi_burned_median=float(np.median(db)) if db.size else np.nan, ndvi_unburned_median=float(np.median(du)) if du.size else np.nan,
        share_burned_px_with_nbr_above_unburned_median=float((nb>med_u).mean()) if nb.size else np.nan,
        share_unburned_px_with_nbr_below_burned_median=float((nu<np.median(nb)).mean()) if nb.size else np.nan))
    base_rows.append(dict(province_folder=prov,event=ev,otsu_nbr_threshold=round(thr,4),n_px=int(L.size),burned_share=round(float((L==2).mean()),4),tp=tp,fp=fp,fn=fn,P=P,R=R,F1=F,IoU=I))
    print(f'{prov} {ev}: {len(sub)} chips, otsu {thr:.3f}, NBR-Otsu burned F1 {F:.3f} [{time.time()-t0:.0f}s]',flush=True)
    json.dump({'spec':spec_rows[-1],'base':base_rows[-1]},open(cf,'w'))
if pending: sys.exit(0)
pd.DataFrame(spec_rows).to_csv(os.path.join(OUT,'label_spectral_consistency.csv'),index=False)
pd.DataFrame(base_rows).to_csv(os.path.join(OUT,'nbr_otsu_baseline_events.csv'),index=False)
# Table-7 test chips: apply the event's Otsu threshold (label-free) to the 30/40 selected chips
TESTS={'Betts':('Betts','AB'),'BOLDING':('BOLDING','SK'),'rossmere lake':('Roosemoore','BC')}
thr_by={r['event']:r['otsu_nbr_threshold'] for r in base_rows}
rows=[]
for tdir,(ev,prov) in TESTS.items():
    tp=fp=fn=0
    for lp in sorted(glob.glob(os.path.join(ROOT,'Test Locations',tdir,'labels','*.tif'))):
        cid=os.path.splitext(os.path.basename(lp))[0]
        im=tifffile.imread(os.path.join(ROOT,'Test Locations',tdir,'images',cid+'.tif')); im=np.transpose(im,(1,2,0)) if im.shape[0]==9 else im
        lb=tifffile.imread(lp); lb=lb[0] if lb.ndim==3 else lb
        valid=~np.isnan(im).any(2)&((lb==1)|(lb==2)); pred=(im[...,7]<thr_by[ev])&valid
        tp+=int((pred&(lb==2)).sum()); fp+=int((pred&(lb==1)).sum()); fn+=int((~pred&valid&(lb==2)).sum())
    P,R,F,I=metrics(tp,fp,fn); rows.append(dict(test_folder=tdir,event=ev,true_province=prov,otsu_nbr_threshold=thr_by[ev],tp=tp,fp=fp,fn=fn,P=P,R=R,F1=F,IoU=I))
    print('TEST',tdir,rows[-1],flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT,'nbr_otsu_baseline_testchips.csv'),index=False)
print('DONE')
