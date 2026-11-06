#!/usr/bin/env python3
"""Unique-pixel recount: the Table-7 chip sets contain adjacent 50 %-overlap chips, so pixel-pooled scores over chips
count shared ground pixels twice. Re-score on unique ground pixels: mosaic labels and predictions onto the window grid using the
chip world files (origins on a 1,280 m lattice), assigning each pixel to the chip whose centre is nearest (ties: first chip).
Output: <AUDIT_DIR>/results/test_event_metrics_uniquepixels.csv (CNNs + flagged extension models)."""
import os, glob, numpy as np, tifffile, pandas as pd
ROOT=os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study')   # root of the archived study folder (chip exports, model files, test predictions)
OUT=os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'results')
TESTS={'Betts':{'DeepLabV3':'betts pred deepv3','Mask2Former':'mmseg pred','PSPNet':'pspnet pred','SamLoRA':'samlora_pred','UNet':'unet pred'},
 'BOLDING':{'DeepLabV3':'DEEP_PRED','Mask2Former':'mmseg_pred','PSPNet':'PSPnet_pred','SamLoRA':'samlora_pred','UNet':'unet_pred'},
 'rossmere lake':{'DeepLabV3':'deeplab_pred','Mask2Former':'mmseg_pred','PSPNet':'pspnet_pred','SamLoRA':'sam_pred','UNet':'unet_pred'}}
def tfw_origin(p):
    v=[float(x) for x in open(p).read().split()]; return v[4]-v[0]/2, v[5]-v[3]/2  # UL corner x, y (y-scale negative)
rows=[]
for ev,models in TESTS.items():
    base=os.path.join(ROOT,'Test Locations',ev)
    # chip origins from any prediction folder's tfw
    tfw_dir=os.path.join(base,models['PSPNet'])
    chips={}
    for lp in sorted(glob.glob(os.path.join(base,'labels','*.tif'))):
        cid=os.path.splitext(os.path.basename(lp))[0]
        t=glob.glob(os.path.join(tfw_dir,cid+'*.tfw'))
        if not t: continue
        chips[cid]=tfw_origin(t[0])
    xs=sorted({o[0] for o in chips.values()}); ys=sorted({o[1] for o in chips.values()},reverse=True)
    x0=min(xs); y0=max(ys)
    W=int((max(xs)-x0)/10)+256; H=int((y0-min(ys))/10)+256
    # distance-to-centre map per pixel to choose the nearest chip
    best=np.full((H,W),np.inf); owner=np.full((H,W),-1,dtype=np.int32); ids=list(chips)
    yy,xx=np.mgrid[0:256,0:256]; dc=np.sqrt((yy-127.5)**2+(xx-127.5)**2)
    for k,cid in enumerate(ids):
        ox,oy=chips[cid]; c=int((ox-x0)/10); r=int((y0-oy)/10)
        sub=best[r:r+256,c:c+256]; m=dc<sub; sub[m]=dc[m]; owner[r:r+256,c:c+256][m]=k
    def mosaic(reader):
        M=np.zeros((H,W),dtype=np.int16)
        for k,cid in enumerate(ids):
            a=reader(cid)
            if a is None: continue
            ox,oy=chips[cid]; c=int((ox-x0)/10); r=int((y0-oy)/10)
            sel=(owner[r:r+256,c:c+256]==k); M[r:r+256,c:c+256][sel]=a[sel]
        return M
    def rd_label(cid):
        a=tifffile.imread(os.path.join(base,'labels',cid+'.tif')); return (a[0] if a.ndim==3 else a).astype(np.int16)
    def rd_imgvalid(cid):
        im=tifffile.imread(os.path.join(base,'images',cid+'.tif')); im=np.transpose(im,(1,2,0)) if im.shape[0]==9 else im
        return (~np.isnan(im).any(2)).astype(np.int16)
    L=mosaic(rd_label); V=mosaic(rd_imgvalid)
    n_unique=int(((L==1)|(L==2)).sum()); n_chip_sum=sum(int(((rd_label(c)==1)|(rd_label(c)==2)).sum()) for c in ids)
    for model,sub in models.items():
        def rd_pred(cid):
            pp=glob.glob(os.path.join(base,sub,cid+'_Classified.tif'))
            if not pp: return None
            a=tifffile.imread(pp[0]); return (a[0] if a.ndim==3 else a).astype(np.int16)
        P=mosaic(rd_pred)
        valid=((L==1)|(L==2))&(V==1)
        pb=(P==1) if model=='SamLoRA' else (P==2)
        tp=int((pb&(L==2)&valid).sum()); fp=int((pb&(L==1)&valid).sum()); fn=int((~pb&(L==2)&valid).sum()); tn=int((~pb&(L==1)&valid).sum())
        F=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
        rows.append(dict(event=ev,model=model+('[A:1=burned]' if model=='SamLoRA' else ''),n_chips=len(ids),labelled_px_unique=n_unique,labelled_px_summed_over_chips=n_chip_sum,
                         double_count_factor=round(n_chip_sum/n_unique,3),tp=tp,fp=fp,fn=fn,tn=tn,P=tp/(tp+fp) if tp+fp else np.nan,R=tp/(tp+fn) if tp+fn else np.nan,F1=F,IoU=tp/(tp+fp+fn) if tp+fp+fn else np.nan))
    print(ev,'unique labelled px',n_unique,'summed over chips',n_chip_sum,flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,'test_event_metrics_uniquepixels.csv'),index=False)
pd.set_option('display.width',220); print(d[['event','model','double_count_factor','P','R','F1','IoU']].round(3).to_string())
