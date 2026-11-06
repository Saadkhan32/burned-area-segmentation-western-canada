#!/usr/bin/env python3
"""Recompute every archived test evaluation from the prediction rasters and labels
under one declared rule (burned = positive class; image-valid pixels only).
Outputs (<AUDIT_DIR>/results/):
  test_perchip_confusion.csv  - per chip, per model, per event: tp/fp/fn/tn, n_pred0, n_gt_burned
  test_event_metrics.csv      - pooled per event x model (+ SamLoRA under both readings)
"""
import os, sys, glob, csv, numpy as np, tifffile, pandas as pd
VARIANT = 'imgvalid' if '--imgvalid' in sys.argv else 'aslabelled'
ROOT=os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study')   # root of the archived study folder (chip exports, model files, test predictions)
OUT=os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'results'); os.makedirs(OUT,exist_ok=True)
TESTS = {  # event folder under Test Locations : {model: prediction subfolder}
 'Betts':            {'DeepLabV3':'betts pred deepv3','Mask2Former':'mmseg pred','PSPNet':'pspnet pred','SamLoRA':'samlora_pred','UNet':'unet pred'},
 'BOLDING':          {'DeepLabV3':'DEEP_PRED','Mask2Former':'mmseg_pred','PSPNet':'PSPnet_pred','SamLoRA':'samlora_pred','UNet':'unet_pred'},
 'rossmere lake':    {'DeepLabV3':'deeplab_pred','Mask2Former':'mmseg_pred','PSPNet':'pspnet_pred','SamLoRA':'sam_pred','UNet':'unet_pred'},
 'Algar Lake Complex':{'DeepLabV3':'deep_pred','Mask2Former':'mm_pred','PSPNet':'pspnet_pred','SamLoRA':'sam_predi','UNet':'unet_pred'},
 'SHAW':             {'DeepLabV3':'deep_pred','Mask2Former':'mm_pred','PSPNet':'pspnet_pred','SamLoRA':'sam_predictions','UNet':'unet_predictions'},
 'Tomais Mountain':  {'DeepLabV3':'deep_pred','Mask2Former':'mm_pred','PSPNet':'pspnet_pred','SamLoRA':'sam_predictions','UNet':'unet_predictions'},
}
TRUE_PROVINCE = {'Betts':'AB','BOLDING':'SK','rossmere lake':'BC','Algar Lake Complex':'AB','SHAW':'SK','Tomais Mountain':'BC'}
TABLE7_LABEL = {'Betts':'Alberta','BOLDING':'British Columbia (as labelled)','rossmere lake':'Saskatchewan (as labelled)'}
def load(p):
    a=np.asarray(tifffile.imread(p)); 
    if a.ndim==3: a=a[0] if a.shape[0]<=4 else a[...,0]
    return a
rows=[]
for ev, models in TESTS.items():
    lab_dir=os.path.join(ROOT,'Test Locations',ev,'labels')
    labs=sorted(glob.glob(os.path.join(lab_dir,'*.tif')))
    for model, sub in models.items():
        pdir=os.path.join(ROOT,'Test Locations',ev,sub)
        if not os.path.isdir(pdir): print('MISSING',ev,model,pdir); continue
        for lp in labs:
            cid=os.path.splitext(os.path.basename(lp))[0]
            pp=glob.glob(os.path.join(pdir,cid+'_Classified.tif')) or glob.glob(os.path.join(pdir,cid+'*.tif'))
            if not pp:
                rows.append(dict(event=ev,model=model,chip_id=cid,status='NO_PREDICTION')); continue
            g=load(lp); p=load(pp[0])
            imp=os.path.join(ROOT,'Test Locations',ev,'images',cid+'.tif')
            im=np.asarray(tifffile.imread(imp)); im=np.transpose(im,(1,2,0)) if im.shape[0]==9 else im
            imgvalid=~np.isnan(im).any(2)
            if g.shape!=p.shape:
                rows.append(dict(event=ev,model=model,chip_id=cid,status=f'SHAPE_MISMATCH {g.shape} vs {p.shape}')); continue
            valid=((g==1)|(g==2)) & (imgvalid if VARIANT=='imgvalid' else True)
            gb=(g==2)&valid; gu=(g==1)&valid
            pv=np.unique(p[valid]).tolist()
            def counts(pred_burned):
                pb=pred_burned&valid
                return int((gb&pb).sum()), int((gu&pb).sum()), int((gb&~pb).sum()), int((gu&~pb).sum())
            if model=='SamLoRA':
                tpA,fpA,fnA,tnA = counts(p==1)   # reading A: index shift, 1 = burned
                tpB,fpB,fnB,tnB = counts(p==2)   # reading B: values as declared, burned=2 (never present)
                rows.append(dict(event=ev,model='SamLoRA[A:1=burned]',chip_id=cid,status='OK',n_valid=int(valid.sum()),n_gt_burned=int(gb.sum()),
                                 n_pred0_on_valid=int(((p==0)&valid).sum()),pred_values=str(pv),tp=tpA,fp=fpA,fn=fnA,tn=tnA))
                rows.append(dict(event=ev,model='SamLoRA[B:2=burned]',chip_id=cid,status='OK',n_valid=int(valid.sum()),n_gt_burned=int(gb.sum()),
                                 n_pred0_on_valid=int(((p==0)&valid).sum()),pred_values=str(pv),tp=tpB,fp=fpB,fn=fnB,tn=tnB))
            else:
                tp,fp,fn,tn = counts(p==2)      # value 0 (no class) on a labelled pixel counts as not-burned
                rows.append(dict(event=ev,model=model,chip_id=cid,status='OK',n_valid=int(valid.sum()),n_gt_burned=int(gb.sum()),
                                 n_pred0_on_valid=int(((p==0)&valid).sum()),pred_values=str(pv),tp=tp,fp=fp,fn=fn,tn=tn))
    print('done',ev,flush=True)
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,f'test_perchip_confusion_{VARIANT}.csv'),index=False)
ok=df[df.status=='OK'].copy()
def m(tp,fp,fn,tn):
    P=tp/(tp+fp) if tp+fp else float('nan'); R=tp/(tp+fn) if tp+fn else float('nan')
    F=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float('nan'); I=tp/(tp+fp+fn) if tp+fp+fn else float('nan')
    Pu=tn/(tn+fn) if tn+fn else float('nan'); Ru=tn/(tn+fp) if tn+fp else float('nan'); Fu=2*tn/(2*tn+fn+fp) if 2*tn+fn+fp else float('nan'); Iu=tn/(tn+fn+fp) if tn+fn+fp else float('nan')
    return dict(P_burned=P,R_burned=R,F1_burned=F,IoU_burned=I,P_unburned=Pu,R_unburned=Ru,F1_unburned=Fu,IoU_unburned=Iu)
out=[]
for (ev,model),gdf in ok.groupby(['event','model']):
    tp,fp,fn,tn=[int(gdf[c].sum()) for c in ('tp','fp','fn','tn')]
    r=dict(event=ev,true_province=TRUE_PROVINCE[ev],table7_label=TABLE7_LABEL.get(ev,'(earlier test, not in paper)'),model=model,n_chips=len(gdf),
           n_chips_with_burned=int((gdf.n_gt_burned>0).sum()),gt_burned_share=round(gdf.n_gt_burned.sum()/gdf.n_valid.sum(),4),
           n_pred0_on_valid=int(gdf.n_pred0_on_valid.sum()),tp=tp,fp=fp,fn=fn,tn=tn)
    r.update({k:round(v,4) for k,v in m(tp,fp,fn,tn).items()})
    # per-chip mean burned F1 over chips that contain burned pixels
    f1s=[]
    for _,c in gdf[gdf.n_gt_burned>0].iterrows():
        d=2*c.tp+c.fp+c.fn; f1s.append(2*c.tp/d if d else np.nan)
    r['mean_chip_F1_burned(chips with burned)']=round(float(np.nanmean(f1s)),4) if f1s else float('nan')
    r['sd_chip_F1_burned']=round(float(np.nanstd(f1s,ddof=1)),4) if len(f1s)>1 else float('nan')
    out.append(r)
res=pd.DataFrame(out); res.to_csv(os.path.join(OUT,f'test_event_metrics_{VARIANT}.csv'),index=False)
pd.set_option('display.width',250); print(res[['event','true_province','model','n_chips','n_pred0_on_valid','tp','fp','fn','tn','P_burned','R_burned','F1_burned','IoU_burned','F1_unburned','mean_chip_F1_burned(chips with burned)']].to_string())
