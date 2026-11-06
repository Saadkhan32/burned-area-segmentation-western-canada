#!/usr/bin/env python3
"""Uncertainty and practical errors on the ARCHIVED predictions (CNN models only are valid; extension models reported but flagged).
Inputs: <AUDIT_DIR>/results/test_perchip_confusion_imgvalid.csv (per chip tp/fp/fn/tn, burned=2 positive, image-valid pixels).
Outputs: chip_bootstrap_intervals.csv (per event x model: pooled burned P/R/F1/IoU with 95 % percentile intervals from a
         2,000-replicate bootstrap over CHIPS, seed 20261012; chips overlap by 50 %, so these intervals are optimistic),
         paired_model_differences.csv (per event: pairwise differences in pooled burned F1 with bootstrap intervals, same chips),
         area_errors.csv (15A: predicted vs reference burned area on the evaluated chips, ha; signed/abs/relative error, omission/commission),
         negative_chip_fp.csv (15B: false-positive burned area on chips with no labelled burn),
         runtime_from_logs.csv (15D: durations in the archived classification_run_log.txt files).
"""
import os, glob, re, numpy as np, pandas as pd
R=os.path.join(os.environ.get('AUDIT_DIR', 'audit'), 'results')
ROOT=os.environ.get('SUBMITTED_STUDY_DIR', 'submitted_study')   # root of the archived study folder (chip exports, model files, test predictions)
d=pd.read_csv(os.path.join(R,'test_perchip_confusion_imgvalid.csv')); d=d[d.status=='OK'].copy()
d=d[~d.model.str.startswith('SamLoRA[B')]
rng=np.random.default_rng(20261012); B=2000
PIX_HA=0.01  # 10 m x 10 m = 100 m2 = 0.01 ha (chips are in projected UTM metres)
def f1(tp,fp,fn): den=2*tp+fp+fn; return 2*tp/den if den else np.nan
rows=[]; pairs=[]; areas=[]; negs=[]
for ev,g in d.groupby('event'):
    models=sorted(g.model.unique()); chips=sorted(g.chip_id.unique())
    piv={m:g[g.model==m].set_index('chip_id').reindex(chips) for m in models}
    common=[c for c in chips if all(not piv[m].loc[c,['tp','fp','fn','tn']].isna().any() for m in models)]
    n=len(common); idx=np.arange(n)
    boots={}
    for m in models:
        P=piv[m].loc[common]; tp=P.tp.values; fp=P.fp.values; fn=P.fn.values; tn=P.tn.values
        stat=np.empty(B)
        for b in range(B):
            s=rng.integers(0,n,n); stat[b]=f1(tp[s].sum(),fp[s].sum(),fn[s].sum())
        boots[m]=stat
        TP,FP,FN,TN=tp.sum(),fp.sum(),fn.sum(),tn.sum()
        rows.append(dict(event=ev,model=m,n_chips_common=n,P_burned=TP/(TP+FP) if TP+FP else np.nan,R_burned=TP/(TP+FN) if TP+FN else np.nan,
                         F1_burned=f1(TP,FP,FN),IoU_burned=TP/(TP+FP+FN) if TP+FP+FN else np.nan,
                         F1_ci_low=np.nanpercentile(stat,2.5),F1_ci_high=np.nanpercentile(stat,97.5),
                         note='chip bootstrap, 2000 reps; chips overlap 50 %; in-sample event'))
        ref_ha=(TP+FN)*PIX_HA; pred_ha=(TP+FP)*PIX_HA
        areas.append(dict(event=ev,model=m,n_chips=n,reference_burned_ha=ref_ha,predicted_burned_ha=pred_ha,signed_error_ha=(FP-FN)*PIX_HA,
                          abs_error_ha=abs(FP-FN)*PIX_HA,relative_error=((FP-FN)/(TP+FN)) if TP+FN else np.nan,omission_ha=FN*PIX_HA,commission_ha=FP*PIX_HA,
                          note='chip sample of one event window, not the whole fire'))
        Pn=P[P.n_gt_burned==0]
        negs.append(dict(event=ev,model=m,n_negative_chips=len(Pn),valid_px=int(Pn.n_valid.sum()),fp_px=int(Pn.fp.sum()),
                         fp_rate=Pn.fp.sum()/Pn.n_valid.sum() if Pn.n_valid.sum() else np.nan,fp_ha=Pn.fp.sum()*PIX_HA,
                         chips_with_any_fp=int((Pn.fp>0).sum()),note='chips chosen because they have no labelled burn (not random negative scenes)'))
    for i,m1 in enumerate(models):
        for m2 in models[i+1:]:
            diff=boots[m1]-boots[m2]
            P1=piv[m1].loc[common]; P2=piv[m2].loc[common]
            pairs.append(dict(event=ev,model_a=m1,model_b=m2,F1_a=f1(P1.tp.sum(),P1.fp.sum(),P1.fn.sum()),F1_b=f1(P2.tp.sum(),P2.fp.sum(),P2.fn.sum()),
                              diff_a_minus_b=f1(P1.tp.sum(),P1.fp.sum(),P1.fn.sum())-f1(P2.tp.sum(),P2.fp.sum(),P2.fn.sum()),
                              diff_ci_low=np.nanpercentile(diff,2.5),diff_ci_high=np.nanpercentile(diff,97.5),n_chips=n))
pd.DataFrame(rows).to_csv(os.path.join(R,'chip_bootstrap_intervals.csv'),index=False)
pd.DataFrame(pairs).to_csv(os.path.join(R,'paired_model_differences.csv'),index=False)
pd.DataFrame(areas).to_csv(os.path.join(R,'area_errors.csv'),index=False)
pd.DataFrame(negs).to_csv(os.path.join(R,'negative_chip_fp.csv'),index=False)
# runtime from logs
rt=[]
for f in glob.glob(os.path.join(ROOT,'**','classification_run_log.txt'),recursive=True):
    t=open(f,errors='replace').read()
    dur=re.search(r'Duration: (\d+):(\d+):([\d.]+)',t); n=re.search(r'\[(\d+)/(\d+)\]',t); emd=re.search(r'model_emd: (.*)',t); mode=re.search(r'processing_mode: (.*)',t)
    if dur and n:
        secs=int(dur.group(1))*3600+int(dur.group(2))*60+float(dur.group(3)); N=int(n.group(2))
        rt.append(dict(log=os.path.relpath(f,ROOT),model_emd=os.path.basename(emd.group(1)) if emd else '',processing_mode=mode.group(1) if mode else '',n_chips=N,total_s=round(secs,1),s_per_chip=round(secs/N,1),
                       note='ArcGIS ClassifyPixelsUsingDeepLearning per 256x256 chip; hardware not recorded'))
pd.DataFrame(rt).to_csv(os.path.join(R,'runtime_from_logs.csv'),index=False)
pd.set_option('display.width',250)
print(pd.DataFrame(rows)[['event','model','n_chips_common','F1_burned','F1_ci_low','F1_ci_high']].round(3).to_string())
print(pd.DataFrame(negs)[['event','model','n_negative_chips','fp_rate','fp_ha','chips_with_any_fp']].round(4).to_string())
print(pd.DataFrame(areas)[['event','model','reference_burned_ha','predicted_burned_ha','signed_error_ha','relative_error']].round(2).to_string())
print(pd.DataFrame(rt).to_string())
