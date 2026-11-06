#!/usr/bin/env python3
"""Qualitative map figure from the quick-look folder written by analysis/export_quicklook.py. 
Rows = windows, columns = false-colour image (SWIR2-NIR-Red), NBAC reference, then one error map per model.
v2 2026-10-15 - colours as in Fig. 2 (reference examples): burned dark red, unburned beige, ignored amber, masked grey;
in the error maps agreement on burn = dark red, in the reference but not mapped (omission) = light red,
mapped but not in the reference (commission) = blue. Thin panel frames. Times New Roman where installed, 600 dpi, plain hyphens.
v3 2026-10-15 - legend lists only the classes that appear in the figure; two-line title of the image column; tighter bottom margin.
v4 2026-10-22 - row labels use the agency fire name (fire_names.csv next to this script: window id -> display name, with the
evidence per entry) instead of the window identifier; the manifest's agency_fire_name remains the fallback.
v5 2026-11-05 - legend entries are the standard confusion-matrix terms, capitalised (True Positive, False Negative, False Positive,
True Negative; Ignored and Masked keep a short note); colours unchanged.
Usage: python analysis/make_maps_planA.py --quicklook quicklook/ --windows W1,W2,... --models segformer,unet --out figP8_qualitative.png
       --windows auto picks, per province, the median and the weakest window of the first model."""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures_planA import MODELS, label, BLUE, ORANGE, AQUA, INK, FONT

plt.rcParams.update({'font.family': FONT, 'font.size': 7})
PROV_NAME = {'AB': 'Alberta', 'BC': 'British Columbia', 'SK': 'Saskatchewan'}
_names_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fire_names.csv')
DISPLAY_NAME = {}
if os.path.exists(_names_csv):
    import csv
    with open(_names_csv, encoding='utf-8') as fh:
        DISPLAY_NAME = {r['window_id']: r['display_name'] for r in csv.DictReader(fh)}
TP, FP, FN, NOD, IGN = np.array([178, 24, 43]), np.array([33, 102, 172]), np.array([244, 165, 130]), np.array([150, 150, 150]), np.array([253, 192, 85])
TN = np.array([238, 236, 225])                                   # unburned in reference and map (beige, as in Fig. 2)

def classes_in(img):
    """Names of the colour classes present in an RGB panel (for the legend)."""
    flat = img.reshape(-1, 3); out = set()
    for k, c in (('TP', TP), ('FN', FN), ('FP', FP), ('TN', TN), ('IGN', IGN), ('NOD', NOD)):
        if np.any(np.all(flat == c, axis=1)): out.add(k)
    return out

def error_rgb(lab, pred, nodata, step):
    lab = lab[::step, ::step]; pred = pred[::step, ::step]; nod = nodata[::step, ::step] > 0
    out = np.empty(lab.shape + (3,), np.uint8); out[:] = TN
    out[(lab == 1) & (pred == 1)] = TP; out[(lab != 1) & (pred == 1) & (lab != 255)] = FP; out[(lab == 1) & (pred != 1)] = FN
    out[lab == 255] = IGN; out[nod | (pred == 255)] = NOD
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quicklook', required=True); ap.add_argument('--windows', default='auto'); ap.add_argument('--models', default='segformer,upernet_convnext,prithvi_lora,unet')
    ap.add_argument('--out', required=True); ap.add_argument('--step', type=int, default=3, help='pixel subsampling of the 2,000-px windows for the panels'); ap.add_argument('--title', default='')
    a = ap.parse_args(); models = a.models.split(',')
    idx = json.load(open(os.path.join(a.quicklook, 'quicklook_index.json')))
    if a.windows == 'auto':
        wins = []
        for p in ('AB', 'BC', 'SK'):
            c = sorted([(v['models'][models[0]]['F1'], w) for w, v in idx.items() if v['province'] == p and v['models'].get(models[0], {}).get('ref_ha', 0) > 0])
            if c: wins += [c[len(c) // 2][1], c[0][1]]
        wins = list(dict.fromkeys(wins))
    else:
        wins = a.windows.split(',')
    ncol = 2 + len(models); nrow = len(wins)
    figh = 7.2 / ncol * nrow * 1.06 + 0.72
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, figh), squeeze=False)
    present = set()
    for i, w in enumerate(wins):
        e = idx[w]; st = a.step
        rgb = np.asarray(Image.open(os.path.join(a.quicklook, w + '_rgb.png')))[::st, ::st]
        lab = np.asarray(Image.open(os.path.join(a.quicklook, w + '_label.png'))); nod = np.asarray(Image.open(os.path.join(a.quicklook, w + '_nodata.png')))
        axes[i, 0].imshow(rgb); l2 = lab[::st, ::st]; ref = np.empty(l2.shape + (3,), np.uint8); ref[:] = TN; ref[l2 == 1] = TP; ref[l2 == 255] = IGN; ref[nod[::st, ::st] > 0] = NOD
        axes[i, 1].imshow(ref); present |= classes_in(ref)
        name = e.get('agency_fire_name') or w; name = w if name in ('', 'nan', 'None') else name
        name = DISPLAY_NAME.get(w, name)
        prov = 'Alberta' if w == 'NEG2023_SK_01' else PROV_NAME.get(e['province'], e['province'])   # window centre west of the 110th meridian (22 Oct 2026 check)
        axes[i, 0].set_ylabel(f"{name}\n{prov}, {e['acquisition'][:10]}\n{e['burned_ha']:,.0f} ha burned" + (' (next season)' if e.get('scene_year_offset') else ''), fontsize=5.8)
        for j, m in enumerate(models):
            ax = axes[i, 2 + j]; pth = os.path.join(a.quicklook, f'{w}_pred_{m}.png')
            if not os.path.exists(pth):
                ax.text(0.5, 0.5, 'no map', ha='center', va='center', transform=ax.transAxes, fontsize=6); continue
            pred = np.asarray(Image.open(pth)); er = error_rgb(lab, pred, nod, st); ax.imshow(er); present |= classes_in(er)
            s = e['models'].get(m)
            if s and s.get('ref_ha', 0) > 0: ax.set_title(f"F1 {s['F1']:.2f}  P {s['P']:.2f}  R {s['R']:.2f}", fontsize=5.8, pad=1.5)
            elif s: ax.set_title(f"mapped as burned: {s['pred_ha']:,.0f} ha", fontsize=5.8, pad=1.5)
        if i == 0:
            axes[0, 0].set_title('Sentinel-2 scene\n(SWIR2-NIR-Red)', fontsize=6.2, pad=1.5, linespacing=1.0); axes[0, 1].set_title('NBAC reference', fontsize=6.2, pad=1.5)
        for j in range(ncol):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values(): sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor('#555555')
    for j, m in enumerate(models):
        axes[nrow - 1, 2 + j].set_xlabel(label(m), fontsize=6.2)
    entries = [('TP', TP, 'True Positive'), ('FN', FN, 'False Negative'), ('FP', FP, 'False Positive'), ('TN', TN, 'True Negative'),
               ('IGN', IGN, 'Ignored (earlier or later fire)'), ('NOD', NOD, 'Masked (cloud, shadow, snow)')]
    handles = [Patch(facecolor=c / 255, edgecolor='#555555', lw=0.4, label=t) for k, c, t in entries if k in present]
    ncl = 3 if len(handles) > 4 else len(handles); nlr = int(np.ceil(len(handles) / ncl))
    fig.legend(handles=handles, loc='lower center', ncol=ncl, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.6)
    if a.title: fig.suptitle(a.title, fontsize=7.5)
    fig.subplots_adjust(left=0.1, right=0.99, top=1 - 0.42 / figh - (0.25 / figh if a.title else 0), bottom=(0.30 + 0.15 * nlr) / figh, wspace=0.03, hspace=0.12)
    fig.savefig(a.out, dpi=600); plt.close(fig); print('wrote', a.out, 'windows:', ', '.join(wins))

if __name__ == '__main__':
    main()
