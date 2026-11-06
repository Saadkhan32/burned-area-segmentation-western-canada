#!/usr/bin/env python3
"""Manuscript figures and tables from a finished run folder.
Input folder (copied from Narval ~/scratch/burn/runs): results/<run>_test_windows.csv, summary/*.csv (summarize.py),
manifest.csv (dataset), run_ledger.csv and <run>/{config.json,ledger.json,log.csv}. Every number in the outputs comes from these files.
Style: Times New Roman where installed (metric-compatible Liberation Serif otherwise), 600 dpi PNG, plain hyphens,
distinct colour-blind-safe hues (checked for the common colour-vision deficiencies): model families vermilion / purple / blue
(#D55E00, #AA4499, #3D9AD1; all pairs pass), label-free rules neutral grey, loss-contrast runs hollow markers of the family colour;
provinces as in the Fig. 1 map, BC #0072B2 / AB #E69F00 / SK #009E73 (all pairs pass); plus an editable PowerPoint with native charts.
v2 2026-10-15: broken F1 axis and Holm letter groups (P1, P2), horizontal per-window and fire-free plots (P3, P6), upper-triangle
pairwise matrix (P4b), dataset distributions moved to the supplement (figS1; the study-area map is built in ArcGIS Pro).
v3 2026-10-22: model families named by architecture (convolutional encoder-decoders / transformer networks / Earth-observation
foundation models; the objective contrast keeps hollow markers), P5 shows the three best-scoring learned configurations, table P2
labels follow the revised Section 2.8; every number is unchanged from v2 (same inputs, same statistics).
Usage: python analysis/make_figures_planA.py --runs drac_runs --out figures [--maps] [--tag 2026-10-14]
"""
import os, sys, json, glob, argparse, itertools, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarize import boot_ci, sign_flip_p, holm   # same statistics as the summary tables

# ---------------------------------------------------------------- style
for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', '*.ttf')):   # Times New Roman .ttf files dropped next to the script
    font_manager.fontManager.addfont(_f)
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = 'Times New Roman' if 'Times New Roman' in fams else ('Liberation Serif' if 'Liberation Serif' in fams else 'serif')
plt.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#9a9a9a', 'axes.linewidth': 0.6, 'ytick.color': '#444444', 'xtick.color': '#444444',
                     'mathtext.fontset': 'stix', 'savefig.dpi': 600, 'figure.dpi': 110})
DPI = 600
GREY, INK = '#6d6d6d', '#222222'
VERMILION, PURPLE, FBLUE = '#D55E00', '#AA4499', '#3D9AD1'            # model families
P_BC, P_AB, P_SK = '#0072B2', '#E69F00', '#009E73'                    # provinces, as in the Fig. 1 map
BLUE, ORANGE, AQUA = P_BC, P_AB, P_SK                                 # names kept for make_maps_planA.py
# model key -> (label, family), families as in Table 4 of the manuscript: cnn (convolutional encoder-decoders), transformer
# (transformer networks), foundation (Earth-observation foundation models), loss (objective contrast: the three ResNet-50
# encoder-decoders trained with plain cross-entropy), baseline (label-free index rules)
MODELS = {'unet': ('U-Net (ResNet-50)', 'cnn'), 'deeplabv3': ('DeepLabV3 (ResNet-50)', 'cnn'), 'pspnet': ('PSPNet (ResNet-50)', 'cnn'),
          'deeplabv3plus': ('DeepLabV3+ (ResNet-50)', 'cnn'), 'upernet_convnext': ('UPerNet (ConvNeXt-T)', 'cnn'),
          'samlora': ('SAM-LoRA (ViT-B)', 'transformer'), 'mask2former': ('Mask2Former (Swin-L)', 'transformer'), 'segformer': ('SegFormer (MiT-B2)', 'transformer'),
          'unet_C2': ('U-Net, plain CE', 'loss'), 'deeplabv3_C2': ('DeepLabV3, plain CE', 'loss'), 'pspnet_C2': ('PSPNet, plain CE', 'loss'),
          'prithvi': ('Prithvi-EO-2.0, full fine-tuning', 'foundation'), 'prithvi_lora': ('Prithvi-EO-2.0, LoRA', 'foundation'), 'terramind': ('TerraMind, full fine-tuning', 'foundation'),
          'dinov3_lora': ('DINOv3 SAT-493M, LoRA', 'foundation'), 'NBR<0': ('NBR < 0 rule', 'baseline'), 'Otsu(NBR)': ('Otsu threshold on NBR', 'baseline'),
          'dNBR>0.1': ('dNBR > 0.1 rule (pre-fire scene)', 'baseline'), 'Otsu(dNBR)': ('Otsu threshold on dNBR (pre-fire scene)', 'baseline')}
FAMILY = {'cnn': ('Convolutional encoder-decoders', VERMILION), 'loss': ('Objective contrast: plain cross-entropy', VERMILION), 'transformer': ('Transformer networks', PURPLE),
          'foundation': ('Earth-observation foundation models', FBLUE), 'baseline': ('Label-free index rules', GREY)}
FAMILY_ORDER = ('cnn', 'loss', 'transformer', 'foundation', 'baseline')                 # legend order
PROV = {'AB': ('Alberta', P_AB), 'BC': ('British Columbia', P_BC), 'SK': ('Saskatchewan', P_SK)}
CNN_SET = ['unet', 'deeplabv3', 'pspnet']                                   # protocol: mean of the three ResNet-50 encoder-decoders ...
TRANSFORMER_SET = ['segformer', 'mask2former', 'samlora', 'prithvi', 'prithvi_lora', 'terramind', 'dinov3_lora']   # ... vs mean of the transformer-based models
RQ2_SET = ['deeplabv3plus', 'segformer', 'upernet_convnext', 'prithvi', 'prithvi_lora', 'terramind']   # Section 2.8: compared with the best-scoring configuration (RQ2)

def label(m): return MODELS.get(m, (m, 'other'))[0]
def family(m): return MODELS.get(m, (m, 'other'))[1]
def colour(m): return FAMILY.get(family(m), ('', INK))[1]
def hollow(m): return family(m) == 'loss'
def fmt_p(p):
    return 'n/a' if p != p else ('< 0.001' if p < 0.001 else f'{p:.3f}')

def family_legend(ax, families, loc='lower right', **kw):
    handles = []
    for f in families:
        name, col = FAMILY[f]
        handles.append(Line2D([], [], marker='o', color=col, markerfacecolor='white' if f == 'loss' else col, markeredgecolor=col, lw=0, markersize=5, label=name))
    ax.legend(handles=handles, loc=loc, frameon=False, **kw)

def holm_letters(pairs, models):
    """Compact letter display (insert-absorb, Piepho 2004) from the Holm-adjusted pairwise sign-flip tests: two models share a
    letter if and only if their difference is not significant (p_holm >= 0.05). Verified after construction."""
    sig = {}
    for _, r in pairs.iterrows():
        if r.model_a in models and r.model_b in models:
            sig[frozenset((r.model_a, r.model_b))] = bool(r.p_holm < 0.05)
    idx = {m: i for i, m in enumerate(models)}; cols = [set(range(len(models)))]
    for a, b in itertools.combinations(models, 2):
        if not sig.get(frozenset((a, b)), False):
            continue
        i, j = idx[a], idx[b]; new = []
        for c in cols:
            new += [c - {i}, c - {j}] if (i in c and j in c) else [c]
        cols = [c for k, c in enumerate(new) if c and not any((c < d) or (c == d and l < k) for l, d in enumerate(new) if l != k)]
    cols = sorted(cols, key=lambda c: (min(c), -len(c)))
    out = {m: '' for m in models}
    for L, c in zip('abcdefghijklmnopqrstuvwxyz', cols):
        for k in sorted(c):
            out[models[k]] += L
    for a, b in itertools.combinations(models, 2):
        assert bool(set(out[a]) & set(out[b])) != sig.get(frozenset((a, b)), False), ('letter display inconsistent', a, b)
    return out

def broken_pair(fig, n, xl, xr, left=0.30, right=0.90, top=0.985, bottom=0.11, ratio=(1.0, 2.35)):
    """Two x-ranges side by side with a shared y (rows); break marks on the bottom spine."""
    gs = fig.add_gridspec(1, 2, width_ratios=ratio, wspace=0.05, left=left, right=right, top=top, bottom=bottom)
    aL = fig.add_subplot(gs[0]); aR = fig.add_subplot(gs[1], sharey=aL)
    aL.set_xlim(*xl); aR.set_xlim(*xr)
    aL.spines['right'].set_visible(False); aR.spines['left'].set_visible(False)
    aR.tick_params(axis='y', length=0); plt.setp(aR.get_yticklabels(), visible=False)
    for ax in (aL, aR):
        ax.xaxis.grid(True, color='#e9e9e9', lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis='x', length=2)
    mk = dict(marker=[(-1, -1.6), (1, 1.6)], markersize=5, linestyle='none', color='#8a8a8a', mec='#8a8a8a', mew=0.7, clip_on=False)
    aL.plot([1], [0], transform=aL.transAxes, **mk); aR.plot([0], [0], transform=aR.transAxes, **mk)
    return aL, aR

def thousands(x, pos=None):
    return f'{x:,.0f}'

# ---------------------------------------------------------------- load
def load(runs):
    S = os.path.join(runs, 'summary')
    summ = pd.read_csv(os.path.join(S, 'summary_by_model.csv'))
    allw = pd.read_csv(os.path.join(S, 'all_windows.csv'))
    pairs = pd.read_csv(os.path.join(S, 'paired_differences.csv'))
    negs = pd.read_csv(os.path.join(S, 'negative_windows_by_model.csv')) if os.path.exists(os.path.join(S, 'negative_windows_by_model.csv')) else pd.DataFrame()
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(runs, 'results', '*_test_windows.csv')))], ignore_index=True)
    raw = raw.drop_duplicates(subset=['model', 'window_id'], keep='last')
    man = pd.read_csv(os.path.join(runs, 'manifest.csv'))
    man = man[man.status == 'OK'].copy()
    for c in ('is_negative',):
        man[c] = man[c].astype(str).str.lower().isin(('true', '1'))
    return summ, allw, pairs, negs, raw, man

def model_order(summ):
    """Learned models sorted by mean F1 (descending), then the two baselines."""
    srt = list(summ.sort_values('mean_F1', ascending=False).model)
    return [m for m in srt if family(m) != 'baseline'] + [m for m in srt if family(m) == 'baseline']

# ---------------------------------------------------------------- figures
def fig_model_comparison(summ, out, pairs=None):
    """P1: mean per-window F1 with window-bootstrap 95 % interval and median (tick), sorted; family colours; broken F1 axis
    (label-free rules left, learned models right); Holm letter groups of the learned models in the right margin."""
    order = model_order(summ)[::-1]
    s = summ.set_index('model').loc[order]
    learned = [m for m in model_order(summ) if family(m) != 'baseline']
    letters = holm_letters(pairs, learned) if pairs is not None else {}
    n = len(order); figh = 0.255 * n + 1.05
    fig = plt.figure(figsize=(6.5, figh))
    aL, aR = broken_pair(fig, n, (0.58, 0.81), (0.82, 0.955), right=0.855, bottom=0.62 / figh, top=1 - 0.2 / figh)
    y = np.arange(n)
    for ax in (aL, aR):
        for yi, m in zip(y, order):
            r = s.loc[m]; col = colour(m)
            ax.plot([r.F1_ci_lo, r.F1_ci_hi], [yi, yi], color=col, lw=1.5, solid_capstyle='butt', zorder=2)
            ax.plot(r.median_F1, yi, marker='|', ms=7, color=col, markeredgewidth=1.0, zorder=3, alpha=0.9)
            ax.plot(r.mean_F1, yi, marker='o', ms=5.5, color=col, markerfacecolor='white' if hollow(m) else col, markeredgewidth=1.2, zorder=4)
    bt = matplotlib.transforms.blended_transform_factory(aR.transAxes, aR.transData)
    for yi, m in zip(y, order):
        r = s.loc[m]
        aR.text(1.03, yi, f'{r.mean_F1:.3f}', transform=bt, va='center', ha='left', fontsize=6.8, color=INK)
        if m in letters:
            aR.text(1.21, yi, letters[m], transform=bt, va='center', ha='left', fontsize=6.8, color=INK)
    aR.text(1.03, n - 0.45, 'Mean', transform=bt, va='bottom', ha='left', fontsize=6.5, color='#555555', style='italic')
    if letters:
        aR.text(1.21, n - 0.45, 'Group', transform=bt, va='bottom', ha='left', fontsize=6.5, color='#555555', style='italic')
    aL.set_yticks(y); aL.set_yticklabels([label(m) for m in order]); aL.set_ylim(-0.6, n - 0.4)
    aL.set_xticks([0.6, 0.65, 0.7, 0.75, 0.8]); aR.set_xticks(np.round(np.arange(0.84, 0.961, 0.02), 2))
    nb = sum(family(m) == 'baseline' for m in order)
    for ax in (aL, aR):
        ax.axhline(nb - 0.5, color='#bbbbbb', lw=0.5, ls=(0, (2, 2)))
    fams = [f for f in FAMILY_ORDER if any(family(m) == f for m in order)]
    hs = [Line2D([], [], marker='o', color=FAMILY[f][1], markerfacecolor='white' if f == 'loss' else FAMILY[f][1], markeredgecolor=FAMILY[f][1], lw=0, markersize=5, label=FAMILY[f][0]) for f in fams]
    pL = aL.get_position()
    lg = fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(pL.x0 + 0.004, pL.y1 - 0.004), frameon=True, fontsize=6.5, borderaxespad=0.2, handletextpad=0.3, borderpad=0.4)
    lg.get_frame().set_facecolor('white'); lg.get_frame().set_edgecolor('none'); lg.get_frame().set_alpha(0.92)
    x0, x1 = aL.get_position().x0, aR.get_position().x1
    fig.text((x0 + x1) / 2, 0.16 / figh, 'Mean per-window burned-class F1 on the held-out province (dot: mean; line: 95 % bootstrap interval; tick: median)', ha='center', va='bottom', fontsize=7.5)
    fig.savefig(os.path.join(out, 'figP1_model_comparison.png'), dpi=DPI); plt.close(fig)
    return letters

def fig_province(summ, out):
    """P2: mean F1 by held-out province (province colours) for every model, overall mean as a black tick; broken F1 axis."""
    order = model_order(summ)[::-1]
    s = summ.set_index('model').loc[order]
    n = len(order); figh = 0.255 * n + 1.05
    fig = plt.figure(figsize=(6.5, figh))
    aL, aR = broken_pair(fig, n, (0.58, 0.81), (0.82, 0.965), right=0.97, bottom=0.62 / figh, top=1 - 0.12 / figh)
    y = np.arange(n)
    for ax in (aL, aR):
        for yi, m in zip(y, order):
            r = s.loc[m]
            vals = [r.get(f'mean_F1_{p}', np.nan) for p in PROV]
            ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color='#cfcfcf', lw=1.2, zorder=1, solid_capstyle='round')
            for (p, (pname, col)), v in zip(PROV.items(), vals):
                ax.plot(v, yi, marker='o', ms=5, color=col, markerfacecolor='white' if hollow(m) else col, markeredgewidth=1.1, zorder=3)
            ax.plot(r.mean_F1, yi, marker='|', ms=9, color=INK, markeredgewidth=1.3, zorder=4)
    aL.set_yticks(y); aL.set_yticklabels([label(m) for m in order]); aL.set_ylim(-0.6, n - 0.4)
    aL.set_xticks([0.6, 0.65, 0.7, 0.75, 0.8]); aR.set_xticks(np.round(np.arange(0.84, 0.961, 0.02), 2))
    nb = sum(family(m) == 'baseline' for m in order)
    for ax in (aL, aR):
        ax.axhline(nb - 0.5, color='#bbbbbb', lw=0.5, ls=(0, (2, 2)))
    handles = [Line2D([], [], marker='o', color=col, lw=0, markersize=5, label=f'{pname} held out') for p, (pname, col) in PROV.items()]
    handles.append(Line2D([], [], marker='|', color=INK, lw=0, markersize=8, markeredgewidth=1.2, label='All 120 fire windows'))
    handles.append(Line2D([], [], marker='o', color=GREY, markerfacecolor='white', lw=0, markersize=5, label='hollow: plain cross-entropy'))
    pL = aL.get_position()
    lg = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(pL.x0 + 0.004, pL.y1 - 0.004), frameon=True, fontsize=6.5, borderaxespad=0.2, handletextpad=0.3, borderpad=0.4)
    lg.get_frame().set_facecolor('white'); lg.get_frame().set_edgecolor('none'); lg.get_frame().set_alpha(0.92)
    x0, x1 = aL.get_position().x0, aR.get_position().x1
    fig.text((x0 + x1) / 2, 0.16 / figh, 'Mean per-window burned-class F1 of the 40 fire windows of each held-out province', ha='center', va='bottom', fontsize=7.5)
    fig.savefig(os.path.join(out, 'figP2_province.png'), dpi=DPI); plt.close(fig)

def fig_window_distribution(allw, summ, out):
    """P3: per-window F1 distribution per model, horizontal (box = quartiles, whiskers = 1.5 IQR, dots = windows by province)."""
    order = model_order(summ)[::-1]
    n = len(order); figh = 0.255 * n + 0.95
    fig, ax = plt.subplots(figsize=(6.5, figh))
    rng = np.random.default_rng(1)
    data = [allw[allw.model == m].F1.values for m in order]
    ax.boxplot(data, positions=range(n), orientation='horizontal', widths=0.56, showfliers=False, patch_artist=True, zorder=2,
               medianprops=dict(color=INK, lw=1.1), boxprops=dict(facecolor='white', edgecolor='#8a8a8a', lw=0.6),
               whiskerprops=dict(color='#8a8a8a', lw=0.6), capprops=dict(color='#8a8a8a', lw=0.6))
    for i, m in enumerate(order):
        g = allw[allw.model == m]
        for p, (pname, col) in PROV.items():
            v = g[g.province == p].F1.values
            ax.plot(v, i + rng.uniform(-0.2, 0.2, len(v)), marker='o', ms=1.8, lw=0, color=col, alpha=0.7, zorder=3, markeredgewidth=0)
    ax.set_yticks(range(n)); ax.set_yticklabels([label(m) for m in order]); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.01, 1.01); ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel('Per-window burned-class F1 of the 120 fire windows (box: quartiles; whiskers: 1.5 x IQR; dots: windows)')
    ax.xaxis.grid(True, color='#e9e9e9', lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)
    nb = sum(family(m) == 'baseline' for m in order); ax.axhline(nb - 0.5, color='#bbbbbb', lw=0.5, ls=(0, (2, 2)))
    handles = [Line2D([], [], marker='o', color=col, lw=0, markersize=4, label=pname) for p, (pname, col) in PROV.items()]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=6.5, borderaxespad=0.2, handletextpad=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figP3_window_distribution.png'), dpi=DPI); plt.close(fig)

def fig_paired(pairs, summ, out, ref='NBR<0'):
    """P4: (a) every learned model against the NBR < 0 rule; (b) matrix of mean per-window F1 differences between learned models
    (row minus column), Holm-significant cells marked."""
    order = [m for m in model_order(summ) if family(m) != 'baseline']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw=dict(width_ratios=[0.92, 1.6]))
    p = pairs[(pairs.model_b == ref) & (pairs.model_a.isin(order))].set_index('model_a').loc[[m for m in order if m in set(pairs.model_a)]]
    ref2 = 'dNBR>0.1' if 'dNBR>0.1' in set(pairs.model_b) else None   # the more-information baseline (pre-fire scene), when scored
    p2 = pairs[(pairs.model_b == ref2) & (pairs.model_a.isin(order))].set_index('model_a') if ref2 else None
    y = np.arange(len(p))[::-1]
    for yi, (m, r) in zip(y, p.iterrows()):
        col = colour(m); dy = 0.18 if ref2 else 0.0
        a1.plot([r.diff_ci_lo, r.diff_ci_hi], [yi + dy, yi + dy], color=col, lw=1.4); a1.plot(r.mean_diff_F1, yi + dy, 'o', ms=4.5, color=col, markerfacecolor='white' if hollow(m) else col, markeredgewidth=1.2)
        a1.text(r.diff_ci_hi + 0.006, yi + dy, f'{int(r.wins_a)}/{int(r.n)}', va='center', fontsize=5.5, color=INK)
        if ref2 and m in p2.index:
            r2 = p2.loc[m]
            a1.plot([r2.diff_ci_lo, r2.diff_ci_hi], [yi - dy, yi - dy], color=GREY, lw=1.0); a1.plot(r2.mean_diff_F1, yi - dy, 'D', ms=3.2, color=GREY, markerfacecolor='white', markeredgewidth=1.0)
            a1.text(r2.diff_ci_hi + 0.006, yi - dy, f'{int(r2.wins_a)}/{int(r2.n)}', va='center', fontsize=5.5, color=GREY)
    a1.set_yticks(y); a1.set_yticklabels([label(m) for m in p.index], fontsize=6.5); a1.axvline(0, color='#999999', lw=0.6)
    a1.set_xlabel('Mean per-window F1 difference to the label-free rule\n(95 % bootstrap interval; windows won / n)'); a1.xaxis.grid(True, color='#e6e6e6', lw=0.5); a1.set_axisbelow(True)
    a1.set_title('(a) Learned models against the label-free rules', loc='left', fontsize=8)
    if ref2:
        handles = [Line2D([], [], marker='o', color=INK, lw=0, markersize=4, label='vs NBR < 0 (post-fire scene only)'),
                   Line2D([], [], marker='D', color=GREY, markerfacecolor='white', lw=0, markersize=3.5, label='vs dNBR > 0.1 (pre- and post-fire scenes)')]
        a1.set_xlim(left=min(-0.02, a1.get_xlim()[0])); a1.set_ylim(-1.6, len(p) - 0.4)
        a1.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, -0.005), frameon=False, fontsize=5.6, handletextpad=0.4, borderaxespad=0.2)
    n = len(order); M = np.full((n, n), np.nan); sig = np.zeros((n, n), bool)
    look = {(r.model_a, r.model_b): r for _, r in pairs.iterrows()}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if a == b: continue
            r = look.get((a, b)); s = 1
            if r is None: r = look.get((b, a)); s = -1
            if r is None: continue
            M[i, j] = s * r.mean_diff_F1; sig[i, j] = r.p_holm < 0.05
    # lower triangle without the empty first row / last column: cell (row r, column c) = F1 of the column model minus F1 of the
    # row model (columns rank higher), outlined when the Holm-adjusted sign-flip p < 0.05
    rows_, cols_ = order[1:], order[:-1]; nr, nc = len(rows_), len(cols_)
    L = np.full((nr, nc), np.nan); S = np.zeros((nr, nc), bool)
    for ii, r_ in enumerate(rows_):
        for jj, c_ in enumerate(cols_):
            i0, j0 = order.index(r_), order.index(c_)
            if j0 < i0:
                L[ii, jj] = -M[i0, j0]; S[ii, jj] = sig[i0, j0]
    vmax = max(0.01, np.nanmax(L)) if np.isfinite(L).any() else 0.05
    seq = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#fbfbfb', '#d7e4f1', '#94b9dc', '#4f86c0', '#1f4e8c'])
    im = a2.imshow(np.clip(L, 0, None), cmap=seq, vmin=0, vmax=vmax)
    for ii in range(nr):
        for jj in range(nc):
            if np.isfinite(L[ii, jj]):
                v = L[ii, jj]; dark = v > 0.6 * vmax
                a2.text(jj, ii, f'{v:.3f}', ha='center', va='center', fontsize=4.9, color='white' if dark else INK, fontweight='bold' if S[ii, jj] else 'normal')
                if S[ii, jj]:
                    a2.add_patch(matplotlib.patches.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.7, zorder=3))
    for sp in a2.spines.values(): sp.set_visible(False)
    a2.set_xticks(range(nc)); a2.set_xticklabels([label(m) for m in cols_], rotation=55, ha='right', fontsize=5.8)
    a2.set_yticks(range(nr)); a2.set_yticklabels([label(m) for m in rows_], fontsize=5.8)
    a2.tick_params(length=0)
    a2.set_title('(b) Pairwise mean F1 differences, column minus row\n(outlined and bold: Holm-adjusted sign-flip p < 0.05)', loc='left', fontsize=8)
    cb = fig.colorbar(im, ax=a2, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label('F1 difference (column minus row)', fontsize=7); cb.outline.set_visible(False)
    for a in (a1, a2):
        a.tick_params(length=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figP4_paired_differences.png'), dpi=DPI); plt.close(fig)

def fig_area(allw, summ, out, models=None):
    """P5: mapped against reference burned area per window (log-log) for the three best-scoring learned configurations
    (by mean F1) and the index rules."""
    if models is None:
        pick = [m for m in model_order(summ) if family(m) != 'baseline'][:3]
        models = pick + (['dNBR>0.1'] if 'dNBR>0.1' in set(summ.model) else []) + ['NBR<0']
    short = {'dNBR>0.1': 'dNBR > 0.1 rule (pre-fire scene)', 'Otsu(dNBR)': 'Otsu on dNBR (pre-fire scene)', 'NBR<0': 'NBR < 0 rule (post-fire scene)', 'Otsu(NBR)': 'Otsu on NBR (post-fire scene)'}
    n = len(models); fig, axes = plt.subplots(1, n, figsize=(7.2, 2.2), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        g = allw[allw.model == m]
        lim = (max(100, g.ref_ha.min() * 0.6), g.ref_ha.max() * 1.6)
        ax.plot(lim, lim, color='#999999', lw=0.6, zorder=1)
        for p, (pname, col) in PROV.items():
            gg = g[g.province == p]; ax.plot(gg.ref_ha, gg.pred_ha.clip(lower=lim[0]), 'o', ms=2.4, color=col, alpha=0.8, lw=0, label=pname)
        rel = (g.pred_ha - g.ref_ha) / g.ref_ha
        ax.set_title(f'{short.get(m, label(m))}\nmedian |error| {np.median((g.pred_ha - g.ref_ha).abs()):,.0f} ha\nmedian bias {np.median(rel) * 100:+.1f} %', fontsize=6.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.FixedLocator([1000, 3000, 10000, 30000])); axis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousands))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(length=2, labelsize=6.2); ax.set_aspect('equal', adjustable='box')
        for lab in ax.get_xticklabels(): lab.set_rotation(35); lab.set_ha('right')
    np.atleast_1d(axes)[0].set_ylabel('Mapped burned area (ha)')
    np.atleast_1d(axes)[0].legend(loc='upper left', frameon=False, fontsize=6, borderaxespad=0.1, handletextpad=0.2, markerscale=1.3)
    fig.subplots_adjust(left=0.075, right=0.995, top=1 - 0.44 / 2.2, bottom=0.5 / 2.2, wspace=0.13)
    fig.text(0.535, 0.012, 'NBAC burned area in the window (ha)', ha='center', va='bottom', fontsize=7.5)
    fig.savefig(os.path.join(out, 'figP5_area_agreement.png'), dpi=DPI); plt.close(fig)

def fig_negatives(raw, summ, out):
    """P6: burned area mapped in the 15 fire-free windows (no burn in the reference), per model: bar = mean, dots = windows."""
    neg = raw[raw.ref_ha <= 0]
    if neg.empty: return
    order = model_order(summ)[::-1]; n = len(order); figh = 0.255 * n + 1.0
    fig, ax = plt.subplots(figsize=(6.5, figh))
    rng = np.random.default_rng(2)
    xmax = max(neg.pred_ha.max(), 100)
    for i, m in enumerate(order):
        g = neg[neg.model == m]; col = colour(m); mean = g.pred_ha.mean()
        ax.barh(i, max(mean, 1) - 0.9, left=0.9, height=0.62, color=col, alpha=0.30 if not hollow(m) else 0.14, edgecolor=col, lw=0.8, zorder=2)
        v = g.pred_ha.values
        ax.plot(np.maximum(v, 1), i + rng.uniform(-0.2, 0.2, len(v)), 'o', ms=2.3, color='#333333', alpha=0.65, lw=0, zorder=3, markeredgewidth=0)
        ax.text(max(v.max(), 1) * 1.7, i, f'{mean:,.0f} ha', va='center', ha='left', fontsize=6.3, color=INK)
    ax.set_xscale('log'); ax.set_xlim(0.9, 400000)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 10, 100, 1000, 10000, 100000])); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousands))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(n)); ax.set_yticklabels([label(m) for m in order]); ax.set_ylim(-0.6, n + 0.35)
    ax.set_xlabel('Burned area mapped in a fire-free window (ha; windows with 0 ha are drawn at 1 ha)')
    ax.xaxis.grid(True, color='#e9e9e9', lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)
    nb = sum(family(m) == 'baseline' for m in order); ax.axhline(nb - 0.5, color='#bbbbbb', lw=0.5, ls=(0, (2, 2)))
    handles = [Patch(facecolor='#bdbdbd', edgecolor='#6d6d6d', label='mean of the 15 fire-free windows (bar, value)'),
               Line2D([], [], marker='o', color='#333333', alpha=0.65, lw=0, markersize=3.5, label='single fire-free window')]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=6.5, borderaxespad=0.1, ncol=2, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figP6_fire_free_windows.png'), dpi=DPI); plt.close(fig)

def covariates(man):
    """Per-window covariates from the manifest for the exploratory analysis (Supplementary Fig. S2)."""
    m = man.set_index('window_id')
    acq = pd.to_datetime(m.acquisition.str[:10], errors='coerce'); end = pd.to_datetime(m.nbac_edate, errors='coerce')
    cov = pd.DataFrame(dict(burned_fraction=m.burned_ha / (m.width * m.height / 100.0), days_after_fire=(acq - end).dt.days,
                            clear_fraction=m.clear_fraction, scene_year_offset=m.scene_year_offset, masked_fraction=m.masked_fraction, burned_ha=m.burned_ha))
    return cov

def fig_covariates(allw, summ, man, out, model=None):
    """P7 (exploratory): per-window F1 of the best model against burned fraction, days after the fire, clear fraction, next-season scene."""
    if model is None: model = [m for m in model_order(summ) if family(m) != 'baseline'][0]
    g = allw[allw.model == model].set_index('window_id'); cov = covariates(man).loc[g.index]
    from scipy.stats import spearmanr
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    specs = [('burned_fraction', 'Burned fraction of the window', False), ('days_after_fire', 'Days from last NBAC fire date to scene', False),
             ('clear_fraction', 'Clear fraction of the scene', False), ('burned_ha', 'NBAC burned area in window (ha)', True)]
    for ax, (c, xl, logx) in zip(axes, specs):
        x = cov[c].values; y = g.F1.values
        for p, (pname, col) in PROV.items():
            sel = (g.province == p).values; ax.plot(x[sel], y[sel], 'o', ms=2.4, color=col, lw=0, alpha=0.85, label=pname)
        ok = np.isfinite(x) & np.isfinite(y)
        rho, pv = spearmanr(x[ok], y[ok]) if ok.sum() > 3 else (np.nan, np.nan)
        ax.set_title(f'Spearman rho = {rho:+.2f}, p = {fmt_p(pv)}', fontsize=7)
        ax.set_xlabel(xl, fontsize=7); ax.tick_params(length=2)
        if logx:
            ax.set_xscale('log'); ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2000, 5000, 10000, 20000]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousands)); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    nxt = cov.scene_year_offset.values > 0
    note = f'Next-season scenes: mean F1 {g.F1.values[nxt].mean():.3f} (n = {nxt.sum()}); same-season scenes: {g.F1.values[~nxt].mean():.3f} (n = {(~nxt).sum()}).' if nxt.any() and (~nxt).any() else ''
    axes[0].set_ylabel(f'Per-window F1, {label(model)}', fontsize=7); axes[0].set_ylim(-0.02, 1.02)
    handles = [Line2D([], [], marker='o', color=col, lw=0, markersize=4, label=pname) for p, (pname, col) in PROV.items()]
    axes[0].legend(handles=handles, loc='lower right', frameon=False, fontsize=6.3, borderaxespad=0.2, handletextpad=0.2)
    for ax, lab_ in zip(axes, '(a) (b) (c) (d)'.split()):
        ax.text(-0.02, 1.13, lab_, transform=ax.transAxes, fontsize=7.5, fontweight='bold', ha='left', va='bottom')
    if note: fig.text(0.5, 0.012, note, ha='center', va='bottom', fontsize=6.5, color='#444444')
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(out, 'figP7_covariates.png'), dpi=DPI); plt.close(fig)
    return model

def fig_dataset(man, out):
    """S1 (supplement): acquisition lag of the same-season scenes and burned area of the fire windows. The study-area map
    (formerly panel (a) of figP0) is Fig. 1, built in ArcGIS Pro."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    cov = covariates(man); fire = man[~man.is_negative].set_index('window_id')
    d = cov.loc[fire.index, 'days_after_fire'].dropna(); same = d[fire.loc[d.index, 'scene_year_offset'] == 0]
    a1.hist(same, bins=np.arange(0, 141, 7), color='#9fb7cf', edgecolor='white', lw=0.6)
    a1.axvline(same.median(), color=INK, lw=0.8, ls=(0, (3, 2)))
    a1.text(same.median() + 2, a1.get_ylim()[1] * 0.93, f'median {same.median():.0f} days', fontsize=6.5, va='top', ha='left', color=INK)
    a1.set_xlabel('Days from the last NBAC fire date to the scene'); a1.set_ylabel('Fire windows')
    a1.set_title(f'(a) {len(same)} same-season scenes ({int((fire.scene_year_offset > 0).sum())} next-season scenes not shown)', fontsize=7, loc='left')
    a2.hist(fire.burned_ha.clip(lower=1), bins=np.logspace(np.log10(1000), np.log10(50000), 21), color='#d6b38a', edgecolor='white', lw=0.6)
    a2.set_xscale('log'); a2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1000, 2000, 5000, 10000, 20000, 50000]))
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousands)); a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.axvline(fire.burned_ha.median(), color=INK, lw=0.8, ls=(0, (3, 2)))
    a2.text(fire.burned_ha.median() * 1.05, a2.get_ylim()[1] * 0.93, f'median {fire.burned_ha.median():,.0f} ha', fontsize=6.5, va='top', ha='left', color=INK)
    a2.set_xlabel('NBAC burned area in the window (ha, log scale)')
    a2.set_title(f'(b) 120 fire windows (range {fire.burned_ha.min():,.0f}-{fire.burned_ha.max():,.0f} ha)', fontsize=7, loc='left')
    for ax in (a1, a2):
        ax.tick_params(length=2); ax.yaxis.grid(True, color='#e9e9e9', lw=0.5); ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figS1_dataset_distributions.png'), dpi=DPI); plt.close(fig)

# ---------------------------------------------------------------- tables
def table_summary(summ, pairs, out):
    order = model_order(summ); s = summ.set_index('model').loc[order]
    rows = []
    for m, r in s.iterrows():
        pn = pairs[(pairs.model_a == m) & (pairs.model_b == 'NBR<0')]
        rows.append({'Model': label(m), 'Family': FAMILY.get(family(m), ('', ''))[0], 'Windows': int(r.n_windows), 'Mean F1': round(r.mean_F1, 3),
                     '95 % CI': f'{r.F1_ci_lo:.3f}-{r.F1_ci_hi:.3f}', 'Median F1': round(r.median_F1, 3), 'Pooled F1': round(r.pooled_F1, 3),
                     'Precision': round(r.mean_P, 3), 'Recall': round(r.mean_R, 3), 'IoU': round(r.mean_IoU, 3),
                     'F1 AB': round(r.get('mean_F1_AB', np.nan), 3), 'F1 BC': round(r.get('mean_F1_BC', np.nan), 3), 'F1 SK': round(r.get('mean_F1_SK', np.nan), 3),
                     'Median |area error| (ha)': round(r.median_abs_ha_err, 0), 'Median relative area error (%)': round(100 * r.median_rel_ha_err, 1),
                     'Gain over NBR < 0 (F1)': round(float(pn.mean_diff_F1.iloc[0]), 3) if len(pn) else '', 'p (Holm)': fmt_p(float(pn.p_holm.iloc[0])) if len(pn) else ''})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out, 'tableP1_summary_by_model.csv'), index=False)
    open(os.path.join(out, 'tableP1_summary_by_model.md'), 'w').write(t.to_markdown(index=False) + '\n')
    return t

def table_planned(allw, pairs, summ, out):
    """Section 2.8 comparisons: every learned configuration vs NBR < 0, PSPNet vs DeepLabV3 (planned), ResNet-50 encoder-decoder
    mean vs transformer mean (planned, computed here per window), the objective contrast, and the RQ2 configurations vs the
    best-scoring configuration; sign-flip tests, Holm within the table."""
    wide = allw.pivot(index='window_id', columns='model', values='F1'); rows = []
    def add(name, a, b, d):
        lo, hi = boot_ci(d); p = sign_flip_p(d)
        rows.append({'Comparison': name, 'A': a, 'B': b, 'n windows': int(len(d)), 'Mean F1 difference (A - B)': round(float(d.mean()), 4), '95 % CI': f'{lo:+.4f} to {hi:+.4f}'.replace('-', '-'),
                     'Windows A better': int((d > 0).sum()), 'p (sign-flip)': p})
    for m in [m for m in model_order(summ) if family(m) != 'baseline']:
        if m in wide and 'NBR<0' in wide: add('model vs NBR < 0 rule', label(m), label('NBR<0'), (wide[m] - wide['NBR<0']).dropna())
    if 'pspnet' in wide and 'deeplabv3' in wide: add('PSPNet vs DeepLabV3 (planned)', label('pspnet'), label('deeplabv3'), (wide['pspnet'] - wide['deeplabv3']).dropna())
    cnn = [m for m in CNN_SET if m in wide]; tr = [m for m in TRANSFORMER_SET if m in wide]
    if cnn and tr: add('ResNet-50 encoder-decoder mean vs transformer mean (planned)', 'mean of ' + ', '.join(label(m) for m in cnn), 'mean of ' + ', '.join(label(m) for m in tr), (wide[cnn].mean(1) - wide[tr].mean(1)).dropna())
    for m in ('unet', 'deeplabv3', 'pspnet'):
        if m in wide and m + '_C2' in wide: add('objective contrast: weighted CE + Dice vs plain CE', label(m), label(m + '_C2'), (wide[m] - wide[m + '_C2']).dropna())
    best = [m for m in model_order(summ) if family(m) != 'baseline'][:1]       # the best-scoring learned configuration (mean F1)
    for m in [m for m in model_order(summ) if m in RQ2_SET]:
        for b in best:
            if m in wide and b in wide and m != b: add('vs best-scoring configuration', label(m), label(b), (wide[m] - wide[b]).dropna())
    t = pd.DataFrame(rows)
    if len(t): t['p (Holm, within this table)'] = holm(t['p (sign-flip)'].values); t['p (sign-flip)'] = t['p (sign-flip)'].map(fmt_p); t['p (Holm, within this table)'] = t['p (Holm, within this table)'].map(fmt_p)
    t.to_csv(os.path.join(out, 'tableP2_planned_comparisons.csv'), index=False); open(os.path.join(out, 'tableP2_planned_comparisons.md'), 'w').write(t.to_markdown(index=False) + '\n')
    if 'dNBR>0.1' in wide:   # more-information baseline (pre-fire scene): its own table and its own Holm family
        rows.clear()
        for m in [m for m in model_order(summ) if family(m) != 'baseline']:
            if m in wide: add('model vs dNBR > 0.1 rule (pre-fire scene)', label(m), label('dNBR>0.1'), (wide[m] - wide['dNBR>0.1']).dropna())
        if 'Otsu(dNBR)' in wide:
            for m in [m for m in model_order(summ) if family(m) != 'baseline']:
                if m in wide: add('model vs Otsu threshold on dNBR', label(m), label('Otsu(dNBR)'), (wide[m] - wide['Otsu(dNBR)']).dropna())
            add('dNBR rules against each other', label('Otsu(dNBR)'), label('dNBR>0.1'), (wide['Otsu(dNBR)'] - wide['dNBR>0.1']).dropna())
        for b in ('dNBR>0.1', 'Otsu(dNBR)'):
            if b in wide and 'NBR<0' in wide: add('pre-fire rule vs single-scene rule', label(b), label('NBR<0'), (wide[b] - wide['NBR<0']).dropna())
        t2 = pd.DataFrame(rows)
        t2['p (Holm, within this table)'] = holm(t2['p (sign-flip)'].values); t2['p (sign-flip)'] = t2['p (sign-flip)'].map(fmt_p); t2['p (Holm, within this table)'] = t2['p (Holm, within this table)'].map(fmt_p)
        t2.to_csv(os.path.join(out, 'tableP2b_dnbr_baseline.csv'), index=False); open(os.path.join(out, 'tableP2b_dnbr_baseline.md'), 'w').write(t2.to_markdown(index=False) + '\n')
    return t

def table_negatives(negs, summ, out):
    if negs.empty: return negs
    order = [m for m in model_order(summ) if m in set(negs.model)]
    t = negs.set_index('model').loc[order].reset_index()
    t.insert(0, 'Model', t.model.map(label)); t = t.drop(columns='model')
    t = t.rename(columns={'n_negative_windows': 'Fire-free windows', 'mean_false_burned_ha': 'Mean mapped burn (ha)', 'max_false_burned_ha': 'Largest mapped burn (ha)', 'mean_false_burned_fraction': 'Mean mapped fraction of the window'})
    t['Mean mapped burn (ha)'] = t['Mean mapped burn (ha)'].round(0); t['Largest mapped burn (ha)'] = t['Largest mapped burn (ha)'].round(0); t['Mean mapped fraction of the window'] = (100 * t['Mean mapped fraction of the window']).round(2).astype(str) + ' %'
    t.to_csv(os.path.join(out, 'tableP3_fire_free_windows.csv'), index=False); open(os.path.join(out, 'tableP3_fire_free_windows.md'), 'w').write(t.to_markdown(index=False) + '\n')
    return t

def table_dataset(man, out):
    cov = covariates(man); rows = []
    for p, (pname, col) in PROV.items():
        g = man[man.province == p]; f = g[~g.is_negative]; c = cov.loc[f.window_id]
        rows.append({'Province': pname, 'Fire windows': len(f), 'of which centred on named agency fires': int(f.source.str.contains('original', case=False).sum()),   # manifest tag of the 32 windows centred on named agency fires
                     'Fire-free windows': int(g.is_negative.sum()), 'Fire years': '-'.join(str(int(v)) for v in sorted(f.fire_year.unique())),
                     'Burned area per window, median (range) ha': f'{f.burned_ha.median():,.0f} ({f.burned_ha.min():,.0f}-{f.burned_ha.max():,.0f})',
                     'Burned fraction, median': f'{c.burned_fraction.median():.2f}', 'Days after the last NBAC fire date, median (range)': f'{c.days_after_fire.median():.0f} ({c.days_after_fire.min():.0f}-{c.days_after_fire.max():.0f})',
                     'Next-season scenes': int((f.scene_year_offset > 0).sum()), 'Clear fraction, median (min)': f'{f.clear_fraction.median():.2f} ({f.clear_fraction.min():.2f})'})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out, 'tableP4_dataset.csv'), index=False); open(os.path.join(out, 'tableP4_dataset.md'), 'w').write(t.to_markdown(index=False) + '\n')
    return t

def table_training(runs, summ, out):
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, '*', 'ledger.json'))):
        L = json.load(open(d)); run = os.path.basename(os.path.dirname(d)); cfgp = os.path.join(os.path.dirname(d), 'config.json')
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
        m = cfg.get('model', run.split('_F-')[0]) + ('_' + cfg['tag'] if cfg.get('tag') else '')
        rows.append({'model': m, 'Model': label(m), 'Fold': cfg.get('fold', run.split('_')[-1]), 'Epochs run': L.get('epochs_run'), 'Best validation F1': round(L.get('best_val_F1_mean_window', np.nan), 3),
                     'Wall time (min)': round(L.get('wall_seconds', np.nan) / 60, 0), 'Batch': cfg.get('batch'), 'GPU': L.get('gpu', '')})
    t = pd.DataFrame(rows)
    if len(t):
        order = {m: i for i, m in enumerate(model_order(summ))}; t['o'] = t.model.map(lambda m: order.get(m, 99)); t = t.sort_values(['o', 'Fold']).drop(columns=['o', 'model'])
        t.to_csv(os.path.join(out, 'tableP5_training_runs.csv'), index=False); open(os.path.join(out, 'tableP5_training_runs.md'), 'w').write(t.to_markdown(index=False) + '\n')
    return t

# ---------------------------------------------------------------- PowerPoint with native, editable charts
def make_pptx(summ, negs, allw, out):
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData, XyChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.oxml.ns import qn
    from lxml import etree
    prs = Presentation(); prs.slide_width = Inches(10); prs.slide_height = Inches(5.625)
    TNR = 'Times New Roman'
    def style_chart(ch, ylabel, ymin=None, ymax=None):
        ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.name = TNR; ch.legend.font.size = Pt(9)
        va = ch.value_axis; va.has_title = True; va.axis_title.text_frame.text = ylabel
        r = va.axis_title.text_frame.paragraphs[0].runs[0]; r.font.name = TNR; r.font.size = Pt(9)
        if ymin is not None: va.minimum_scale = ymin
        if ymax is not None: va.maximum_scale = ymax
        va.tick_labels.font.name = TNR; va.tick_labels.font.size = Pt(8); ch.category_axis.tick_labels.font.name = TNR; ch.category_axis.tick_labels.font.size = Pt(7)
        va.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE6, 0xE6)
    def add_slide(title):
        s = prs.slides.add_slide(prs.slide_layouts[5]); s.shapes.title.text = title
        run = s.shapes.title.text_frame.paragraphs[0].runs[0]; run.font.size = Pt(14); run.font.name = TNR
        return s
    def error_bars(ser, plus, minus):
        """Custom per-point error bars (not in the python-pptx API): c:errBars inserted before c:cat/c:xVal."""
        eb = etree.SubElement(ser._element, qn('c:errBars'))
        etree.SubElement(eb, qn('c:errBarType')).set('val', 'both'); etree.SubElement(eb, qn('c:errValType')).set('val', 'cust'); etree.SubElement(eb, qn('c:noEndCap')).set('val', '0')
        for tag, vals in (('c:plus', plus), ('c:minus', minus)):
            e = etree.SubElement(eb, qn(tag)); lit = etree.SubElement(e, qn('c:numLit')); etree.SubElement(lit, qn('c:formatCode')).text = 'General'
            etree.SubElement(lit, qn('c:ptCount')).set('val', str(len(vals)))
            for i, v in enumerate(vals):
                pt = etree.SubElement(lit, qn('c:pt')); pt.set('idx', str(i)); etree.SubElement(pt, qn('c:v')).text = f'{float(v):.5f}'
        sp = etree.SubElement(eb, qn('c:spPr')); ln = etree.SubElement(sp, qn('a:ln')); ln.set('w', '9525'); sf = etree.SubElement(ln, qn('a:solidFill')); etree.SubElement(sf, qn('a:srgbClr')).set('val', '333333')
        anchor = ser._element.find(qn('c:cat'))
        if anchor is None: anchor = ser._element.find(qn('c:xVal'))
        if anchor is not None: anchor.addprevious(eb)
    order = model_order(summ); s = summ.set_index('model').loc[order]
    # slide 1: mean F1 per model with 95 % CI error bars, one series per family (so the colours are editable per family)
    sl = add_slide('Mean per-window burned-class F1 on the held-out province (95 % window-bootstrap interval)')
    cd = CategoryChartData(); cd.categories = [label(m) for m in order]
    fams_present = [f for f in FAMILY_ORDER if any(family(m) == f for m in order)]
    for f in fams_present:
        cd.add_series(FAMILY[f][0], [round(float(s.loc[m].mean_F1), 4) if family(m) == f else None for m in order])
    gf = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.3), Inches(0.9), Inches(9.4), Inches(4.5), cd); ch = gf.chart; style_chart(ch, 'Mean per-window burned-class F1', 0.0, 1.0)
    pl = ch.plots[0]; pl.gap_width = 40; pl.overlap = 100; pl.has_data_labels = True; pl.data_labels.font.size = Pt(6); pl.data_labels.font.name = TNR; pl.data_labels.number_format = '0.000'; pl.data_labels.number_format_is_linked = False
    for ser, f in zip(pl.series, fams_present):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(FAMILY[f][1].lstrip('#').upper())
        if f == 'loss': ser.format.fill.fore_color.rgb = RGBColor(0xF2, 0xBC, 0x99)
        plus = [float(s.loc[m].F1_ci_hi - s.loc[m].mean_F1) if family(m) == f else 0.0 for m in order]; minus = [float(s.loc[m].mean_F1 - s.loc[m].F1_ci_lo) if family(m) == f else 0.0 for m in order]
        error_bars(ser, plus, minus)
    # slide 2: per-province means
    sl = add_slide('Mean per-window burned-class F1 by held-out province')
    cd = CategoryChartData(); cd.categories = [label(m) for m in order]
    for p, (pname, col) in PROV.items(): cd.add_series(pname, [round(float(s.loc[m].get(f'mean_F1_{p}', np.nan)), 4) for m in order])
    gf = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.3), Inches(0.9), Inches(9.4), Inches(4.5), cd); ch = gf.chart; style_chart(ch, 'Mean per-window burned-class F1', 0.0, 1.0)
    pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -10
    for ser, (p, (pname, col)) in zip(pl.series, PROV.items()): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(col.lstrip('#').upper())
    # slide 3: fire-free windows
    if not negs.empty:
        sl = add_slide('Burned area mapped in the fire-free windows (mean of 15 windows, ha)')
        n = negs.set_index('model'); cats = [m for m in order if m in n.index]
        cd = CategoryChartData(); cd.categories = [label(m) for m in cats]; cd.add_series('Mean mapped burn (ha)', [round(float(n.loc[m].mean_false_burned_ha), 1) for m in cats]); cd.add_series('Largest mapped burn (ha)', [round(float(n.loc[m].max_false_burned_ha), 1) for m in cats])
        gf = sl.shapes.add_slide if False else sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.3), Inches(0.9), Inches(9.4), Inches(4.5), cd); ch = gf.chart; style_chart(ch, 'Hectares mapped as burned (log scale)')
        va = ch.value_axis; sc = va._element.find(qn('c:scaling')); lb = etree.SubElement(sc, qn('c:logBase')); lb.set('val', '10'); sc.insert(0, lb); va.minimum_scale = 1
        pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -10
        for ser, col in zip(pl.series, ('#4D4D4D', '#B3B3B3')): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(col.lstrip('#').upper())
    # slide 4: area agreement scatter for the best learned model
    best = [m for m in order if family(m) != 'baseline'][0]; g = allw[allw.model == best]
    sl = add_slide(f'Mapped against NBAC burned area per window - {label(best)}')
    xy = XyChartData()
    for p, (pname, col) in PROV.items():
        se = xy.add_series(pname); gg = g[g.province == p]
        for x, y in zip(gg.ref_ha, gg.pred_ha): se.add_data_point(round(float(x), 1), round(float(y), 1))
    gf = sl.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(0.3), Inches(0.9), Inches(9.4), Inches(4.5), xy); ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.name = TNR; ch.legend.font.size = Pt(9)
    for ax_, name in ((ch.category_axis, 'NBAC burned area in window (ha)'), (ch.value_axis, 'Mapped burned area (ha)')):
        ax_.has_title = True; ax_.axis_title.text_frame.text = name; r = ax_.axis_title.text_frame.paragraphs[0].runs[0]; r.font.name = TNR; r.font.size = Pt(9); ax_.tick_labels.font.name = TNR; ax_.tick_labels.font.size = Pt(8)
        sc = ax_._element.find(qn('c:scaling')); lb = etree.SubElement(sc, qn('c:logBase')); lb.set('val', '10'); sc.insert(0, lb); ax_.minimum_scale = 100
    for ser, (p, (pname, col)) in zip(ch.plots[0].series, PROV.items()):
        ser.marker.style = XL_MARKER_STYLE.CIRCLE; ser.marker.size = 5; ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(col.lstrip('#').upper()); ser.format.line.fill.background()
    prs.save(os.path.join(out, 'figures_planA_editable_charts.pptx'))

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tag', default='')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summ, allw, pairs, negs, raw, man = load(a.runs)
    fig_dataset(man, a.out); letters = fig_model_comparison(summ, a.out, pairs); fig_province(summ, a.out); fig_window_distribution(allw, summ, a.out)
    fig_paired(pairs, summ, a.out); fig_area(allw, summ, a.out); fig_negatives(raw, summ, a.out); best = fig_covariates(allw, summ, man, a.out)
    t1 = table_summary(summ, pairs, a.out); t2 = table_planned(allw, pairs, summ, a.out); table_negatives(negs, summ, a.out); table_dataset(man, a.out); table_training(a.runs, summ, a.out)
    make_pptx(summ, negs, allw, a.out)
    n_models = len([m for m in summ.model if family(m) != 'baseline']); n_rules = len([m for m in summ.model if family(m) == 'baseline']); has_dnbr = 'dNBR>0.1' in set(summ.model)
    open(os.path.join(a.out, 'README_planA_figures.md'), 'w').write(f"""# Figures and tables - generated by analysis/make_figures_planA.py {a.tag}
Source: {a.runs} (results of eval.py, summary of summarize.py, manifest of build_dataset.py). {n_models} learned models + {n_rules} index rules{' (two on the single post-fire scene, two on dNBR with one pre-fire scene per window from build_prefire.py / eval_dnbr.py)' if has_dnbr else ''}, 120 fire windows (40 per province) scored in the fold that held out their province, 15 fire-free windows.
Font in the PNGs: {FONT}{'' if FONT == 'Times New Roman' else ' (Times New Roman not installed on the rendering machine; Liberation Serif is metric-compatible - re-run on a machine with Times New Roman or drop the .ttf files into a fonts/ folder next to this script)'}. 600 dpi. Plain hyphens.
Colours (checked for the common colour-vision deficiencies, all pairs distinct): convolutional encoder-decoders {VERMILION}, transformer networks {PURPLE}, Earth-observation foundation models {FBLUE}, index rules {GREY} (neutral by design); objective-contrast runs (plain cross-entropy) = hollow markers; provinces AB {P_AB} / BC {P_BC} / SK {P_SK} (as in the ArcGIS Pro map, Fig. 1).
- figS1_dataset_distributions.png (supplement): acquisition lag of the same-season scenes and burned area of the fire windows. The study-area map (Fig. 1) and the reference examples (Fig. 2) are built in ArcGIS Pro; the workflow (Fig. 3) is an editable PowerPoint. figP0_dataset.png of 15 Oct is superseded.
- figP1_model_comparison.png: mean per-window burned-class F1 (dot), 95 % window-bootstrap interval (2,000 replicates, seed 42), median (tick); models sorted; broken F1 axis (label-free rules left, learned models right); letters = groups of learned models whose pairwise differences are not significant (Holm-adjusted sign-flip p >= 0.05): {', '.join(f'{label(m)} {l}' for m, l in letters.items())}.
- figP2_province.png: the same statistic per held-out province.
- figP3_window_distribution.png: per-window F1 (boxes: quartiles, whiskers: 1.5 IQR, dots: windows by province).
- figP4_paired_differences.png: (a) each learned model minus the NBR < 0 rule (filled marker){' and minus the dNBR > 0.1 rule (grey diamond)' if has_dnbr else ''} with bootstrap interval and windows won; (b) pairwise mean differences of the learned models (lower triangle, column minus row; models ordered by mean F1), outlined bold = Holm-adjusted sign-flip p < 0.05 with the adjustment over all 105 pairs of the fourteen models and the NBR < 0 rule.
- figP5_area_agreement.png: mapped vs NBAC burned area per window for the three best-scoring learned configurations{', the dNBR > 0.1 rule' if has_dnbr else ''} and the NBR < 0 rule.
- figP6_fire_free_windows.png: hectares mapped as burned in the 15 fire-free windows.
- figP7_covariates.png (exploratory): per-window F1 of {label(best)} against burned fraction, days after the fire, clear fraction and burned area (Spearman).
- tableP1_summary_by_model: Mean F1 with CI, median, pooled, precision, recall, IoU, per-province means, area errors, gain over NBR < 0 with Holm p (over all pairs).
- tableP2_planned_comparisons: the Section 2.8 comparisons (model vs NBR < 0, PSPNet vs DeepLabV3, ResNet-50 encoder-decoder mean vs transformer mean, objective contrast, RQ2 configurations vs the best-scoring configuration), sign-flip tests, Holm within the table.{chr(10) + '- tableP2b_dnbr_baseline: more-information baseline - every learned model against the dNBR > 0.1 rule and the Otsu(dNBR) rule, the two dNBR rules against each other and against NBR < 0; sign-flip tests, Holm within the table.' if has_dnbr else ''}
- tableP3_fire_free_windows, tableP4_dataset, tableP5_training_runs (epochs, best validation F1, wall time, GPU).
- figures_planA_editable_charts.pptx: native PowerPoint charts of P1 (with CI error bars), P2, P6 and P5 (Times New Roman).
- figP8_qualitative.png, figP9_hard_windows.png, figP10_fire_free_examples.png: written by analysis/make_maps_planA.py (v3) from the quick-look export (export_quicklook.py, windows chosen on SegFormer: per province the best, median and weakest window, the five weakest windows overall and two fire-free windows):
  P8 = the window of median SegFormer F1 in each province: --windows Hoohey_Creek,Greer_Creek,NBAC2024_SK_388 --models mask2former,segformer,prithvi_lora,unet
  P9 = the five windows of lowest SegFormer F1 and the weakest Saskatchewan window: --windows NBAC2023_AB_718,NBAC2024_BC_200,NBAC2024_BC_208,NBAC2024_BC_193,NBAC2024_BC_171,NBAC2024_SK_401 --models mask2former,segformer,prithvi_lora,unet,terramind
  P10 = two fire-free windows: --windows NEG2023_SK_01,NEG2023_AB_01 --models mask2former,segformer,unet,terramind
  Colours as in Fig. 2: burned in reference and map dark red (178,24,43), omission light red (244,165,130), commission blue (33,102,172), unburned beige (238,236,225), ignored amber (253,192,85), masked grey (150,150,150).
""")
    print('wrote', a.out, '| models:', ', '.join(model_order(summ)))

if __name__ == '__main__':
    main()
