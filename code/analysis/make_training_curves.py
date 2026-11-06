"""Training curves of the 42 runs (14 configurations x 3 leave-one-province-out folds).

One panel per configuration (Table 4 order of the revised manuscript): training loss (solid, left axis) and mean
per-window burned-class F1 on the seven validation windows (dashed, right axis) per
epoch, one colour per fold (province colours of Fig. 1); the dot marks the epoch of the
best validation F1, whose checkpoint was kept (early stopping: ten epochs without an
improvement of 0.001). Data: <run>/log.csv written by train.py. Validation loss was not
logged; the validation F1 is the early-stopping criterion.
Style as make_figures_planA.py: Times New Roman, 600 dpi PNG, plain hyphens.
Usage: python analysis/make_training_curves.py --runs runs/ --out figP11_training_curves.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.ttf")):
    font_manager.fontManager.addfont(_f)
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Times New Roman" if "Times New Roman" in fams else ("Liberation Serif" if "Liberation Serif" in fams else "serif")
plt.rcParams.update({"font.family": FONT, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
                     "ytick.color": "#444444", "xtick.color": "#444444", "savefig.dpi": 600, "figure.dpi": 110})
P_BC, P_AB, P_SK = "#0072B2", "#E69F00", "#009E73"
FOLD = {"F-BC": ("BC held out", P_BC), "F-AB": ("AB held out", P_AB), "F-SK": ("SK held out", P_SK)}
CONFIGS = [  # (run prefix, label as in Table 4): convolutional encoder-decoders, transformer networks, foundation models, objective contrast
    ("unet", "U-Net (ResNet-50)"), ("deeplabv3", "DeepLabV3 (ResNet-50)"), ("pspnet", "PSPNet (ResNet-50)"),
    ("deeplabv3plus", "DeepLabV3+ (ResNet-50)"), ("upernet_convnext", "UPerNet (ConvNeXt-T)"),
    ("samlora", "SAM-LoRA (ViT-B)"), ("mask2former", "Mask2Former (Swin-L)"), ("segformer", "SegFormer (MiT-B2)"),
    ("prithvi", "Prithvi-EO-2.0, full fine-tuning"), ("prithvi_lora", "Prithvi-EO-2.0, LoRA"), ("terramind", "TerraMind, full fine-tuning"),
    ("unet_C2", "U-Net, plain CE"), ("deeplabv3_C2", "DeepLabV3, plain CE"), ("pspnet_C2", "PSPNet, plain CE"),
]


def run_dir(runs, prefix, fold):
    if prefix.endswith("_C2"):                       # objective-contrast runs (plain CE): <model>_<fold>_C2
        return os.path.join(runs, f"{prefix[:-3]}_{fold}_C2")
    return os.path.join(runs, f"{prefix}_{fold}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ncol, nrow = 2, 7
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 9.3))
    xmax = 0
    for k, (prefix, label) in enumerate(CONFIGS):
        ax = axes[k // ncol, k % ncol]
        ax2 = ax.twinx()
        f1_lo, f1_hi = 1.0, 0.0
        for fold, (fname, col) in FOLD.items():
            lg = pd.read_csv(os.path.join(run_dir(a.runs, prefix, fold), "log.csv"))
            xmax = max(xmax, int(lg.epoch.max()))
            ax.plot(lg.epoch, lg.train_loss, color=col, lw=0.9, ls="-")
            ax2.plot(lg.epoch, lg.val_F1_mean_window, color=col, lw=0.9, ls="--")
            b = int(lg.val_F1_mean_window.idxmax())
            ax2.plot(lg.epoch[b], lg.val_F1_mean_window[b], "o", color=col, ms=3.2, mec="white", mew=0.5, zorder=5)
            nan = lg.index[lg.train_loss.isna()]
            if len(nan):
                e0 = int(lg.epoch[nan[0]])
                ax2.annotate(f"loss not a number\nfrom epoch {e0}", xy=(e0, 0.5), xytext=(e0 - 2, 0.45), ha="right", va="center",
                             fontsize=5.8, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.5))
            f1_lo, f1_hi = min(f1_lo, lg.val_F1_mean_window.min()), max(f1_hi, lg.val_F1_mean_window.max())
        ax.set_title(label, fontsize=8, pad=3)
        ax.set_ylabel("Training loss", fontsize=7)
        ax2.set_ylabel("Validation F1", fontsize=7)
        ax2.set_ylim(max(0.0, f1_lo - 0.04), min(1.0, f1_hi + 0.02))
        ax.tick_params(axis="y", labelsize=6.3)
        ax2.tick_params(axis="y", labelsize=6.3)
        ax.grid(axis="x", color="#e6e6e6", lw=0.5)
        for s in ("top",):
            ax.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)
        if k // ncol == nrow - 1:
            ax.set_xlabel("Epoch")
    for ax in axes.ravel():
        ax.set_xlim(0, xmax + 1)
    folds = [Line2D([], [], color=c, lw=1.2, label=n) for n, c in FOLD.values()]
    styles = [Line2D([], [], color="#444444", lw=1.0, ls="-", label="training loss (left axis)"),
              Line2D([], [], color="#444444", lw=1.0, ls="--", label="validation F1, mean per window (right axis)"),
              Line2D([], [], color="#444444", marker="o", ms=3.5, mec="white", lw=0, label="best validation F1: checkpoint kept")]
    handles = [folds[0], styles[0], folds[1], styles[1], folds[2], styles[2]]   # column-major fill: folds on row 1, styles on row 2
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.995), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.6, w_pad=1.6)
    fig.savefig(a.out, dpi=600)
    print("wrote", a.out, "max epochs", xmax)


if __name__ == "__main__":
    main()
