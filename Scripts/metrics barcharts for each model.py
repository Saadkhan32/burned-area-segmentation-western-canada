# -*- coding: utf-8 -*-
"""
SPYDER: Paper-ready figure (2x2)
✅ Burned = solid bars
✅ Unburned = hatched bars
✅ Robust Excel column matching (fixes "bars not showing")
✅ Times New Roman everywhere
✅ Tkinter selects Alberta / BC / Saskatchewan files + output folder

Expected Excel format (first sheet):
  metric | (model columns...)
Rows include:
  precision_burned, recall_burned, iou_burned, f1_burned
  precision_unburned, recall_unburned, iou_unburned, f1_unburned
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


# ----------------------------- Tkinter helpers -----------------------------
_TK = None
def _root():
    global _TK
    if _TK is None:
        _TK = tk.Tk()
        _TK.withdraw()
        try:
            _TK.attributes("-topmost", True)
        except Exception:
            pass
    return _TK

def pick_excel_file(title):
    r = _root()
    f = filedialog.askopenfilename(
        title=title,
        parent=r,
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    if not f:
        raise SystemExit(f"Cancelled: {title}")
    return os.path.normpath(f)

def pick_out_dir(title):
    r = _root()
    d = filedialog.askdirectory(title=title, parent=r)
    if not d:
        raise SystemExit("Cancelled output folder selection.")
    d = os.path.normpath(d)
    os.makedirs(d, exist_ok=True)
    return d

def ask_str(title, prompt, default=""):
    r = _root()
    v = simpledialog.askstring(title, prompt, initialvalue=str(default), parent=r)
    return default if v is None else str(v)

def ask_int(title, prompt, default=0):
    r = _root()
    v = simpledialog.askinteger(title, prompt, initialvalue=int(default), parent=r)
    return int(default) if v is None else int(v)

def show_info(title, msg):
    try:
        r = _root()
        messagebox.showinfo(title, msg, parent=r)
    except Exception:
        pass

def cleanup_tk():
    global _TK
    try:
        if _TK is not None:
            _TK.destroy()
    except Exception:
        pass
    _TK = None


# ----------------------------- Normalization helpers -----------------------------
def _norm(s):
    """Aggressive normalize for matching: remove spaces, dashes, underscores, dots; lowercase."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "").replace("_", "").replace(".", "")
    return s

# ✅ Your EXACT model names (canonical)
MODEL_ORDER = [
    "DeepLabV3-ResNet50",
    "Mask2Former",
    "PSPNet-ResNet50",
    "SamLoRA-ViTB",
    "UNet-ResNet50",
]

# Aliases (handles messy Excel headers)
MODEL_ALIASES = {
    "DeepLabV3-ResNet50": [
        "deeplabv3-resnet34", "deeplabv3resnet34", "deeplabv3_resnet34",
        "deeplabv3", "deeplab"
    ],
    "Mask2Former": [
        "mask2former", "mask_2_former", "mask2-former"
    ],
    "PSPNet-ResNet50": [
        "pspnet-resnet34", "pspnetresnet34", "pspnet_resnet34",
        "pspnet"
    ],
    "SamLoRA-ViTB": [
        "samlora-vitb", "samlora_vitb", "samlora vitb",
        "samlora"
    ],
    "UNet-ResNet50": [
        "unet-resnet34", "unetresnet34", "unet_resnet34",
        "unet"
    ],
}

def standardize_model_columns(df, file_label=""):
    """
    Renames df columns to canonical model names using MODEL_ALIASES.
    Raises a clear error if any canonical model cannot be found.
    """
    col_map = {}  # original_col -> canonical
    norm_cols = {_norm(c): c for c in df.columns}

    for canonical, aliases in MODEL_ALIASES.items():
        found = None

        # 1) exact canonical match
        if _norm(canonical) in norm_cols:
            found = norm_cols[_norm(canonical)]
        else:
            # 2) alias exact match
            for a in aliases:
                if _norm(a) in norm_cols:
                    found = norm_cols[_norm(a)]
                    break

            # 3) substring fallback
            if found is None:
                for nc, orig in norm_cols.items():
                    if any(_norm(a) in nc for a in aliases):
                        found = orig
                        break

        if found is not None:
            col_map[found] = canonical

    df2 = df.rename(columns=col_map)

    missing = [m for m in MODEL_ORDER if m not in df2.columns]
    if missing:
        raise ValueError(
            f"Missing model columns in {file_label or 'Excel file'}: {missing}\n\n"
            f"Columns found:\n{list(df.columns)}\n\n"
            f"Fix options:\n"
            f"  1) Rename your Excel headers to exactly:\n     {MODEL_ORDER}\n"
            f"  2) Or tell me your exact Excel headers and I will add aliases."
        )

    return df2


# ----------------------------- Read Excel -----------------------------
def read_metrics_wide_excel(path):
    df = pd.read_excel(path, sheet_name=0)

    metric_col = None
    for c in df.columns:
        if _norm(c) == "metric":
            metric_col = c
            break
    if metric_col is None:
        raise ValueError(f"'metric' column not found in: {os.path.basename(path)}")

    df = df.copy()
    df[metric_col] = df[metric_col].astype(str)
    df = df.set_index(metric_col)
    df.index = [_norm(i) for i in df.index]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")

    # ✅ fix bars-not-showing: normalize model columns
    df = standardize_model_columns(df, file_label=os.path.basename(path))
    return df

def get_row(df, key):
    k = _norm(key)
    if k not in df.index:
        raise ValueError(
            f"Missing row '{k}' in file.\n\n"
            f"Rows present (sample): {list(df.index)[:25]}"
        )
    return df.loc[k]


# ----------------------------- Plotting -----------------------------
def plot_like_sample_improved(dfs, provinces, out_png, out_pdf, dpi=600):
    plt.rcParams["font.family"] = "Times New Roman"

    # Model colors (distinct)
    colors = {
        "DeepLabV3-ResNet50": "#1f77b4",  # blue
        "Mask2Former":        "#ff7f0e",  # orange
        "PSPNet-ResNet50":    "#2ca02c",  # green
        "SamLoRA-ViTB":       "#d62728",  # red
        "UNet-ResNet50":      "#9467bd",  # purple
    }

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("iou", "IoU"),
        ("f1", "F-score"),
    ]

    n_prov = len(provinces)
    n_mod = len(MODEL_ORDER)
    x = np.arange(n_prov)

    # Paired bars layout per model within province
    group_span = 0.90
    total_slots = (2 * n_mod) + (n_mod - 1)  # bars + gap slots
    slot_w = group_span / total_slots
    bar_w = slot_w * 0.95

    offsets = []
    cur = -(group_span / 2.0) + slot_w / 2.0
    for j in range(n_mod):
        offsets.append(cur)     # burned
        cur += slot_w
        offsets.append(cur)     # unburned
        cur += slot_w
        if j < n_mod - 1:
            cur += slot_w       # gap
    offsets = np.array(offsets, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.ravel()

    yticks = np.arange(0, 1.01, 0.2)

    def style_ax(ax, ylabel):
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(provinces, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=12)

        ax.grid(True, axis="y", linestyle="--", linewidth=1.0, alpha=0.38)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    for ax_i, (mkey, mlabel) in enumerate(metrics):
        ax = axes[ax_i]

        burned = np.zeros((n_prov, n_mod), dtype=float)
        unburned = np.zeros((n_prov, n_mod), dtype=float)

        for i, df in enumerate(dfs):
            b_row = get_row(df, f"{mkey}_burned")
            u_row = get_row(df, f"{mkey}_unburned")
            for j, model in enumerate(MODEL_ORDER):
                burned[i, j] = float(b_row[model])
                unburned[i, j] = float(u_row[model])

        for j, model in enumerate(MODEL_ORDER):
            c = colors.get(model, "#444444")
            pos_b = x + offsets[2 * j]
            pos_u = x + offsets[2 * j + 1]

            ax.bar(pos_b, burned[:, j], width=bar_w, color=c,
                   edgecolor="black", linewidth=0.8, zorder=3)

            ax.bar(pos_u, unburned[:, j], width=bar_w, color=c,
                   edgecolor="black", linewidth=0.8, hatch="//", zorder=3)

        style_ax(ax, mlabel)

        for s in range(n_prov - 1):
            ax.axvline(s + 0.5, color="0.85", linewidth=0.8, zorder=0)

    # Legend
    import matplotlib.patches as mpatches
    model_patches = [mpatches.Patch(facecolor=colors[m], edgecolor="black", label=m) for m in MODEL_ORDER]
    burned_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Burned = solid")
    unburned_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Unburned = hatched")

    fig.legend(handles=model_patches + [burned_patch, unburned_patch],
               loc="upper center", ncol=3, frameon=True, fontsize=13,
               bbox_to_anchor=(0.5, 0.985), borderpad=0.6)

    fig.subplots_adjust(top=0.90, wspace=0.22, hspace=0.25)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Main -----------------------------
def main():
    try:
        f_ab = pick_excel_file("Select Alberta metrics Excel")
        f_bc = pick_excel_file("Select British Columbia metrics Excel")
        f_sk = pick_excel_file("Select Saskatchewan metrics Excel")

        loc_ab = ask_str("Province label", "Label for Alberta:", "Alberta")
        loc_bc = ask_str("Province label", "Label for British Columbia:", "British Columbia")
        loc_sk = ask_str("Province label", "Label for Saskatchewan:", "Saskatchewan")

        out_dir = pick_out_dir("Select/CREATE output folder for figure")
        dpi = ask_int("Export DPI", "Export DPI (600 recommended):", 600)

        df_ab = read_metrics_wide_excel(f_ab)
        df_bc = read_metrics_wide_excel(f_bc)
        df_sk = read_metrics_wide_excel(f_sk)

        out_png = os.path.join(out_dir, "burned_unburned_metrics_2x2_improved.png")
        out_pdf = os.path.join(out_dir, "burned_unburned_metrics_2x2_improved.pdf")

        plot_like_sample_improved(
            dfs=[df_ab, df_bc, df_sk],
            provinces=[loc_ab, loc_bc, loc_sk],
            out_png=out_png,
            out_pdf=out_pdf,
            dpi=dpi
        )

        msg = f"Saved:\n{out_png}\n{out_pdf}"
        print(msg)
        show_info("Done", msg)

    except Exception as e:
        messagebox.showerror("Error", str(e), parent=_root())
        raise
    finally:
        cleanup_tk()

if __name__ == "__main__":
    main()
