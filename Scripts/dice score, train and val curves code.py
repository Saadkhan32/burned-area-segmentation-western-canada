# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 07:45:38 2026

@author: saadz
"""

# -*- coding: utf-8 -*-
"""
Multi-model training curves plotter (5 rows, single frame)
- Reads: CSV, JSON (dict/list), JSONL (mmengine log.json)
- Single legend at top
- Times New Roman, font size 12 everywhere (labels, ticks, titles, legend)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# ------------------------- Global Styling (ALL = 12) -------------------------
BASE_FONT = 16
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
    "legend.fontsize": BASE_FONT,
    "figure.titlesize": BASE_FONT,
})

# solid, distinctive colors
COL_TRAIN = "#0057D9"   # blue
COL_VAL   = "#6A00FF"   # purple
COL_DICE  = "#0A8F3D"   # green

LW_TRAIN = 1.6
LW_VAL   = 1.6
LW_DICE  = 2.0

# ------------------------- Helpers -------------------------
def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "").replace("-", "").replace(".", "").replace("/", "").replace("\\", "")

def _find_col(columns, candidates):
    cols = list(columns)
    cols_norm = {_norm_key(c): c for c in cols}

    for cand in candidates:
        cn = _norm_key(cand)
        if cn in cols_norm:
            return cols_norm[cn]

    for cand in candidates:
        cn = _norm_key(cand)
        for k_norm, k_orig in cols_norm.items():
            if cn in k_norm:
                return k_orig
    return None

def _to_float_array(x):
    if x is None:
        return None
    arr = np.asarray(x)
    try:
        return arr.astype(float)
    except Exception:
        return pd.to_numeric(arr, errors="coerce").to_numpy()

def _read_first_lines(path, n=25):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for _ in range(n):
                ln = f.readline()
                if not ln:
                    break
                lines.append(ln.rstrip("\n"))
        return "\n".join(lines)
    except Exception:
        return "(Could not read file preview.)"

def _align_lengths(h):
    arrs = [h.get("epochs"), h.get("train_loss"), h.get("val_loss"), h.get("dice")]
    lengths = [len(a) for a in arrs if a is not None]
    if not lengths:
        return h
    m = min(lengths)
    out = {}
    for k in ["epochs", "train_loss", "val_loss", "dice"]:
        v = h.get(k)
        out[k] = v[:m] if v is not None else None
    return out

# ------------------------- Parse dataframe -------------------------
def parse_from_dataframe(df: pd.DataFrame, source_label="DATA"):
    if df is None or df.empty:
        raise ValueError(f"[{source_label}] Empty dataframe.")

    epoch_col = _find_col(df.columns, ["epoch", "epochs", "ep", "iter", "iteration", "step", "steps"])
    epochs = np.arange(1, len(df) + 1) if epoch_col is None else _to_float_array(df[epoch_col].values)

    train_loss_col = _find_col(df.columns, ["training_loss", "train_loss", "loss", "trainingloss", "trainloss"])
    val_loss_col   = _find_col(df.columns, ["validation_loss", "val_loss", "valid_loss", "valloss"])
    dice_col       = _find_col(df.columns, ["Dice", "dice", "dice_score", "DiceScore", "val_dice", "mDice", "MeanDice", "dice_coef"])

    train_loss = _to_float_array(df[train_loss_col].values) if train_loss_col else None
    val_loss   = _to_float_array(df[val_loss_col].values) if val_loss_col else None
    dice       = _to_float_array(df[dice_col].values) if dice_col else None

    if train_loss is None and val_loss is None and dice is None:
        raise ValueError(f"[{source_label}] Could not detect metrics.\nColumns found: {list(df.columns)}")

    return {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss, "dice": dice}

# ------------------------- Load any history file -------------------------
def load_history_any(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt", ".log"]:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)
        return parse_from_dataframe(df, source_label="CSV")

    if ext == ".json":
        # Try normal JSON
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
        except Exception:
            obj = None

        # JSONL fallback (mmengine log.json)
        if obj is None:
            records = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        records.append(json.loads(ln))
                    except Exception:
                        continue
            if records:
                df = pd.DataFrame([r for r in records if isinstance(r, dict)])
                return parse_from_dataframe(df, source_label="JSONL")

            preview = _read_first_lines(path, 25)
            raise ValueError("[JSON] Failed to parse JSON or JSONL.\n\nFIRST LINES:\n" + preview)

        # JSON list-of-dicts
        if isinstance(obj, list):
            df = pd.DataFrame([r for r in obj if isinstance(r, dict)])
            return parse_from_dataframe(df, source_label="JSON_LIST")

        # JSON dict with embedded list-of-dicts
        if isinstance(obj, dict):
            for key in ["training_validation_loss_per_epoch", "history", "log_history", "logs", "data"]:
                if key in obj and isinstance(obj[key], list) and obj[key] and isinstance(obj[key][0], dict):
                    df = pd.DataFrame(obj[key])
                    return parse_from_dataframe(df, source_label=f"JSON.{key}")

            if len(obj) == 1:
                only_key = next(iter(obj.keys()))
                val = obj[only_key]
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    df = pd.DataFrame(val)
                    return parse_from_dataframe(df, source_label=f"JSON.{only_key}")

            raise ValueError(f"[JSON_DICT] Parsed JSON dict but did not find list-of-dicts per epoch.\nKeys: {list(obj.keys())}")

        raise ValueError("[JSON] Unsupported JSON structure.")

    raise ValueError(f"Unsupported file type: {ext}")

# ------------------------- UI -------------------------
def ask_file_for_model(model_name):
    messagebox.showinfo(
        "Select history/log file",
        f"Select the training history/log for: {model_name}\n\nSupported: CSV, JSON, JSONL(mmengine log.json)"
    )
    return filedialog.askopenfilename(
        title=f"Select history/log for {model_name}",
        filetypes=[
            ("All supported", "*.csv *.json *.txt *.log"),
            ("JSON", "*.json"),
            ("CSV", "*.csv"),
            ("Text/Log", "*.txt *.log"),
            ("All files", "*.*")
        ]
    )

def ask_save_path():
    return filedialog.asksaveasfilename(
        title="Save figure as...",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
    )

# ------------------------- Plotting -------------------------
def plot_all_models(histories, model_names):
    n = len(model_names)

    FIG_W = 15
    ROW_H = 3.0
    TOP_EXTRA = 1.8
    fig_h = n * ROW_H + TOP_EXTRA

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(FIG_W, fig_h))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Training Curves (Loss) and Validation Performance (Dice) — All Models",
        y=0.985, fontsize=BASE_FONT, fontweight="bold"
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.06,
        left=0.08,
        right=0.92,
        hspace=0.45
    )

    legend_handles = None

    for ax, name in zip(axes, model_names):
        h = _align_lengths(histories[name])
        epochs = h["epochs"]

        line_train = None
        line_val = None

        if h["train_loss"] is not None:
            line_train, = ax.plot(
                epochs, h["train_loss"],
                color=COL_TRAIN, linewidth=LW_TRAIN, linestyle="-",
                label="Training Loss"
            )
        if h["val_loss"] is not None:
            line_val, = ax.plot(
                epochs, h["val_loss"],
                color=COL_VAL, linewidth=LW_VAL, linestyle="-",
                label="Validation Loss"
            )

        ax.set_ylabel("Loss", fontsize=BASE_FONT)
        ax.set_title(name, pad=6, fontsize=BASE_FONT, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
        ax.set_xlabel("Epoch", fontsize=BASE_FONT)

        # Force tick labels = 12 (sometimes backend ignores rcParams)
        ax.tick_params(axis="both", labelsize=BASE_FONT)

        ax2 = ax.twinx()
        line_dice = None
        if h["dice"] is not None:
            line_dice, = ax2.plot(
                epochs, h["dice"],
                color=COL_DICE, linewidth=LW_DICE, linestyle="-",
                label="Dice Score"
            )
        ax2.set_ylabel("Dice Score", fontsize=BASE_FONT)
        ax2.tick_params(axis="y", labelsize=BASE_FONT)

        if legend_handles is None:
            handles, labels = [], []
            for ln in [line_train, line_val, line_dice]:
                if ln is not None:
                    handles.append(ln)
                    labels.append(ln.get_label())
            legend_handles = (handles, labels)

    if legend_handles and legend_handles[0]:
        fig.legend(
            legend_handles[0], legend_handles[1],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=True,
            prop={"family": "Times New Roman", "size": BASE_FONT}
        )

    return fig

# ------------------------- Main -------------------------
def main():
    root = tk.Tk()
    root.withdraw()

    models = ["DeepLabV3-ResNet50", "UNET-ResNet50", "PSPNet-ResNet50", "SamLoRa-ViTB", "Mask2Former"]

    selected = {}
    for m in models:
        p = ask_file_for_model(m)
        if not p:
            messagebox.showwarning("Cancelled", f"No file selected for {m}. Exiting.")
            return
        selected[m] = p

    histories = {}
    for m, p in selected.items():
        try:
            histories[m] = load_history_any(p)
        except Exception as e:
            preview = _read_first_lines(p, 25)
            messagebox.showerror(
                "Read error",
                f"Failed to read file for {m}:\n{p}\n\nERROR:\n{e}\n\nFIRST LINES PREVIEW:\n{preview}"
            )
            return

    fig = plot_all_models(histories, models)

    if messagebox.askyesno("Save figure", "Do you want to save the plot?"):
        out = ask_save_path()
        if out:
            try:
                if out.lower().endswith(".png"):
                    fig.savefig(out, dpi=450, bbox_inches="tight")
                else:
                    fig.savefig(out, bbox_inches="tight")
                messagebox.showinfo("Saved", f"Saved figure to:\n{out}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

    plt.show()

if __name__ == "__main__":
    main()
