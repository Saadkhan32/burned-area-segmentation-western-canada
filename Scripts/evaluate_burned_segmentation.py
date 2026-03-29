import os
import json
import csv
import numpy as np
import arcpy
import tkinter as tk
from tkinter import filedialog, messagebox

arcpy.env.overwriteOutput = True


def pick_config_json():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select evaluator_config.json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    if not path:
        raise SystemExit("No config file selected.")
    return os.path.normpath(path)


def pick_gt_folder():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(
        title="Select ground-truth folder (manual labelled chips)"
    )
    root.destroy()
    if not path:
        raise SystemExit("No ground-truth folder selected.")
    return os.path.normpath(path)


def raster_to_array(path):
    try:
        arr = arcpy.RasterToNumPyArray(path)
        if hasattr(arr, "filled"):
            arr = arr.filled(np.nan)
        return np.array(arr)
    except Exception as e:
        raise RuntimeError(f"Failed to read raster: {path} -> {e}")


def find_matching_pred(gt_path, pred_dir):
    gt_name = os.path.basename(gt_path)
    base, _ = os.path.splitext(gt_name)

    valid_ext = (".tif", ".tiff", ".png")
    candidates = [
        f for f in os.listdir(pred_dir)
        if f.lower().endswith(valid_ext)
    ]

    if not candidates:
        return None

    for f in candidates:
        if os.path.splitext(f)[0] == base:
            return os.path.join(pred_dir, f)

    for f in candidates:
        b2, _ = os.path.splitext(f)
        if b2 == base + "_Classified":
            return os.path.join(pred_dir, f)

    for f in candidates:
        if f.startswith(base):
            return os.path.join(pred_dir, f)

    return None


def safe_div(num, den):
    return float(num) / float(den) if den not in (0, 0.0) else float("nan")


def compute_confusion(gt_arr, pred_arr, target_class, nodata_values=None):
    gt = np.array(gt_arr)
    pr = np.array(pred_arr)

    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs PRED {pr.shape}")

    gt_flat = gt.ravel()
    pr_flat = pr.ravel()

    valid = np.ones_like(gt_flat, dtype=bool)

    if np.issubdtype(gt_flat.dtype, np.floating):
        valid &= ~np.isnan(gt_flat)
    if np.issubdtype(pr_flat.dtype, np.floating):
        valid &= ~np.isnan(pr_flat)

    if nodata_values:
        nodata_values = set(nodata_values)
        for nv in nodata_values:
            valid &= (gt_flat != nv)
            valid &= (pr_flat != nv)

    if not np.any(valid):
        return 0, 0, 0, 0

    gt_v = gt_flat[valid]
    pr_v = pr_flat[valid]

    gt_pos = (gt_v == target_class)
    gt_neg = ~gt_pos
    pr_pos = (pr_v == target_class)
    pr_neg = ~pr_pos

    tp = int(np.logical_and(gt_pos, pr_pos).sum())
    tn = int(np.logical_and(gt_neg, pr_neg).sum())
    fp = int(np.logical_and(gt_neg, pr_pos).sum())
    fn = int(np.logical_and(gt_pos, pr_neg).sum())

    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou_burn = safe_div(tp, tp + fp + fn)
    dice_burn = safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou_burned": iou_burn,
        "dice_burned": dice_burn,
        "support_burned_pixels": tp + fn,
        "total_valid_pixels": total,
    }


def main():
    cfg_path = pick_config_json()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    gt_dir = pick_gt_folder()

    pred_dir = cfg.get("PRED_DIR")
    report_dir = cfg.get("REPORT_DIR")
    target_class = int(cfg.get("TARGET_CLASS_ID", 1))
    background_id = int(cfg.get("BACKGROUND_ID", 0))
    nodata_values = cfg.get("NODATA_VALUES", [])

    if not (pred_dir and report_dir):
        raise SystemExit("Config JSON must define PRED_DIR and REPORT_DIR.")

    gt_dir = os.path.normpath(gt_dir)
    pred_dir = os.path.normpath(pred_dir)
    report_dir = os.path.normpath(report_dir)
    os.makedirs(report_dir, exist_ok=True)

    arcpy.AddMessage(f"Ground truth directory (manual): {gt_dir}")
    arcpy.AddMessage(f"PRED_DIR: {pred_dir}")
    arcpy.AddMessage(f"REPORT_DIR: {report_dir}")
    arcpy.AddMessage(f"TARGET_CLASS_ID (burned): {target_class}")
    arcpy.AddMessage(f"BACKGROUND_ID (not used in metrics): {background_id}")
    arcpy.AddMessage(f"NODATA_VALUES: {nodata_values}")

    valid_ext = (".tif", ".tiff", ".png")
    gt_files = [
        f for f in os.listdir(gt_dir)
        if f.lower().endswith(valid_ext)
    ]
    if not gt_files:
        raise SystemExit(f"No label rasters found in ground-truth directory: {gt_dir}")

    gt_files.sort()
    print(f"Found {len(gt_files)} ground-truth tiles for evaluation.")
    arcpy.AddMessage(f"Found {len(gt_files)} ground-truth tiles for evaluation.")

    per_chip_csv = os.path.join(report_dir, "per_chip_metrics.csv")
    per_scene_csv = os.path.join(report_dir, "per_scene_metrics.csv")
    global_csv = os.path.join(report_dir, "global_metrics.csv")
    cm_csv = os.path.join(report_dir, "confusion_matrix.csv")

    G_tp = G_tn = G_fp = G_fn = 0
    scene_conf = {}
    skipped = 0

    with open(per_chip_csv, "w", newline="", encoding="utf-8") as fchip:
        chip_writer = csv.writer(fchip)
        chip_writer.writerow([
            "chip_name",
            "scene_name",
            "gt_path",
            "pred_path",
            "tp", "tn", "fp", "fn",
            "precision",
            "recall",
            "f1_score",
            "iou_burned",
            "dice_burned",
            "support_burned_pixels",
            "total_valid_pixels"
        ])

        for i, gt_name in enumerate(gt_files, start=1):
            gt_path = os.path.join(gt_dir, gt_name)
            scene_name = gt_name.split("__")[0]

            pred_path = find_matching_pred(gt_path, pred_dir)
            if pred_path is None or not os.path.exists(pred_path):
                msg = f"[{i}/{len(gt_files)}] No prediction for GT: {gt_name}"
                print(msg)
                arcpy.AddWarning(msg)
                skipped += 1
                continue

            msg = f"[{i}/{len(gt_files)}] Evaluating: {gt_name}"
            print(msg)
            arcpy.AddMessage(msg)

            try:
                gt_arr = raster_to_array(gt_path)
                pred_arr = raster_to_array(pred_path)
            except Exception as e:
                arcpy.AddWarning(f"Read failed for {gt_name}: {e}")
                skipped += 1
                continue

            try:
                tp, tn, fp, fn = compute_confusion(
                    gt_arr, pred_arr,
                    target_class=target_class,
                    nodata_values=nodata_values
                )
            except Exception as e:
                arcpy.AddWarning(f"Confusion failed for {gt_name}: {e}")
                skipped += 1
                continue

            if tp + tn + fp + fn == 0:
                arcpy.AddWarning(f"No valid pixels in tile (all nodata?): {gt_name}")
                skipped += 1
                continue

            G_tp += tp
            G_tn += tn
            G_fp += fp
            G_fn += fn

            if scene_name not in scene_conf:
                scene_conf[scene_name] = [0, 0, 0, 0]
            sc_tp, sc_tn, sc_fp, sc_fn = scene_conf[scene_name]
            scene_conf[scene_name] = [
                sc_tp + tp,
                sc_tn + tn,
                sc_fp + fp,
                sc_fn + fn,
            ]

            m = metrics_from_confusion(tp, tn, fp, fn)

            chip_writer.writerow([
                gt_name,
                scene_name,
                gt_path,
                pred_path,
                tp, tn, fp, fn,
                m["precision"],
                m["recall"],
                m["f1_score"],
                m["iou_burned"],
                m["dice_burned"],
                m["support_burned_pixels"],
                m["total_valid_pixels"],
            ])

    with open(per_scene_csv, "w", newline="", encoding="utf-8") as fscene:
        sw = csv.writer(fscene)
        sw.writerow([
            "scene_name",
            "tp", "tn", "fp", "fn",
            "precision",
            "recall",
            "f1_score",
            "iou_burned",
            "dice_burned",
            "support_burned_pixels",
            "total_valid_pixels"
        ])

        for scene_name, (tp, tn, fp, fn) in sorted(scene_conf.items()):
            m = metrics_from_confusion(tp, tn, fp, fn)
            sw.writerow([
                scene_name,
                tp, tn, fp, fn,
                m["precision"],
                m["recall"],
                m["f1_score"],
                m["iou_burned"],
                m["dice_burned"],
                m["support_burned_pixels"],
                m["total_valid_pixels"],
            ])

    G_metrics = metrics_from_confusion(G_tp, G_tn, G_fp, G_fn)

    with open(global_csv, "w", newline="", encoding="utf-8") as fglob:
        gw = csv.writer(fglob)
        gw.writerow(["metric", "value"])
        gw.writerow(["tp", G_tp])
        gw.writerow(["tn", G_tn])
        gw.writerow(["fp", G_fp])
        gw.writerow(["fn", G_fn])
        for k, v in G_metrics.items():
            gw.writerow([k, v])
        gw.writerow(["skipped_tiles", skipped])

    with open(cm_csv, "w", newline="", encoding="utf-8") as fcm:
        cmw = csv.writer(fcm)
        cmw.writerow(["", "PRED_not_burned", "PRED_burned"])
        cmw.writerow(["GT_not_burned", G_tn, G_fp])
        cmw.writerow(["GT_burned", G_fn, G_tp])

    summary = (
        f"[DONE] Evaluation complete.\n"
        f"  Ground truth directory: {gt_dir}\n"
        f"  PRED_DIR: {pred_dir}\n"
        f"  REPORT_DIR: {report_dir}\n"
        f"  Target class (burned): {target_class}\n"
        f"  Global precision (burned): {G_metrics['precision']:.4f}\n"
        f"  Global recall (burned): {G_metrics['recall']:.4f}\n"
        f"  Global F1 (burned): {G_metrics['f1_score']:.4f}\n"
        f"  Global IoU (burned): {G_metrics['iou_burned']:.4f}\n"
        f"  Tiles skipped (no pred / error / no valid pixels): {skipped}\n"
        f"  Outputs:\n"
        f"    - per_chip_metrics.csv\n"
        f"    - per_scene_metrics.csv\n"
        f"    - global_metrics.csv\n"
        f"    - confusion_matrix.csv\n"
    )

    print(summary)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Evaluation complete", summary)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()