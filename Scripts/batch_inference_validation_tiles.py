import os
import arcpy
import tkinter as tk
from tkinter import filedialog, messagebox

arcpy.env.overwriteOutput = True


def pick_folder(title, must_exist=True):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if not folder:
        raise SystemExit(f"Cancelled: {title}")
    folder = os.path.normpath(folder)
    if not os.path.exists(folder):
        if must_exist:
            raise SystemExit(f"Folder does not exist: {folder}")
        os.makedirs(folder, exist_ok=True)
    return folder


def pick_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise SystemExit(f"Cancelled: {title}")
    return os.path.normpath(path)


def main():
    val_images = pick_folder("Select folder with VALIDATION IMAGES (val_images)")

    val_predictions = pick_folder(
        "Select/CREATE folder for PREDICTION RASTERS (val_predictions)",
        must_exist=False
    )

    model_emd = pick_file(
        "Select Deep Learning MODEL definition (.emd)",
        filetypes=[("Esri Model Definition", "*.emd"), ("All files", "*.*")]
    )

    valid_ext = {".tif", ".tiff", ".png"}

    tiles = [
        f for f in os.listdir(val_images)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    if not tiles:
        raise SystemExit(f"No image tiles (.tif/.tiff/.png) found in {val_images}")

    print(f"Found {len(tiles)} tiles to classify.")
    arcpy.AddMessage(f"Found {len(tiles)} tiles to classify.")

    for i, fname in enumerate(sorted(tiles), start=1):
        in_ras = os.path.join(val_images, fname)
        base, ext = os.path.splitext(fname)
        out_ras = os.path.join(val_predictions, base + "_Classified" + ext)

        msg = f"[{i}/{len(tiles)}] Classifying {fname} -> {os.path.basename(out_ras)}"
        print(msg)
        arcpy.AddMessage(msg)

        try:
            classified = arcpy.ia.ClassifyPixelsUsingDeepLearning(
                in_ras,
                model_emd,
                "",
                "PROCESS_AS_MOSAICKED_IMAGE"
            )

            classified.save(out_ras)

        except arcpy.ExecuteError:
            err = arcpy.GetMessages(2)
            print("ArcPy error:", err)
            arcpy.AddError(err)
            continue
        except Exception as e:
            print("Python error:", e)
            arcpy.AddError(str(e))
            continue

    final_msg = (
        f"Done. Classified {len(tiles)} tiles.\n"
        f"Prediction rasters saved in:\n{val_predictions}"
    )
    print(final_msg)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Batch classification complete", final_msg)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()