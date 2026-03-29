import arcpy, os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

arcpy.env.overwriteOutput = True

FIELD_ID = "ClassID"
FIELD_NAME = "ClassName"

UNBURNED_ID = 1
BURNED_ID = 2
UNBURNED_NAME = "Unburned Area"
BURNED_NAME = "Burned Area"

BURNED_KEYWORDS = ("burned", "burn scar", "burnscar", "burnt")
UNBURNED_BLOCK = ("unburned",)

root = tk.Tk()
root.withdraw()

def pick_input_polygons():
    messagebox.showinfo(
        "Select input polygons",
        "Pick your polygons feature.\n\n"
        "- If shapefile: choose .shp\n"
        "- If GDB feature class: Cancel then select .gdb folder and type feature class name"
    )
    shp = filedialog.askopenfilename(
        title="Select .shp (or Cancel to choose a GDB feature class)",
        filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")]
    )
    if shp and shp.lower().endswith(".shp"):
        return shp

    gdb = filedialog.askdirectory(title="Select input file geodatabase (.gdb)")
    if not gdb:
        raise SystemExit("No input selected.")
    if not gdb.lower().endswith(".gdb"):
        raise SystemExit("Selected folder is not a .gdb.")

    fc_name = simpledialog.askstring("Feature class name", "Enter the feature class name inside the .gdb:")
    if not fc_name:
        raise SystemExit("No feature class name provided.")
    return os.path.join(gdb, fc_name)

def pick_reference_raster():
    messagebox.showinfo(
        "Select reference raster",
        "Select your reference raster (e.g., Sentinel-2 stack).\n"
        "Used for snap/extent/cellsize so label raster aligns perfectly."
    )
    ras = filedialog.askopenfilename(
        title="Select reference raster",
        filetypes=[("Raster", "*.tif;*.tiff;*.img;*.jp2;*.crf"), ("All files", "*.*")]
    )
    if ras and arcpy.Exists(ras):
        return ras

    gdb = filedialog.askdirectory(title="Or select a .gdb that contains the raster")
    if not gdb:
        raise SystemExit("No reference raster selected.")
    if not gdb.lower().endswith(".gdb"):
        raise SystemExit("Selected folder is not a .gdb.")

    ras_name = simpledialog.askstring("Raster name", "Enter raster dataset name inside the .gdb:")
    if not ras_name:
        raise SystemExit("No raster name provided.")
    ras = os.path.join(gdb, ras_name)
    if not arcpy.Exists(ras):
        raise SystemExit(f"Reference raster not found: {ras}")
    return ras

def ask_output_polygons(default_name="Merged_Binary_Burned_Unburned"):
    choice = messagebox.askyesno(
        "Save binary polygons",
        "Save NEW binary polygons into a File Geodatabase (.gdb)?\n\nYES = GDB\nNO = Shapefile"
    )
    if choice:
        gdb = filedialog.askdirectory(title="Select output .gdb for polygons")
        if not gdb or not gdb.lower().endswith(".gdb"):
            raise SystemExit("Please select a valid .gdb folder.")
        fc_name = simpledialog.askstring(
            "Output feature class name",
            "Enter output feature class name:",
            initialvalue=default_name
        )
        if not fc_name:
            raise SystemExit("No output feature class name provided.")
        return os.path.join(gdb, fc_name)

    out_shp = filedialog.asksaveasfilename(
        title="Save output shapefile as",
        defaultextension=".shp",
        initialfile=f"{default_name}.shp",
        filetypes=[("Shapefile", "*.shp")]
    )
    if not out_shp:
        raise SystemExit("No output shapefile chosen.")
    if not out_shp.lower().endswith(".shp"):
        out_shp += ".shp"
    return out_shp

def ask_output_label_tif(default_name="Label_Binary_Burned_Unburned.tif"):
    messagebox.showinfo(
        "Save label raster",
        "Choose where to save the LABEL raster (GeoTIFF).\n"
        "This script will force it to integer (thematic) and build the attribute table."
    )
    out_tif = filedialog.asksaveasfilename(
        title="Save label raster as",
        defaultextension=".tif",
        initialfile=default_name,
        filetypes=[("GeoTIFF", "*.tif;*.tiff")]
    )
    if not out_tif:
        raise SystemExit("No output label raster chosen.")
    if not (out_tif.lower().endswith(".tif") or out_tif.lower().endswith(".tiff")):
        out_tif += ".tif"
    return out_tif

def ensure_fields_exist(fc):
    existing = {f.name for f in arcpy.ListFields(fc)}
    missing = [fld for fld in (FIELD_ID, FIELD_NAME) if fld not in existing]
    if missing:
        raise ValueError(f"Required fields missing in attribute table: {missing}")

def make_working_copy(in_fc):
    scratch_gdb = arcpy.env.scratchGDB
    if not scratch_gdb or not arcpy.Exists(scratch_gdb):
        raise RuntimeError("scratchGDB is not available. Run inside ArcGIS Pro.")

    base = os.path.splitext(os.path.basename(in_fc))[0]
    safe = arcpy.ValidateTableName(base + "_work", scratch_gdb)
    work_fc = os.path.join(scratch_gdb, safe)

    if arcpy.Exists(work_fc):
        arcpy.management.Delete(work_fc)

    arcpy.management.CopyFeatures(in_fc, work_fc)

    if not arcpy.Exists(work_fc):
        raise RuntimeError(f"Working copy was not created: {work_fc}")

    return work_fc

def get_classid_numeric_field(fc):
    f = [x for x in arcpy.ListFields(fc) if x.name == FIELD_ID][0]
    if f.type in ("SmallInteger", "Integer"):
        return FIELD_ID

    new_field = "ClassID_num"
    existing = {x.name for x in arcpy.ListFields(fc)}
    if new_field not in existing:
        arcpy.management.AddField(fc, new_field, "SHORT")

    return new_field

def is_burned_by_name(class_name):
    nm = "" if class_name is None else str(class_name).strip().lower()
    if any(b in nm for b in UNBURNED_BLOCK):
        return False
    return any(k in nm for k in BURNED_KEYWORDS) or (nm == "burned area")

def merge_to_binary_classes(fc):
    classid_value_field = get_classid_numeric_field(fc)

    burned = 0
    unburned = 0
    changed = 0

    fields = [FIELD_NAME, FIELD_ID, classid_value_field]
    if classid_value_field == FIELD_ID:
        fields = [FIELD_NAME, FIELD_ID]

    with arcpy.da.UpdateCursor(fc, fields) as cur:
        for row in cur:
            if classid_value_field == FIELD_ID:
                name, cid = row
                cid_num = cid
            else:
                name, cid, cid_num = row

            burned_flag = is_burned_by_name(name)

            new_name = BURNED_NAME if burned_flag else UNBURNED_NAME
            new_id = BURNED_ID if burned_flag else UNBURNED_ID

            if burned_flag:
                burned += 1
            else:
                unburned += 1

            need_update = (str(name) != new_name) or (str(cid) != str(new_id))
            if classid_value_field != FIELD_ID:
                try:
                    need_update = need_update or (int(cid_num) != int(new_id))
                except Exception:
                    need_update = True

            if need_update:
                if classid_value_field == FIELD_ID:
                    cur.updateRow([new_name, new_id])
                else:
                    cur.updateRow([new_name, str(new_id), int(new_id)])
                changed += 1

    return burned, unburned, changed, classid_value_field

def save_polygons(fc, out_path):
    if ".gdb" in out_path.lower():
        idx = out_path.lower().rfind(".gdb") + 4
        gdb = out_path[:idx]
        fc_name = out_path[idx+1:] if len(out_path) > idx+1 else None
        if not os.path.exists(gdb) or not fc_name:
            raise ValueError("Invalid GDB output path for polygons.")
        dest = os.path.join(gdb, fc_name)
        if arcpy.Exists(dest):
            arcpy.management.Delete(dest)
        arcpy.management.CopyFeatures(fc, dest)
        return dest

    folder = os.path.dirname(out_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if arcpy.Exists(out_path):
        arcpy.management.Delete(out_path)
    arcpy.management.CopyFeatures(fc, out_path)
    return out_path

def create_aoi_unburned_polygon_from_raster(ref_raster, out_fc, value_field):
    desc = arcpy.Describe(ref_raster)
    sr = desc.spatialReference
    ext = desc.extent

    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)

    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(out_fc),
        out_name=os.path.basename(out_fc),
        geometry_type="POLYGON",
        spatial_reference=sr
    )
    arcpy.management.AddField(out_fc, value_field, "SHORT")
    arcpy.management.AddField(out_fc, FIELD_NAME, "TEXT", field_length=64)

    arr = arcpy.Array([
        arcpy.Point(ext.XMin, ext.YMin),
        arcpy.Point(ext.XMin, ext.YMax),
        arcpy.Point(ext.XMax, ext.YMax),
        arcpy.Point(ext.XMax, ext.YMin),
        arcpy.Point(ext.XMin, ext.YMin)
    ])
    poly = arcpy.Polygon(arr, sr)

    with arcpy.da.InsertCursor(out_fc, ["SHAPE@", value_field, FIELD_NAME]) as ic:
        ic.insertRow([poly, UNBURNED_ID, UNBURNED_NAME])

    return out_fc

def build_thematic_label_tif(binary_fc, value_field, ref_raster, out_tif):
    scratch_gdb = arcpy.env.scratchGDB
    if not scratch_gdb or not arcpy.Exists(scratch_gdb):
        raise RuntimeError("scratchGDB is not available. Run inside ArcGIS Pro.")

    old_env = (arcpy.env.snapRaster, arcpy.env.extent, arcpy.env.cellSize, arcpy.env.outputCoordinateSystem)
    arcpy.env.snapRaster = ref_raster
    arcpy.env.extent = ref_raster
    arcpy.env.cellSize = ref_raster
    try:
        arcpy.env.outputCoordinateSystem = arcpy.Describe(ref_raster).spatialReference
    except Exception:
        arcpy.env.outputCoordinateSystem = ref_raster

    try:
        aoi_fc = os.path.join(scratch_gdb, arcpy.ValidateTableName("tmp_aoi_unburned", scratch_gdb))
        burned_fc = os.path.join(scratch_gdb, arcpy.ValidateTableName("tmp_burned_only", scratch_gdb))
        labels_fc = os.path.join(scratch_gdb, arcpy.ValidateTableName("tmp_labels_fc", scratch_gdb))
        tmp_ras = os.path.join(scratch_gdb, arcpy.ValidateTableName("tmp_labels_ras", scratch_gdb))
        tmp_u8_ras = os.path.join(scratch_gdb, arcpy.ValidateTableName("tmp_labels_u8", scratch_gdb))

        for p in (aoi_fc, burned_fc, labels_fc, tmp_ras, tmp_u8_ras):
            if arcpy.Exists(p):
                arcpy.management.Delete(p)
        if arcpy.Exists(out_tif):
            arcpy.management.Delete(out_tif)

        create_aoi_unburned_polygon_from_raster(ref_raster, aoi_fc, value_field)

        where = f"{arcpy.AddFieldDelimiters(binary_fc, value_field)} = {BURNED_ID}"
        arcpy.analysis.Select(binary_fc, burned_fc, where_clause=where)

        arcpy.management.Merge([aoi_fc, burned_fc], labels_fc)

        cellsize = float(arcpy.management.GetRasterProperties(ref_raster, "CELLSIZEX").getOutput(0))
        arcpy.conversion.PolygonToRaster(
            in_features=labels_fc,
            value_field=value_field,
            out_rasterdataset=tmp_ras,
            cell_assignment="MAXIMUM_AREA",
            priority_field=value_field,
            cellsize=cellsize
        )

        arcpy.management.CopyRaster(
            in_raster=tmp_ras,
            out_rasterdataset=tmp_u8_ras,
            pixel_type="8_BIT_UNSIGNED"
        )

        arcpy.management.BuildRasterAttributeTable(tmp_u8_ras, "Overwrite")
        try:
            arcpy.management.CalculateStatistics(tmp_u8_ras)
        except Exception:
            pass

        arcpy.management.CopyRaster(
            in_raster=tmp_u8_ras,
            out_rasterdataset=out_tif,
            pixel_type="8_BIT_UNSIGNED"
        )

        try:
            arcpy.management.BuildRasterAttributeTable(out_tif, "Overwrite")
        except Exception:
            pass
        try:
            arcpy.management.CalculateStatistics(out_tif)
        except Exception:
            pass

        for p in (aoi_fc, burned_fc, labels_fc, tmp_ras, tmp_u8_ras):
            try:
                if arcpy.Exists(p):
                    arcpy.management.Delete(p)
            except Exception:
                pass

        return out_tif

    finally:
        arcpy.env.snapRaster, arcpy.env.extent, arcpy.env.cellSize, arcpy.env.outputCoordinateSystem = old_env

def main():
    in_fc = pick_input_polygons()
    if not arcpy.Exists(in_fc):
        raise SystemExit(f"Input not found: {in_fc}")

    ensure_fields_exist(in_fc)

    ref_raster = pick_reference_raster()
    if not arcpy.Exists(ref_raster):
        raise SystemExit(f"Reference raster not found: {ref_raster}")

    work_fc = make_working_copy(in_fc)

    burned, unburned, changed, value_field = merge_to_binary_classes(work_fc)

    msg_merge = (
        f"[INFO] Merge complete\n\n"
        f"Burned (forced to ClassID=2): {burned}\n"
        f"Unburned (forced to ClassID=1): {unburned}\n"
        f"Rows updated: {changed}\n\n"
        f"Burned detection: ClassName contains {BURNED_KEYWORDS} (and not {UNBURNED_BLOCK})\n"
        f"Raster value field used: {value_field}"
    )
    arcpy.AddMessage(msg_merge)
    messagebox.showinfo("Merge summary", msg_merge)

    out_poly = ask_output_polygons()
    final_fc = save_polygons(work_fc, out_poly)

    messagebox.showinfo(
        "Polygons saved",
        f"Saved binary polygons:\n{final_fc}\n\nBurned=2, Unburned=1"
    )

    out_tif = ask_output_label_tif()
    out_tif = build_thematic_label_tif(final_fc, value_field, ref_raster, out_tif)

    messagebox.showinfo(
        "Label raster saved",
        f"Saved THEMATIC label raster:\n{out_tif}\n\nValues: Unburned=1, Burned=2"
    )
    arcpy.AddMessage(f"[DONE] Label raster: {out_tif}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        arcpy.AddWarning(str(e))
    except Exception as ex:
        arcpy.AddError(str(ex))
        messagebox.showerror("Error", str(ex))
        raise