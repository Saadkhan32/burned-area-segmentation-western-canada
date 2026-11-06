"""Shared helpers for the benchmark pipeline.
Band order of every window raster: Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NBR, NDMI (same as the original study).
Labels: 0 = unburned, 1 = burned (NBAC polygons of the fire year), 255 = ignore (prior-year burn scars, optional)."""
import os, json, math, hashlib, csv
try:
    import h5py  # noqa: F401 - Alliance clusters: h5py (built on HDF5 1.14.6) must be loaded before rasterio/GDAL (built on the hdf5/1.14.2
except Exception:  # module) so that the newer, backward-compatible libhdf5 is the one in memory; terratorch imports h5py and fails otherwise
    pass
try:
    import rasterio  # noqa: F401 - and rasterio (GDAL + the proj module's libproj) before pyproj: the wheelhouse pyproj otherwise loads a newer
except Exception:    # libproj whose database does not match PROJ_DATA ("proj_create: no database context specified")
    pass
import numpy as np
# reading public Sentinel-2 COGs over HTTPS/S3 with GDAL: anonymous access, no directory listings, retries on transient errors
for _k, _v in (('AWS_NO_SIGN_REQUEST', 'YES'), ('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR'), ('GDAL_HTTP_MAX_RETRY', '3'), ('GDAL_HTTP_RETRY_DELAY', '2'),
               ('VSI_CACHE', 'TRUE'), ('GDAL_CACHEMAX', '512')):
    os.environ.setdefault(_k, _v)

BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NBR', 'NDMI']
IGNORE = 255
PROVINCES = ['AB', 'BC', 'SK']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def read_manifest(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    for r in rows:
        assert r['province'] in PROVINCES, r
    return rows

def validation_set(rows, fold, validation_windows, extra_per_province=0, seed=42):
    """The reserved validation windows of the fold (pre-registered list) plus, when the dataset was enlarged with
    --per-province, extra_per_province windows per development province drawn with a fixed seed from the windows whose
    source is 'extra NBAC fire' (never from the held-out province, never a negative window). Deterministic, so train.py
    and eval.py reach the same set; train.py also records it in config.json."""
    held = fold.split('-')[1]
    val = [w for w in validation_windows.get(fold, [])]
    if extra_per_province:
        rng = np.random.default_rng(seed)
        for pv in PROVINCES:
            if pv == held: continue
            pool = sorted(r['window_id'] for r in rows if r['province'] == pv and r.get('source', '') == 'extra NBAC fire'
                          and r['window_id'] not in val and r.get('is_negative', '') not in ('True', 'true', '1'))
            if pool:
                pick = sorted(rng.choice(len(pool), min(extra_per_province, len(pool)), replace=False))
                val += [pool[i] for i in pick]
    return val


def fold_roles(rows, fold, validation_windows, extra_per_province=0, seed=42):
    """fold = 'F-AB' | 'F-BC' | 'F-SK'. Returns dict window_id -> TRAIN / VALIDATION / TEST.
    TEST = every window of the held-out province (fire and negative windows); VALIDATION = the reserved windows of the
    development provinces (validation_windows[fold], plus extra_per_province extra-NBAC windows per development province,
    see validation_set); TRAIN = the rest."""
    held = fold.split('-')[1]
    val = set(validation_windows) if isinstance(validation_windows, (list, set, tuple)) else set(validation_set(rows, fold, validation_windows, extra_per_province, seed))
    roles = {}
    for r in rows:
        if r['province'] == held:
            roles[r['window_id']] = 'TEST'
        elif r['window_id'] in val:
            roles[r['window_id']] = 'VALIDATION'
        else:
            roles[r['window_id']] = 'TRAIN'
    return roles

def read_window(data_dir, window_id):
    import rasterio
    with rasterio.open(os.path.join(data_dir, window_id, window_id + '_S2.tif')) as src:
        img = src.read().astype(np.float32)          # (9, H, W), NaN = masked
        profile = src.profile
    with rasterio.open(os.path.join(data_dir, window_id, window_id + '_label.tif')) as src:
        lab = src.read(1).astype(np.uint8)
    return img, lab, profile

def write_gtiff(path, arr, profile, dtype, nodata=None):
    import rasterio
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    p = dict(profile)
    p.update(driver='GTiff', count=arr.shape[0], dtype=dtype, compress='lzw', tiled=True, blockxsize=256, blockysize=256, nodata=nodata)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(dtype))

def indices(refl):
    """refl: (6, H, W) harmonised reflectance in band order Blue..SWIR2 -> (9, H, W) with NDVI, NBR, NDMI appended."""
    b, g, r, nir, sw1, sw2 = refl
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - r) / (nir + r)
        nbr = (nir - sw2) / (nir + sw2)
        ndmi = (nir - sw1) / (nir + sw1)
    return np.concatenate([refl, ndvi[None], nbr[None], ndmi[None]], 0).astype(np.float32)

def harmonise(dn, baseline, offset_applied, subtract=None):
    """Sentinel-2 L2A DN -> surface reflectance. Products with processing baseline >= 04.00 carry BOA_ADD_OFFSET = -1000
    unless the catalogue already applied it (Earth Search flags this). `subtract` overrides that rule when given
    (see offset_in_dn: the catalogue flag was wrong for three 2022 scenes). Returns float32 reflectance, NaN where DN == 0."""
    dn = dn.astype(np.float32)
    out = np.where(dn > 0, dn, np.nan)
    if subtract is None:
        subtract = baseline >= 4.0 and not offset_applied
    if subtract:
        out = out - 1000.0
    out = out / 10000.0
    return np.clip(out, 0.0, None)

OFFSET_DN_P1 = 800.0   # darkest clear pixels of the blue band: DN ~100-400 without the offset, ~1000-1400 with it

def offset_in_dn(blue_dn, clear):
    """Data check for the BOA offset: the 1st percentile of the blue-band DN over clear pixels (and whether it is at or above
    OFFSET_DN_P1, i.e. the offset is present). Checked on the dataset: three 2022 scenes (baseline 04.00) flagged
    'earthsearch:boa_offset_applied = False' hold DN without the offset; subtracting 1000 clipped half the scene to zero
    reflectance and made NDVI/NBR/NDMI NaN over 85 % of the burned pixels. Returns (p1, present) - p1 is NaN (present=None)
    when fewer than 1,000 clear pixels are available."""
    v = blue_dn[clear & (blue_dn > 0)]
    if v.size < 1000:
        return float('nan'), None
    p1 = float(np.percentile(v, 1))
    return p1, p1 >= OFFSET_DN_P1

def confusion(pred, ref, valid):
    """pred, ref: arrays with 1 = burned; valid: boolean. Returns tp, fp, fn, tn (ints)."""
    p = (pred == 1) & valid; q = (ref == 1) & valid
    tp = int((p & q).sum()); fp = int((p & ~q & valid).sum()); fn = int((~p & q).sum()); tn = int((~p & ~q & valid).sum())
    return tp, fp, fn, tn

def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else float('nan')
    R = tp / (tp + fn) if tp + fn else float('nan')
    F = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float('nan')
    I = tp / (tp + fp + fn) if tp + fp + fn else float('nan')
    return P, R, F, I

def otsu_threshold(values, nbins=256):
    """Otsu's threshold (maximum between-class variance) of a 1-D sample; NaN if fewer than 100 finite values."""
    v = values[np.isfinite(values)]
    if v.size < 100:
        return float('nan')
    hist, edges = np.histogram(v, bins=nbins)
    p = hist.astype(np.float64) / max(hist.sum(), 1); centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1.0 - w0
    cum_mu = np.cumsum(p * centres); mu_total = cum_mu[-1]
    mu0 = cum_mu / np.maximum(w0, 1e-12); mu1 = (mu_total - cum_mu) / np.maximum(w1, 1e-12)
    var = w0 * w1 * (mu0 - mu1) ** 2
    v = var[:-1]; top = np.flatnonzero(v >= np.nanmax(v) * (1 - 1e-6))   # plateau when the modes are separated by empty bins
    k = int(top[len(top) // 2])
    return float(edges[k + 1])

def sliding_logits(predict_fn, img, tile=256, stride=128, n_classes=2, batch=16):
    """Average class logits over overlapping tiles so that every pixel is scored once (weighted mean).
    predict_fn: np.ndarray (B, 9, tile, tile) -> np.ndarray (B, n_classes, tile, tile). NaNs are filled with 0 before prediction;
    the caller masks them afterwards."""
    C, H, W = img.shape
    out = np.zeros((n_classes, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H - tile, 0) + 1, stride)); xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] != max(H - tile, 0): ys.append(max(H - tile, 0))
    if xs[-1] != max(W - tile, 0): xs.append(max(W - tile, 0))
    coords = [(y, x) for y in ys for x in xs]
    filled = np.nan_to_num(img, nan=0.0)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = np.stack([np.pad(filled[:, y:y + tile, x:x + tile], ((0, 0), (0, max(0, tile - min(tile, H - y))), (0, max(0, tile - min(tile, W - x))))) for y, x in chunk])
        lg = predict_fn(tiles)
        for (y, x), l in zip(chunk, lg):
            h = min(tile, H - y); w = min(tile, W - x)
            out[:, y:y + h, x:x + w] += l[:, :h, :w]; cnt[y:y + h, x:x + w] += 1
    return out / np.maximum(cnt, 1)[None]

def window_grid(lon, lat, radius_m=10000.0, res=10.0):
    """Target grid for a window: UTM zone of the centre, bounds of a geodesic circle of radius_m (as the GEE script did),
    snapped outwards to res. Returns (epsg, transform, width, height)."""
    from pyproj import CRS, Transformer
    from rasterio.transform import from_origin
    zone = int(math.floor((lon + 180) / 6) + 1); epsg = 32600 + zone
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    # circle bounds in geographic coordinates (approximation of ee.Geometry.buffer().bounds())
    dlat = radius_m / 110574.0; dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    xs, ys = tr.transform([lon - dlon, lon + dlon, lon - dlon, lon + dlon], [lat - dlat, lat - dlat, lat + dlat, lat + dlat])
    x0 = math.floor(min(xs) / res) * res; x1 = math.ceil(max(xs) / res) * res
    y0 = math.floor(min(ys) / res) * res; y1 = math.ceil(max(ys) / res) * res
    width = int(round((x1 - x0) / res)); height = int(round((y1 - y0) / res))
    return epsg, from_origin(x0, y1, res, res), width, height
