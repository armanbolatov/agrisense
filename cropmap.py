"""Seasonal per-pixel crop classification from multi-temporal Sentinel-2.

This is a lightweight, reproducible alternative to the tabular EU-CropMap model:
instead of GEE-engineered features, it classifies each pixel from its NDVI
phenology profile across the growing season. Classes use the EU-CropMap
Level-2 naming so results line up with that scheme.

Pipeline:
  download_monthly_ndvi_stack -> [T, H, W] NDVI  (one scene per month)
  classify_seasonal           -> [H, W] class ids + legend + area %
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from satellite import search_sentinel2, read_band
from indices import compute_ndvi, compute_ndwi


# EU-CropMap-style classes used by the seasonal classifier. (id, label, hex)
CROP_CLASSES = [
    (0, "Water",                "#1f6feb"),  # 700
    (1, "Bare / fallow",        "#b08968"),  # 290 / 600
    (2, "Grassland",            "#7bc96f"),  # 500
    (3, "Winter cereals",       "#d4a017"),  # 211–214 (early peak)
    (4, "Summer crops",         "#e8590c"),  # 216 / 231 (mid-summer peak)
    (5, "Late / forage crops",  "#9d4edd"),  # 250 (late peak)
]
CLASS_COLORS = np.array([
    [31, 111, 235],
    [176, 137, 104],
    [123, 201, 111],
    [212, 160, 23],
    [232, 89, 12],
    [157, 78, 221],
], dtype=np.uint8)


def download_monthly_ndvi_stack(bbox, date_windows, size=256, on_progress=None):
    """Download one least-cloudy scene per window and return per-pixel NDVI.

    Returns:
        months:    list of 'YYYY-MM' labels for windows that had a scene
        ndvi_stack: [T, H, W] float32 NDVI (T = len(months))
        ndwi_mean:  [H, W] mean NDWI (used to detect water)
    """
    target = (size, size)

    def fetch(window):
        item = search_sentinel2(bbox, window)
        if item is None:
            return None
        with ThreadPoolExecutor(max_workers=3) as pool:
            (green, _, _), (red, _, _), (nir, _, _) = pool.map(
                lambda spec: read_band(item, spec[0], bbox, spec[1], target_size=target),
                [("B03", 10), ("B04", 10), ("B8A", 20)],
            )
        ndvi = compute_ndvi(nir, red)
        ndwi = compute_ndwi(green, nir)
        invalid = (red == 0) & (nir == 0)
        ndvi[invalid] = np.nan
        ndwi[invalid] = np.nan
        label = item.properties.get("datetime", window)[:7]  # YYYY-MM
        return label, ndvi.astype(np.float32), ndwi.astype(np.float32)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, date_windows))

    months, ndvi_list, ndwi_list = [], [], []
    for window, res in zip(date_windows, results):
        if on_progress:
            on_progress(f"{window}: {'ok' if res else 'no scene'}")
        if res:
            months.append(res[0])
            ndvi_list.append(res[1])
            ndwi_list.append(res[2])

    if not ndvi_list:
        raise ValueError("No cloud-free scenes found in any month.")

    ndvi_stack = np.stack(ndvi_list, axis=0)
    ndwi_mean = np.nanmean(np.stack(ndwi_list, axis=0), axis=0)
    return months, ndvi_stack, ndwi_mean


def classify_seasonal(months, ndvi_stack, ndwi_mean):
    """Rule-based per-pixel classification from the NDVI phenology profile.

    Returns:
        class_map: [H, W] int8 class ids (-1 = nodata)
        areas:     list of {id, label, color, percent}
    """
    T, H, W = ndvi_stack.shape
    # Per-pixel temporal stats (nan-safe).
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(ndvi_stack, axis=0)
        trough = np.nanmin(ndvi_stack, axis=0)
        mean = np.nanmean(ndvi_stack, axis=0)
        peak_idx = np.nanargmax(np.nan_to_num(ndvi_stack, nan=-2), axis=0)
    amplitude = peak - trough

    # Month index of each window -> calendar month number for the peak.
    month_nums = np.array([int(m[5:7]) if len(m) >= 7 else 7 for m in months])
    peak_month = month_nums[np.clip(peak_idx, 0, T - 1)]

    cls = np.full((H, W), -1, dtype=np.int8)
    valid = ~np.isnan(mean)

    water = valid & (ndwi_mean > 0.0) & (peak < 0.2)
    bare = valid & ~water & (peak < 0.25)
    grass = valid & ~water & ~bare & (amplitude < 0.18)
    rest = valid & ~water & ~bare & ~grass

    winter = rest & (peak_month <= 6)
    summer = rest & (peak_month >= 7) & (peak_month <= 8)
    late = rest & (peak_month >= 9)

    cls[water] = 0
    cls[bare] = 1
    cls[grass] = 2
    cls[winter] = 3
    cls[summer] = 4
    cls[late] = 5

    total = int(valid.sum()) or 1
    areas = []
    for cid, label, color in CROP_CLASSES:
        pct = round(100.0 * int((cls == cid).sum()) / total, 1)
        areas.append({"id": cid, "label": label, "color": color, "percent": pct})

    return cls, areas


def class_map_to_png(class_map):
    """Render an int class map to a base64 RGBA PNG (nodata transparent)."""
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = class_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cid in range(len(CROP_CLASSES)):
        m = class_map == cid
        rgba[m, :3] = CLASS_COLORS[cid]
        rgba[m, 3] = 255

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
