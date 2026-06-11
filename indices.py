"""Vegetation and water index computation."""

import numpy as np


def compute_ndvi(nir, red):
    """Normalized Difference Vegetation Index.
    NDVI = (NIR - RED) / (NIR + RED)
    Range: -1 to 1. Healthy vegetation > 0.3
    """
    denom = nir + red + 1e-10
    return (nir - red) / denom


def compute_ndwi(green, nir):
    """Normalized Difference Water Index.
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    Range: -1 to 1. Water bodies > 0.
    """
    denom = green + nir + 1e-10
    return (green - nir) / denom


def compute_evi(nir, red, blue):
    """Enhanced Vegetation Index.
    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    Less sensitive to atmospheric effects than NDVI.
    """
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    return 2.5 * (nir - red) / (denom + 1e-10)


def describe_phenology(series):
    """Rule-based interpretation of an NDVI time series (list of points with
    'date' and 'ndvi_mean'). Returns a dict with peak info and a crop-type
    hint based on simple phenology heuristics for northern-hemisphere crops.
    """
    if len(series) < 2:
        return {"hint": "Not enough cloud-free scenes for a temporal profile."}

    values = [p["ndvi_mean"] for p in series]
    peak_idx = int(np.argmax(values))
    peak = series[peak_idx]
    peak_month = int(peak["date"][5:7])
    amplitude = round(max(values) - min(values), 3)

    if max(values) < 0.25:
        hint = "Consistently low NDVI — likely bare soil, fallow land, or water."
    elif amplitude < 0.15:
        hint = "High but flat NDVI — likely permanent vegetation (grassland or forest), not an annual crop."
    elif peak_month <= 6:
        hint = "Early-season peak (May–June) — consistent with winter crops (e.g. winter wheat)."
    elif peak_month <= 8:
        hint = "Mid-summer peak (July–August) — consistent with spring-sown crops (e.g. spring wheat, corn, sunflower)."
    else:
        hint = "Late-season peak — possibly a late crop or regrowth after harvest."

    return {
        "peak_date": peak["date"],
        "peak_ndvi": peak["ndvi_mean"],
        "amplitude": amplitude,
        "hint": hint,
    }


def compute_all_indices(bands):
    """Compute all vegetation indices from band dict.

    Args:
        bands: dict with keys 'blue', 'green', 'red', 'nir'
               each value is [H, W] float32 array

    Returns:
        dict of index_name -> {'array': [H,W], 'stats': {min, max, mean}}
    """
    results = {}

    ndvi = compute_ndvi(bands["nir"], bands["red"])
    results["ndvi"] = {
        "array": ndvi,
        "stats": {
            "min": float(np.nanmin(ndvi)),
            "max": float(np.nanmax(ndvi)),
            "mean": float(np.nanmean(ndvi)),
        },
        "colormap": "RdYlGn",
        "label": "NDVI",
    }

    ndwi = compute_ndwi(bands["green"], bands["nir"])
    results["ndwi"] = {
        "array": ndwi,
        "stats": {
            "min": float(np.nanmin(ndwi)),
            "max": float(np.nanmax(ndwi)),
            "mean": float(np.nanmean(ndwi)),
        },
        "colormap": "RdYlBu",
        "label": "NDWI",
    }

    evi = compute_evi(bands["nir"], bands["red"], bands["blue"])
    results["evi"] = {
        "array": evi,
        "stats": {
            "min": float(np.nanmin(evi)),
            "max": float(np.nanmax(evi)),
            "mean": float(np.nanmean(evi)),
        },
        "colormap": "RdYlGn",
        "label": "EVI",
    }

    return results
