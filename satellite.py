"""Sentinel-2 data access via Microsoft Planetary Computer.

Adapted from /home/arman/terramind/run_crop_kz.py
"""

import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from pyproj import Transformer

# Scene search and band reads are network I/O against Planetary Computer and
# dominate every analysis. Both are memoized, so one area+date is fetched once
# and reused across tools (preview, NDVI, SAM, crop map, change detection).
# Signed asset URLs expire after ~1 h, so entries are dropped after _TTL.
_TTL = 1800
_BAND_CACHE_MAX = 32

# Native reads are capped to this many pixels on the long edge. Map overlays and
# SAM (which resizes to 1024 internally) gain nothing from a full 10 m raster,
# and rasterio serves the smaller read from the COG's overviews instead of
# fetching every tile. Set AGRISENSE_MAX_PX=0 for full native resolution.
MAX_PIXELS = int(os.environ.get("AGRISENSE_MAX_PX", "1024"))

_catalog = None
_scene_cache: dict = {}
_band_cache: OrderedDict = OrderedDict()
_cache_lock = Lock()


def clear_caches():
    """Drop every cached scene and band (used by bench.py)."""
    with _cache_lock:
        _scene_cache.clear()
        _band_cache.clear()


def _cache_key_bbox(bbox):
    return tuple(round(float(v), 4) for v in bbox)


def _get_catalog():
    """One signed STAC client per process — opening it costs a round trip."""
    global _catalog
    if _catalog is None:
        import pystac_client
        import planetary_computer
        _catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    return _catalog


def _ensure_size(array, target_size):
    """Crop or pad array to exactly match target_size (H, W)."""
    h, w = array.shape
    th, tw = target_size
    if h == th and w == tw:
        return array
    # Crop if larger
    array = array[:min(h, th), :min(w, tw)]
    # Pad if smaller
    if array.shape[0] < th or array.shape[1] < tw:
        padded = np.zeros(target_size, dtype=array.dtype)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded
    return array


def search_sentinel2(bbox, date_range, max_cloud=20):
    """Search Planetary Computer for best cloud-free Sentinel-2 scene.

    Args:
        bbox: [west, south, east, north] in lon/lat
        date_range: string like "2023-07-01/2023-07-31"
        max_cloud: maximum cloud cover percentage

    Returns:
        pystac Item or None
    """
    key = (_cache_key_bbox(bbox), date_range, max_cloud)
    with _cache_lock:
        hit = _scene_cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]

    results = _get_catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=3,
    )
    items = list(results.items())
    item = items[0] if items else None
    with _cache_lock:
        _scene_cache[key] = (time.time(), item)
    return item


def read_band(item, asset_name, bbox_lonlat, native_res, target_size=None):
    """Read a single band from a COG asset, cropped to bbox.

    Args:
        item: pystac Item
        asset_name: e.g. "B02", "B8A"
        bbox_lonlat: [west, south, east, north] in EPSG:4326
        native_res: native resolution in meters
        target_size: (H, W) tuple to resample to (for 20m bands → 10m)

    Returns:
        np.ndarray [H, W] float32, rasterio transform, CRS string
    """
    key = (item.id, asset_name, _cache_key_bbox(bbox_lonlat),
           tuple(target_size) if target_size else None)
    with _cache_lock:
        hit = _band_cache.get(key)
        if hit and time.time() - hit[0] < _TTL:
            _band_cache.move_to_end(key)
            return hit[1]

    href = item.assets[asset_name].href
    with rasterio.open(href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        west, south = transformer.transform(bbox_lonlat[0], bbox_lonlat[1])
        east, north = transformer.transform(bbox_lonlat[2], bbox_lonlat[3])
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().round_lengths()

        if target_size is None and MAX_PIXELS:
            longest = max(window.height, window.width)
            if longest > MAX_PIXELS:
                scale = MAX_PIXELS / longest
                target_size = (max(1, round(window.height * scale)),
                               max(1, round(window.width * scale)))

        if target_size:
            data = src.read(1, window=window, out_shape=target_size, resampling=Resampling.bilinear)
        else:
            data = src.read(1, window=window)

        # Compute the transform for the read window
        win_transform = src.window_transform(window)
        crs = str(src.crs)

    # Cached arrays are shared between callers, so treat them as read-only.
    out = (data.astype(np.float32), win_transform, crs)
    with _cache_lock:
        _band_cache[key] = (time.time(), out)
        while len(_band_cache) > _BAND_CACHE_MAX:
            _band_cache.popitem(last=False)
    return out


def download_rgb(bbox, date_range, on_progress=None):
    """Download RGB satellite image for SAM or preview.

    Args:
        bbox: [west, south, east, north]
        date_range: single date range string

    Returns:
        rgb: np.ndarray [H, W, 3] uint8 (0-255)
        geo_info: dict with transform, crs, bbox, shape
    """
    if on_progress:
        on_progress(f"Searching satellite imagery for {date_range}...")

    item = search_sentinel2(bbox, date_range)
    if item is None:
        raise ValueError(f"No cloud-free image found for {date_range}")

    if on_progress:
        cloud = item.properties.get("eo:cloud_cover", "?")
        on_progress(f"Found scene (cloud: {cloud:.1f}%), downloading RGB...")

    # Read B04 (Red), B03 (Green), B02 (Blue) — all native 10 m, so they share
    # a grid. Read them concurrently (rasterio releases the GIL during I/O).
    with ThreadPoolExecutor(max_workers=3) as pool:
        (red, _, _), (green, _, _), (blue, win_transform, crs) = pool.map(
            lambda b: read_band(item, b, bbox, 10), ["B04", "B03", "B02"]
        )

    # Ensure consistent sizes (use red as the reference)
    target = red.shape
    green = _ensure_size(green, target)
    blue = _ensure_size(blue, target)

    # Stack and normalize to 0-255
    rgb = np.stack([red, green, blue], axis=-1)
    valid = rgb[rgb > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
    else:
        p2, p98 = 0, 1
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)

    h, w = red.shape
    lonlat_transform = transform_from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    geo_info = {
        "transform": lonlat_transform,
        "native_transform": win_transform,
        "crs": crs,
        "lonlat_crs": "EPSG:4326",
        "bbox": bbox,
        "shape": (h, w),
    }

    return rgb, geo_info


def download_ndvi_series(bbox, date_windows, size=384, on_progress=None):
    """Download a lightweight NDVI/NDWI/EVI time series across date windows.

    For each window, picks the least-cloudy Sentinel-2 scene and reads B02/B03/
    B04/B8A downsampled to `size`x`size` — enough for area statistics.

    Returns:
        list of points: {date, cloud_cover, ndvi_mean/median/p25/p75/max,
                         ndwi_mean, evi_mean}  (windows with no scene skipped)
    """
    from indices import compute_ndvi, compute_ndwi, compute_evi

    series = []
    target = (size, size)

    def fetch_window(window):
        item = search_sentinel2(bbox, window)
        if item is None:
            return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            (blue, _, _), (green, _, _), (red, _, _), (nir, _, _) = pool.map(
                lambda spec: read_band(item, spec[0], bbox, spec[1], target_size=target),
                [("B02", 10), ("B03", 10), ("B04", 10), ("B8A", 20)],
            )
        valid = (red > 0) & (nir > 0)
        if not valid.any():
            return None
        ndvi = compute_ndvi(nir, red)[valid]
        ndwi = compute_ndwi(green, nir)[valid]
        evi = np.clip(compute_evi(nir, red, blue)[valid], -1, 1)
        return {
            "date": item.properties.get("datetime", window)[:10],
            "cloud_cover": round(float(item.properties.get("eo:cloud_cover", -1)), 1),
            "ndvi_mean": round(float(np.mean(ndvi)), 3),
            "ndvi_median": round(float(np.median(ndvi)), 3),
            "ndvi_p25": round(float(np.percentile(ndvi, 25)), 3),
            "ndvi_p75": round(float(np.percentile(ndvi, 75)), 3),
            "ndvi_max": round(float(np.percentile(ndvi, 98)), 3),
            "ndwi_mean": round(float(np.mean(ndwi)), 3),
            "evi_mean": round(float(np.mean(evi)), 3),
        }

    # Windows are independent — fetch them concurrently.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch_window, date_windows))

    for window, point in zip(date_windows, results):
        if on_progress:
            on_progress(f"{window}: {'ok' if point else 'no scene'}")
        if point:
            series.append(point)

    series.sort(key=lambda p: p["date"])
    return series


_TEN_M_ASSETS = {"blue": "B02", "green": "B03", "red": "B04"}


def download_bands_for_indices(bbox, date_range, on_progress=None,
                               bands=("blue", "green", "red")):
    """Download bands needed for vegetation indices.

    Args:
        bands: which 10 m visible bands to fetch. NIR (B8A) is always fetched.
               Change detection needs only ("red",) — half the downloads.

    Returns:
        bands: dict of band_name -> np.ndarray [H, W]
        geo_info: dict
    """
    if on_progress:
        on_progress("Downloading bands for indices...")

    item = search_sentinel2(bbox, date_range)
    if item is None:
        raise ValueError(f"No cloud-free image found for {date_range}")

    # B02/B03/B04 are native 10 m (shared grid) — read concurrently.
    wanted = [b for b in ("blue", "green", "red") if b in bands]
    with ThreadPoolExecutor(max_workers=3) as pool:
        reads = list(pool.map(
            lambda b: read_band(item, _TEN_M_ASSETS[b], bbox, 10), wanted))
    out = dict(zip(wanted, (r[0] for r in reads)))

    # B8A is native 20 m — resample it onto the 10 m grid.
    target_size = reads[0][0].shape if reads else None
    nir, win_transform, crs = read_band(item, "B8A", bbox, 20, target_size=target_size)
    out["nir"] = nir
    target_size = target_size or nir.shape

    out = {name: _ensure_size(arr, target_size) for name, arr in out.items()}

    h, w = target_size
    lonlat_transform = transform_from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    return out, {
        "transform": lonlat_transform,
        "crs": crs,
        "bbox": bbox,
        "shape": (h, w),
    }
