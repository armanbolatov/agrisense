"""Sentinel-2 data access via Microsoft Planetary Computer.

Adapted from /home/arman/terramind/run_crop_kz.py
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from pyproj import Transformer

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
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    results = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=3,
    )
    items = list(results.items())
    return items[0] if items else None


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
    href = item.assets[asset_name].href
    with rasterio.open(href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        west, south = transformer.transform(bbox_lonlat[0], bbox_lonlat[1])
        east, north = transformer.transform(bbox_lonlat[2], bbox_lonlat[3])
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().round_lengths()

        if target_size:
            data = src.read(1, window=window, out_shape=target_size, resampling=Resampling.bilinear)
        else:
            data = src.read(1, window=window)

        # Compute the transform for the read window
        win_transform = src.window_transform(window)
        crs = str(src.crs)

    return data.astype(np.float32), win_transform, crs


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

    # Read B04 (Red), B03 (Green), B02 (Blue)
    red, _, _ = read_band(item, "B04", bbox, 10)
    target = red.shape
    green, _, _ = read_band(item, "B03", bbox, 10, target_size=target)
    blue, win_transform, crs = read_band(item, "B02", bbox, 10, target_size=target)

    # Ensure consistent sizes
    red = _ensure_size(red, target)
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


def download_bands_for_indices(bbox, date_range, on_progress=None):
    """Download bands needed for vegetation indices.

    Returns:
        bands: dict of band_name -> np.ndarray [H, W]
        geo_info: dict
    """
    if on_progress:
        on_progress(f"Downloading bands for indices...")

    item = search_sentinel2(bbox, date_range)
    if item is None:
        raise ValueError(f"No cloud-free image found for {date_range}")

    blue, _, _ = read_band(item, "B02", bbox, 10)
    target_size = blue.shape
    green, _, _ = read_band(item, "B03", bbox, 10, target_size=target_size)
    red, _, _ = read_band(item, "B04", bbox, 10, target_size=target_size)
    nir, win_transform, crs = read_band(item, "B8A", bbox, 20, target_size=target_size)

    blue = _ensure_size(blue, target_size)
    green = _ensure_size(green, target_size)
    red = _ensure_size(red, target_size)
    nir = _ensure_size(nir, target_size)

    h, w = target_size
    lonlat_transform = transform_from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    return {
        "blue": blue, "green": green, "red": red, "nir": nir,
    }, {
        "transform": lonlat_transform,
        "crs": crs,
        "bbox": bbox,
        "shape": (h, w),
    }
