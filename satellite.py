"""Sentinel-2 data access via Microsoft Planetary Computer.

Adapted from /home/arman/terramind/run_crop_kz.py
"""

import hashlib
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

# A band read is the dominant cost and COG pixels never change, so reads are
# cached twice: in memory on a byte budget (one 12-month Prithvi stack is 130+
# reads, which an entry-counted cache evicted wholesale), and on disk, which
# survives restarts. The disk key carries MAX_PIXELS because the app reads at
# 1024 px and the Prithvi service natively -- same band, different pixels.
_BAND_CACHE_BYTES = int(float(os.environ.get("AGRISENSE_BAND_CACHE_GB", "2")) * 2**30)
_DISK_CACHE = os.environ.get("AGRISENSE_CACHE_DIR", "") or \
    os.path.join(os.path.expanduser("~"), ".cache", "agrisense", "bands")
_DISK_CACHE_BYTES = int(float(os.environ.get("AGRISENSE_DISK_CACHE_GB", "20")) * 2**30)

# Candidates per search. Cloud-sorted hits are often tiles that merely clip the
# area, so _pick_scene needs enough of them to find a pass that covers it.
_SEARCH_ITEMS = 50

# Native reads are capped to this many pixels on the long edge. Map overlays and
# SAM (which resizes to 1024 internally) gain nothing from a full 10 m raster,
# and rasterio serves the smaller read from the COG's overviews instead of
# fetching every tile. Set AGRISENSE_MAX_PX=0 for full native resolution.
MAX_PIXELS = int(os.environ.get("AGRISENSE_MAX_PX", "1024"))

_catalog = None
_scene_cache: dict = {}
_band_cache: OrderedDict = OrderedDict()
_band_bytes = 0
_disk_writes = 0
_cache_lock = Lock()


def clear_caches():
    """Drop every cached scene and band (used by bench.py). Disk cache is kept."""
    global _band_bytes
    with _cache_lock:
        _scene_cache.clear()
        _band_cache.clear()
        _band_bytes = 0


def _disk_path(key):
    return os.path.join(_DISK_CACHE, key[:2], key + ".npz")


def _disk_get(key):
    """Cached read from disk, or None. A half-written file just misses."""
    path = _disk_path(key)
    try:
        with np.load(path) as z:
            out = (z["data"], rasterio.Affine(*z["tf"]), str(z["crs"]))
        os.utime(path, None)            # touch: disk eviction is LRU by mtime
        return out
    except Exception:                   # noqa: BLE001 (missing, truncated, bad npz)
        return None


def _disk_put(key, out):
    global _disk_writes
    path = _disk_path(key)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "wb") as fh:     # a file object: savez would append .npz to a name
            np.savez(fh, data=out[0], tf=np.array(tuple(out[1])[:6]), crs=out[2])
        os.replace(tmp, path)           # atomic: a reader never sees a partial file
    except Exception:                   # noqa: BLE001 (full disk, read-only mount)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return
    with _cache_lock:
        _disk_writes += 1
        due = _disk_writes % 64 == 0
    if due:
        _disk_prune()


def _disk_prune():
    """Drop the least recently used files until the cache fits its budget."""
    files = []
    for root, _, names in os.walk(_DISK_CACHE):
        for n in names:
            f = os.path.join(root, n)
            try:
                st = os.stat(f)
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, f))
    total = sum(f[1] for f in files)
    for _, size, f in sorted(files):
        if total <= _DISK_CACHE_BYTES:
            break
        try:
            os.remove(f)
            total -= size
        except OSError:
            pass


def _band_remember(key, out):
    """Keep the read in memory, evicting the least recently used over budget."""
    global _band_bytes
    with _cache_lock:
        if key not in _band_cache:
            _band_bytes += out[0].nbytes
        _band_cache[key] = (time.time(), out)
        _band_cache.move_to_end(key)
        while _band_bytes > _BAND_CACHE_BYTES and len(_band_cache) > 1:
            _, old = _band_cache.popitem(last=False)
            _band_bytes -= old[1][0].nbytes


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


class Scene:
    """One acquisition: the tiles of a single pass that cover the bbox.

    An area straddling an MGRS tile edge (the Akmola preset straddles four) fits
    in no single tile. Attributes fall through to the first tile, so only
    read_band needs to know about the rest.
    """

    def __init__(self, tiles):
        self.tiles = tiles

    def __getattr__(self, name):
        return getattr(self.tiles[0], name)

    @property
    def id(self):
        return "+".join(t.id for t in self.tiles)


def _minimal_tiles(tiles, area):
    """Tiles of one pass that add new ground, least cloudy first.

    Drops neighbours that miss the area and the duplicate tiles the catalog
    returns for reprocessed versions.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    kept, covered = [], None
    for tile in sorted(tiles, key=lambda t: float(t.properties.get("eo:cloud_cover") or 0)):
        part = shape(tile.geometry).intersection(area)
        if part.is_empty or (covered is not None and part.difference(covered).is_empty):
            continue
        kept.append(tile)
        covered = part if covered is None else unary_union([covered, part])
        if covered.area >= area.area - 1e-12:
            break
    return kept, covered


def _pick_scene(items, bbox):
    """The pass covering most of the bbox, then fewest tiles, then least cloud.

    A scene only has to *intersect* the bbox, so the least-cloudy hit is often a
    tile clipping one corner. Tiles of one pass share illumination, so merging
    them in read_band leaves no seam.
    """
    if not items:
        return None
    from shapely.geometry import box
    area = box(*bbox)

    passes = {}
    for it in items:
        passes.setdefault(it.properties["datetime"], []).append(it)

    def rank(tiles):
        kept, covered = _minimal_tiles(tiles, area)
        if not kept:
            return None
        cloud = max(float(t.properties.get("eo:cloud_cover") or 0) for t in kept)
        return (-round(covered.area / area.area, 3), len(kept), cloud), kept

    ranked = [r for r in map(rank, passes.values()) if r]
    if not ranked:
        return Scene([items[0]])
    return Scene(min(ranked, key=lambda r: r[0])[1])


def search_sentinel2(bbox, date_range, max_cloud=20):
    """Search Planetary Computer for best cloud-free Sentinel-2 scene.

    Args:
        bbox: [west, south, east, north] in lon/lat
        date_range: string like "2023-07-01/2023-07-31"
        max_cloud: maximum cloud cover percentage

    Returns:
        Scene (one or more tiles of a single pass) or None
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
        max_items=_SEARCH_ITEMS,
    )
    item = _pick_scene(list(results.items()), bbox)
    with _cache_lock:
        _scene_cache[key] = (time.time(), item)
    return item


def _harmonize(item, asset_name, arr):
    """Undo ESA's +1000 offset (processing baseline 04.00, 2022+).

    Done at the one place bands are read, so every tool shares one reflectance
    scale; a 2022+ scene otherwise reads ~1000 DN brighter than an older one.
    """
    if not asset_name.startswith("B"):
        return arr              # SCL is a class band, not reflectance
    if float(item.properties.get("s2:processing_baseline") or 0) < 4.0:
        return arr
    return np.clip(arr - 1000, 0, None)     # new array: cached reads stay untouched


def _read_tile(tile, asset_name, bbox_lonlat, target_size, resampling):
    """One tile's pixels over the bbox, padded with nodata where it does not reach."""
    with rasterio.open(tile.assets[asset_name].href) as src:
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

        # Without boundless a partial overlap is silently clipped, and the
        # caller then stretches it over the whole bbox. Pad with nodata instead.
        inside = (window.col_off >= 0 and window.row_off >= 0
                  and window.col_off + window.width <= src.width
                  and window.row_off + window.height <= src.height)
        edge = {} if inside else {"boundless": True, "fill_value": 0}

        if target_size:
            data = src.read(1, window=window, out_shape=target_size,
                            resampling=resampling, **edge)
        else:
            data = src.read(1, window=window, **edge)

        return data.astype(np.float32), src.window_transform(window), str(src.crs)


def read_band(item, asset_name, bbox_lonlat, native_res, target_size=None):
    """Read a single band from a COG asset, cropped to bbox.

    Args:
        item: a Scene (or pystac Item) from search_sentinel2
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

    # Pixels are immutable, so a disk hit is as good as a fresh read and costs
    # no network. MAX_PIXELS is in the key: the same band read at 1024 px and
    # natively are different arrays.
    dkey = hashlib.sha1("|".join(map(str, (*key, MAX_PIXELS))).encode()).hexdigest()
    cached = _disk_get(dkey)
    if cached is not None:
        _band_remember(key, cached)
        return cached

    # SCL holds class numbers, so averaging neighbours would invent classes.
    resampling = Resampling.nearest if asset_name == "SCL" else Resampling.bilinear
    tiles = getattr(item, "tiles", [item])

    raw, win_transform, crs = _read_tile(
        tiles[0], asset_name, bbox_lonlat, target_size, resampling)
    gap = raw == 0
    data = _harmonize(tiles[0], asset_name, raw)

    # Sibling tiles fill what the first did not reach. Same pass and resolution,
    # so filling the gap on the first one's grid is enough; a tile in another UTM
    # zone cannot be merged that way and is skipped, leaving nodata.
    for tile in tiles[1:]:
        if not gap.any():
            break
        arr, _, tile_crs = _read_tile(
            tile, asset_name, bbox_lonlat, data.shape, resampling)
        if tile_crs != crs:
            continue
        data[gap] = _harmonize(tile, asset_name, arr)[gap]
        gap &= arr == 0

    # Cached arrays are shared between callers, so treat them as read-only.
    out = (data, win_transform, crs)
    _band_remember(key, out)
    _disk_put(dkey, out)
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
