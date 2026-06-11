"""Geospatial utilities: raster-to-vector conversion and PNG/chart rendering."""

import io
import base64
import numpy as np
from rasterio.features import shapes as rasterio_shapes
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mask_to_geojson(mask, transform, crs):
    """Convert binary mask to GeoJSON polygon in EPSG:4326.

    Args:
        mask: 2D boolean/uint8 array
        transform: rasterio Affine transform
        crs: source CRS string

    Returns:
        GeoJSON Feature dict or None if mask is empty
    """
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.sum() == 0:
        return None

    polygons = []
    for geom, value in rasterio_shapes(mask_uint8, mask=mask_uint8, transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    if not polygons:
        return None

    merged = unary_union(polygons)

    # Transform to EPSG:4326 if needed
    if crs and str(crs) != "EPSG:4326":
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        def transform_coords(coords):
            return [t.transform(x, y) for x, y in coords]

        geojson_geom = mapping(merged)
        if geojson_geom["type"] == "Polygon":
            geojson_geom["coordinates"] = [transform_coords(ring) for ring in geojson_geom["coordinates"]]
        elif geojson_geom["type"] == "MultiPolygon":
            geojson_geom["coordinates"] = [
                [transform_coords(ring) for ring in poly]
                for poly in geojson_geom["coordinates"]
            ]
    else:
        geojson_geom = mapping(merged)

    return {
        "type": "Feature",
        "geometry": geojson_geom,
        "properties": {"area_pixels": int(mask_uint8.sum())},
    }


def index_to_colored_png(array, bbox, colormap="RdYlGn", vmin=-0.2, vmax=0.9):
    """Convert a 2D index array (e.g. NDVI) to a colored base64 PNG.

    Args:
        array: 2D numpy array of float values
        bbox: [west, south, east, north]
        colormap: matplotlib colormap name
        vmin, vmax: value range for colormap

    Returns:
        base64-encoded PNG string
    """
    cmap = plt.get_cmap(colormap)
    normalized = np.clip((array - vmin) / (vmax - vmin + 1e-10), 0, 1)
    colored = (cmap(normalized) * 255).astype(np.uint8)
    # Set nodata to transparent
    nodata_mask = np.isnan(array) | (array == 0)
    colored[nodata_mask] = [0, 0, 0, 0]

    buf = io.BytesIO()
    plt.imsave(buf, colored, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def rgb_to_png(rgb_array):
    """Convert RGB array [H, W, 3] uint8 to base64 PNG."""
    buf = io.BytesIO()
    plt.imsave(buf, rgb_array, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
