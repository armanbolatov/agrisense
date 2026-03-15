"""AgriSense MCP Server — Tools for Claude Code integration.

Provides satellite data query, SAM segmentation, and NDVI computation
tools accessible via Claude Code.

Usage:
    python mcp_server.py
"""

import os
import json
from mcp.server.fastmcp import FastMCP, Context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REGION_PRESETS = {
    "kostanay": [63.2, 52.35, 63.45, 52.48],
    "akmola": [69.0, 51.35, 69.25, 51.48],
    "turkistan": [68.2, 42.2, 68.45, 42.35],
    "almaty": [76.8, 43.15, 77.05, 43.3],
}

mcp = FastMCP("AgriSense Remote Sensing")


@mcp.tool()
def get_region_presets(ctx: Context) -> str:
    """List predefined agricultural regions with their bounding boxes."""
    return json.dumps(REGION_PRESETS, indent=2)


@mcp.tool()
def search_satellite_imagery(bbox: str, date_range: str, ctx: Context) -> str:
    """Search for available Sentinel-2 satellite imagery.

    Args:
        bbox: Bounding box as 'west,south,east,north' (decimal degrees)
        date_range: Date range as 'YYYY-MM-DD/YYYY-MM-DD'
    """
    from satellite import search_sentinel2

    coords = [float(x.strip()) for x in bbox.split(",")]
    item = search_sentinel2(coords, date_range)
    if item is None:
        return json.dumps({"error": "No cloud-free imagery found"})

    return json.dumps({
        "scene_id": item.id,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "datetime": str(item.datetime),
        "bbox": list(item.bbox) if item.bbox else coords,
    }, indent=2)


@mcp.tool()
def compute_ndvi(bbox: str, date_range: str, ctx: Context) -> str:
    """Compute NDVI vegetation index for an area.

    Args:
        bbox: Bounding box as 'west,south,east,north' (decimal degrees)
        date_range: Date range as 'YYYY-MM-DD/YYYY-MM-DD'
    """
    from satellite import download_bands_for_indices
    from indices import compute_all_indices

    coords = [float(x.strip()) for x in bbox.split(",")]
    bands, geo_info = download_bands_for_indices(coords, date_range)
    all_indices = compute_all_indices(bands)

    return json.dumps({
        "bbox": coords,
        "image_size": list(geo_info["shape"]),
        "indices": {k: v["stats"] for k, v in all_indices.items()},
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
