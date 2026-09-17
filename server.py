"""AgriSense — Remote Sensing Platform Backend.

FastAPI server providing satellite imagery download, crop classification,
SAM interactive segmentation, vegetation indices, and AI chat.

Usage:
    python server.py
    # Then open http://localhost:8000
"""

import os
import json
import asyncio
import traceback
import warnings
import numpy as np
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
WORKSPACE = SCRIPT_DIR / "workspace"
WORKSPACE.mkdir(exist_ok=True)

app = FastAPI(title="AgriSense", version="1.0")
app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR / "static")), name="static")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

jobs: dict = {}  # job_id -> {status, progress, result, error, type}
_JOBS_MAX = 64

# Lazy-loaded models
_sam_segmenter = None

# Chat session history
chat_sessions: dict = {}  # session_id -> list of messages

# Latest results for chat context
latest_results: dict = {}  # type -> result data

# Latest raw arrays for GeoTIFF export: kind -> {"array": np.ndarray, "bbox": [..]}
export_store: dict = {}


def _store_export(kind, array, bbox):
    export_store[kind] = {"array": array, "bbox": bbox}


def _new_job(kind):
    """Register a job, evicting the oldest so `jobs` can't grow without bound."""
    while len(jobs) >= _JOBS_MAX:
        jobs.pop(next(iter(jobs)))
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "type": kind}
    return job_id


# Result cache: re-running the same analysis on the same area+dates is instant
# (no re-download). key -> {result, latest, export, export_kind}.
result_cache: dict = {}
_CACHE_MAX = 16


def _bbox_key(bbox):
    return ",".join(f"{round(float(v), 4)}" for v in bbox)


def _cache_complete(job_id, kind, key):
    """If a cached result exists for this key, finish the job from it instantly."""
    entry = result_cache.get(key)
    if not entry:
        return False
    if entry.get("latest"):
        latest_results[entry["latest"][0]] = entry["latest"][1]
    if entry.get("export") and entry.get("export_kind"):
        export_store[entry["export_kind"]] = entry["export"]
    jobs[job_id] = {"status": "complete", "type": kind, "result": entry["result"], "cached": True}
    return True


def _cache_store(key, result, latest=None, export=None, export_kind=None):
    if len(result_cache) >= _CACHE_MAX:
        result_cache.pop(next(iter(result_cache)))
    result_cache[key] = {"result": result, "latest": latest,
                         "export": export, "export_kind": export_kind}


def _decode_array(payload):
    """Decode a {dtype, shape, b64} array (from a crop-model service)."""
    import base64
    buf = base64.b64decode(payload["b64"])
    return np.frombuffer(buf, dtype=payload["dtype"]).reshape(payload["shape"])


def get_sam_segmenter():
    global _sam_segmenter
    if _sam_segmenter is None:
        from sam_segmenter import SamSegmenter
        _sam_segmenter = SamSegmenter()
        _sam_segmenter.load()
    return _sam_segmenter


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class BboxRequest(BaseModel):
    bbox: list[float]  # [west, south, east, north]
    date_ranges: list[str] = [
        "2023-04-20/2023-05-31",
        "2023-07-01/2023-07-31",
        "2023-09-01/2023-09-30",
    ]


class SamLoadRequest(BaseModel):
    bbox: list[float]
    date_range: str = "2023-07-01/2023-07-31"


class SamPredictRequest(BaseModel):
    points: list[list[float]] | None = None  # [[lon, lat], ...]
    labels: list[int] | None = None  # [1=fg, 0=bg, ...]
    box: list[float] | None = None  # [west, south, east, north]


class TimeSeriesRequest(BaseModel):
    bbox: list[float]
    year: int = 2023


class CropMapRequest(BaseModel):
    bbox: list[float]
    year: int = 2023


class ChangeRequest(BaseModel):
    bbox: list[float]
    year_a: int = 2020
    year_b: int = 2023


def _monthly_windows(year):
    """Apr–Oct monthly acquisition windows for a given year."""
    return [f"{year}-{m:02d}-01/{year}-{m:02d}-28" for m in range(4, 11)]


def _seasonal_ranges(year):
    """Spring / peak / late ranges for a given year (TerraMind input)."""
    return [
        f"{year}-04-20/{year}-05-31",
        f"{year}-07-01/{year}-07-31",
        f"{year}-09-01/{year}-09-30",
    ]


class ChatRequest(BaseModel):
    message: str
    bbox: list[float] | None = None
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

REGION_PRESETS = [
    {"name": "Kostanay", "bbox": [63.2, 52.35, 63.45, 52.48], "country": "Kazakhstan"},
    {"name": "Akmola", "bbox": [69.0, 51.35, 69.25, 51.48], "country": "Kazakhstan"},
    {"name": "Turkistan", "bbox": [68.2, 42.2, 68.45, 42.35], "country": "Kazakhstan"},
    {"name": "Almaty Region", "bbox": [76.8, 43.15, 77.05, 43.3], "country": "Kazakhstan"},
]

async def _geocode_region(text, force=False):
    """Geocode a place name anywhere in the world using OpenStreetMap Nominatim.

    Extracts the place name from the text (strips command words like
    'select', 'show', 'go to'), queries Nominatim globally, and returns a ~25km
    bbox centered on the result.

    force=True (used for explicit "select X" commands) skips the conversational
    filters and always attempts a lookup.

    Returns (bbox, name) or (None, None).
    """
    import re
    import httpx

    text_lower = text.lower().strip()
    if not force:
        # Opportunistic path (bare text): skip obvious chat so greetings and
        # questions don't get geocoded ("hi" -> some embassy and hijacks the map).
        GREETINGS = {
            "hi", "hii", "hiya", "hello", "helo", "hey", "yo", "sup", "hola",
            "thanks", "thank you", "thx", "ty", "ok", "okay", "k", "yes", "yep",
            "no", "nope", "cool", "nice", "good", "great", "hmm", "test", "testing",
            "bye", "goodbye", "please", "sure", "wow", "lol", "morning",
        }
        if text_lower.strip(".,!?") in GREETINGS:
            return None, None
        if any(w in text_lower for w in ['what', 'how', 'why', 'when', 'which', 'where',
                                          'can you', 'tell me', 'explain', 'help',
                                          'classify', 'ndvi', 'analysis', 'satellite', '?']):
            return None, None
        if len(text.split()) > 8:
            return None, None

    # Strip command + greeting/politeness words to extract the place name
    # ("hi, select london please" -> "london", "take me to tokyo" -> "tokyo").
    cleaned = re.sub(
        r"\b(hi+|hello|hey|yo|hiya|please|thanks|thank|you|can|could|would|will|ok|okay|"
        r"let'?s|let|lemme|i|we|us|want|wanna|wish|need|like|"
        r"select|show|go|navigate|find|zoom|display|open|fly|take|move|center|"
        r"to|on|me|the|region|area|oblast|province)\b",
        '', text, flags=re.IGNORECASE,
    ).strip().strip('.,!?')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return None, None

    async def _search(client, q, kz=False):
        params = {"q": q, "format": "json", "limit": 1, "addressdetails": 1}
        if kz:
            params["countrycodes"] = "kz"
        resp = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params=params, headers={"User-Agent": "AgriSense/1.0"},
        )
        r = resp.json()
        return r[0] if r else None

    # Query both a Kazakhstan-restricted and a worldwide result, then prefer the
    # KZ one unless the worldwide hit is much more prominent. This keeps KZ towns
    # like "Tekeli" in Kazakhstan while still sending "Paris"/"Tokyo" abroad.
    kz_hit = world_hit = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                kz_hit = await _search(client, cleaned, kz=True)
                world_hit = await _search(client, cleaned)
            break
        except Exception as e:
            print(f"Geocode error (attempt {attempt + 1}): {e}")

    def _imp(h):
        return float(h.get("importance", 0)) if h else 0.0

    if kz_hit and (not world_hit or _imp(kz_hit) >= 0.6 * _imp(world_hit)):
        hit = kz_hit
    else:
        hit = world_hit
    if not hit:
        return None, None

    try:
        lat = float(hit["lat"])
        lon = float(hit["lon"])
        name = hit.get("display_name", cleaned).split(",")[0]

        # Build a ~25x25 km bbox centered on the point
        dlat = 0.125  # ~13.9 km
        dlon = 0.125 / abs(np.cos(np.radians(lat)))  # adjust for latitude
        bbox = [
            round(lon - dlon, 4),
            round(lat - dlat, 4),
            round(lon + dlon, 4),
            round(lat + dlat, 4),
        ]
        return bbox, name
    except Exception as e:
        print(f"Geocode error: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(SCRIPT_DIR / "static" / "index.html"))


@app.get("/api/presets")
async def presets():
    return {"regions": REGION_PRESETS}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = jobs[job_id]
    resp = {"status": job["status"], "message": job.get("message", "")}
    if job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
    return resp


@app.get("/api/result/{job_id}")
async def job_result(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = jobs[job_id]
    if job["status"] == "complete":
        return job["result"]
    elif job["status"] == "error":
        return JSONResponse({"error": job.get("error", "Unknown error")}, status_code=500)
    else:
        return JSONResponse({"status": job["status"], "message": "Still processing"}, status_code=202)


# ---------------------------------------------------------------------------
# Satellite Preview
# ---------------------------------------------------------------------------

@app.post("/api/satellite-preview")
async def satellite_preview(req: BboxRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = _new_job("preview")
    background_tasks.add_task(_run_preview, job_id, req.bbox, req.date_ranges)
    return {"job_id": job_id}


def _run_preview(job_id, bbox, date_ranges):
    try:
        date_range = date_ranges[1] if len(date_ranges) > 1 else date_ranges[0]
        key = f"preview|{_bbox_key(bbox)}|{date_range}"
        if _cache_complete(job_id, "preview", key):
            return

        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = "Downloading satellite imagery..."

        from satellite import download_rgb
        from geo_utils import rgb_to_png

        rgb, geo_info = download_rgb(bbox, date_range)
        overlay = rgb_to_png(rgb)

        result = {"overlay_image": overlay, "bbox": bbox, "shape": list(geo_info["shape"])}
        _cache_store(key, result)
        jobs[job_id] = {"status": "complete", "type": "preview", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e), "type": "preview"}


# ---------------------------------------------------------------------------
# SAM Segmentation
# ---------------------------------------------------------------------------

@app.post("/api/sam/load")
async def sam_load(req: SamLoadRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = _new_job("sam_load")
    background_tasks.add_task(_run_sam_load, job_id, req.bbox, req.date_range)
    return {"job_id": job_id}


def _run_sam_load(job_id, bbox, date_range):
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = "Downloading satellite imagery..."

        from satellite import download_rgb
        from geo_utils import rgb_to_png

        rgb, geo_info = download_rgb(bbox, date_range)

        jobs[job_id]["status"] = "encoding"
        jobs[job_id]["message"] = "Computing SAM embeddings..."

        segmenter = get_sam_segmenter()
        segmenter.set_image(rgb, geo_info)

        overlay = rgb_to_png(rgb)

        jobs[job_id] = {
            "status": "complete",
            "type": "sam_load",
            "result": {
                "overlay_image": overlay,
                "bbox": bbox,
                "shape": list(geo_info["shape"]),
                "message": "SAM ready. Click on the map to segment.",
            },
        }
    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "sam_load"}


@app.post("/api/sam/predict")
async def sam_predict(req: SamPredictRequest):
    segmenter = get_sam_segmenter()
    if not segmenter.is_ready:
        return JSONResponse(
            {"error": "No image loaded. Use SAM Load first."},
            status_code=400,
        )

    result = segmenter.predict(
        points=req.points,
        labels=req.labels,
        box=req.box,
    )

    return result


# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------

@app.post("/api/ndvi")
async def ndvi(req: BboxRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = _new_job("ndvi")
    background_tasks.add_task(_run_ndvi, job_id, req.bbox, req.date_ranges)
    return {"job_id": job_id}


def _run_ndvi(job_id, bbox, date_ranges):
    try:
        date_range = date_ranges[1] if len(date_ranges) > 1 else date_ranges[0]
        key = f"ndvi|{_bbox_key(bbox)}|{date_range}"
        if _cache_complete(job_id, "ndvi", key):
            return

        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = "Downloading bands for NDVI..."

        from satellite import download_bands_for_indices
        from indices import compute_all_indices
        from geo_utils import index_to_colored_png

        bands, geo_info = download_bands_for_indices(bbox, date_range)

        jobs[job_id]["status"] = "computing"
        jobs[job_id]["message"] = "Computing vegetation indices..."

        indices = compute_all_indices(bands)
        ndvi_data = indices["ndvi"]

        overlay = index_to_colored_png(
            ndvi_data["array"], bbox,
            colormap=ndvi_data["colormap"],
            vmin=-0.2, vmax=0.9,
        )

        result = {
            "overlay_image": overlay,
            "bbox": bbox,
            "stats": ndvi_data["stats"],
            "indices": {k: v["stats"] for k, v in indices.items()},
        }

        latest_results["ndvi"] = result
        exp = {"array": ndvi_data["array"].astype("float32"), "bbox": bbox}
        _store_export("ndvi", exp["array"], bbox)
        _cache_store(key, result, latest=("ndvi", result), export=exp, export_kind="ndvi")
        jobs[job_id] = {"status": "complete", "type": "ndvi", "result": result}

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "ndvi"}


# ---------------------------------------------------------------------------
# NDVI Time Series
# ---------------------------------------------------------------------------

@app.post("/api/timeseries")
async def timeseries(req: TimeSeriesRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = _new_job("timeseries")
    background_tasks.add_task(_run_timeseries, job_id, req.bbox, _monthly_windows(req.year))
    return {"job_id": job_id}


def _run_timeseries(job_id, bbox, date_windows):
    try:
        key = f"timeseries|{_bbox_key(bbox)}|{date_windows[0]}|{date_windows[-1]}"
        if _cache_complete(job_id, "timeseries", key):
            return

        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = f"Sampling NDVI across {len(date_windows)} date windows..."

        from satellite import download_ndvi_series
        from indices import describe_phenology

        series = download_ndvi_series(bbox, date_windows)
        if not series:
            raise ValueError("No cloud-free scenes found in any date window.")

        phenology = describe_phenology(series)
        # The frontend renders `series` as an interactive Plotly chart.
        result = {
            "series": series,
            "phenology": phenology,
            "bbox": bbox,
        }

        latest_results["timeseries"] = result
        _cache_store(key, result, latest=("timeseries", result))
        jobs[job_id] = {"status": "complete", "type": "timeseries", "result": result}

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "timeseries"}


# ---------------------------------------------------------------------------
# Crop Map — seasonal per-pixel classification + per-month overlays (slider)
# ---------------------------------------------------------------------------

@app.post("/api/cropmap")
async def cropmap(req: CropMapRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = _new_job("cropmap")
    background_tasks.add_task(_run_cropmap, job_id, req.bbox, _monthly_windows(req.year), req.year)
    return {"job_id": job_id}


# Deep-learning crop-model services, tried in order. Each speaks the same
# /classify contract; if none answers, the rule-based classifier runs.
_CROP_SERVICES = ("PRITHVI_URL", "TERRAMIND_URL")


def _cropmap_service(job_id, bbox, year):
    """Classify with the first crop-model service that answers. Result dict or None."""
    import httpx
    for env in _CROP_SERVICES:
        url = os.environ.get(env)
        if not url:
            continue
        jobs[job_id]["status"] = "computing"
        jobs[job_id]["message"] = "Running crop model on the GPU..."
        try:
            resp = httpx.post(f"{url}/classify", timeout=240.0, json={
                "bbox": bbox, "year": year, "date_ranges": _seasonal_ranges(year)})
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"Crop service {env} unavailable: {e}")
            continue
        if "error" not in result:
            result.setdefault("bbox", bbox)
            return result
        print(f"Crop service {env} declined: {result['error']}")
    return None


def _run_cropmap(job_id, bbox, date_windows, year=2023):
    try:
        key = f"cropmap|{_bbox_key(bbox)}|{year}"
        if _cache_complete(job_id, "cropmap", key):
            return

        # Prefer a deep-learning crop model; fall back to the rule-based classifier.
        tm = _cropmap_service(job_id, bbox, year)
        if tm is not None:
            latest = {"legend": tm["legend"], "months": tm.get("months", []),
                      "model": tm.get("model", "deep-learning crop model")}
            latest_results["cropmap"] = latest
            export = None
            if tm.get("class_array"):  # raw class map for GeoTIFF export
                arr = _decode_array(tm["class_array"])
                _store_export("cropmap", arr, bbox)
                export = {"array": arr, "bbox": bbox}
            result = {k: v for k, v in tm.items() if k != "class_array"}
            _cache_store(key, result, latest=("cropmap", latest),
                         export=export, export_kind="cropmap")
            jobs[job_id] = {"status": "complete", "type": "cropmap", "result": result}
            return

        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = f"Building monthly NDVI stack ({len(date_windows)} months)..."

        from cropmap import (
            download_monthly_ndvi_stack, classify_seasonal, class_map_to_png,
        )
        from geo_utils import index_to_colored_png

        months, ndvi_stack, ndwi_mean = download_monthly_ndvi_stack(bbox, date_windows)

        jobs[job_id]["status"] = "computing"
        jobs[job_id]["message"] = "Classifying crops from seasonal profile..."

        class_map, areas = classify_seasonal(months, ndvi_stack, ndwi_mean)

        # Per-month NDVI overlays drive the season slider on the frontend.
        monthly = [
            {
                "month": m,
                "image": index_to_colored_png(ndvi_stack[i], bbox, colormap="RdYlGn",
                                              vmin=-0.2, vmax=0.9),
            }
            for i, m in enumerate(months)
        ]

        result = {
            "bbox": bbox,
            "class_image": class_map_to_png(class_map),
            "legend": areas,
            "months": months,
            "monthly": monthly,
        }
        latest = {"legend": areas, "months": months}
        latest_results["cropmap"] = latest
        cm_arr = class_map.astype("int16")
        _store_export("cropmap", cm_arr, bbox)
        _cache_store(key, result, latest=("cropmap", latest),
                     export={"array": cm_arr, "bbox": bbox}, export_kind="cropmap")
        jobs[job_id] = {"status": "complete", "type": "cropmap", "result": result}

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "cropmap"}


# ---------------------------------------------------------------------------
# Export (GeoTIFF) — NDVI / crop map for use in QGIS etc.
# ---------------------------------------------------------------------------

@app.get("/api/export/{kind}.tif")
async def export_geotiff(kind: str):
    item = export_store.get(kind)
    if not item:
        return JSONResponse({"error": f"No {kind} result to export yet."}, status_code=404)
    from geo_utils import write_geotiff_bytes
    data = write_geotiff_bytes(item["array"], item["bbox"])
    return Response(content=data, media_type="image/tiff", headers={
        "Content-Disposition": f'attachment; filename="agrisense_{kind}.tif"'})


# ---------------------------------------------------------------------------
# Change detection — NDVI difference between two years
# ---------------------------------------------------------------------------

@app.post("/api/change")
async def change(req: ChangeRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = _new_job("change")
    background_tasks.add_task(_run_change, job_id, req.bbox, req.year_a, req.year_b)
    return {"job_id": job_id}


def _run_change(job_id, bbox, year_a, year_b):
    try:
        key = f"change|{_bbox_key(bbox)}|{year_a}|{year_b}"
        if _cache_complete(job_id, "change", key):
            return

        from satellite import download_bands_for_indices
        from indices import compute_ndvi
        from geo_utils import diff_to_colored_png

        ndvis = {}
        for label, year in (("a", year_a), ("b", year_b)):
            jobs[job_id]["status"] = "downloading"
            jobs[job_id]["message"] = f"Downloading peak-season NDVI for {year}..."
            bands, _ = download_bands_for_indices(
                bbox, f"{year}-07-01/{year}-07-31", bands=("red",))
            ndvis[label] = compute_ndvi(bands["nir"], bands["red"])

        a, b = ndvis["a"], ndvis["b"]
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        diff = b[:h, :w] - a[:h, :w]
        nodata = (a[:h, :w] == 0) | (b[:h, :w] == 0)
        diff[nodata] = np.nan

        valid = diff[~np.isnan(diff)]
        result = {
            "bbox": bbox,
            "overlay_image": diff_to_colored_png(diff),
            "year_a": year_a, "year_b": year_b,
            "stats": {
                "mean_change": round(float(np.mean(valid)), 3) if valid.size else 0,
                "greener_pct": round(float(np.mean(valid > 0.05) * 100), 1) if valid.size else 0,
                "browner_pct": round(float(np.mean(valid < -0.05) * 100), 1) if valid.size else 0,
            },
        }
        diff_arr = diff.astype("float32")
        _store_export("change", diff_arr, bbox)
        latest = {"year_a": year_a, "year_b": year_b, "stats": result["stats"]}
        latest_results["change"] = latest
        _cache_store(key, result, latest=("change", latest),
                     export={"array": diff_arr, "bbox": bbox}, export_kind="change")
        jobs[job_id] = {"status": "complete", "type": "change", "result": result}
    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "change"}


# ---------------------------------------------------------------------------
# Chat — supports Anthropic API, OpenAI API, or any OpenAI-compatible local LLM
# ---------------------------------------------------------------------------
#
# Configure via environment variables:
#   CHAT_PROVIDER=anthropic|openai|local   (default: auto-detect from available keys)
#   ANTHROPIC_API_KEY=...                  (for Anthropic)
#   OPENAI_API_KEY=...                     (for OpenAI)
#   LOCAL_LLM_URL=http://localhost:11434/v1 (for Ollama, vLLM, LM Studio, etc.)
#   LOCAL_LLM_MODEL=llama3                 (model name for local LLM)
#   LOCAL_LLM_API_KEY=none                 (API key if needed, default "none")
#

def _build_system_prompt(bbox, results):
    """Build the system prompt with current analysis state."""
    context_parts = []
    if bbox:
        context_parts.append(f"Currently selected area: bbox={bbox}")
    if "ndvi" in results:
        ndvi_stats = results["ndvi"].get("stats", {})
        context_parts.append(f"Latest NDVI stats: {json.dumps(ndvi_stats)}")
    if "timeseries" in results:
        ts = results["timeseries"]
        # Serialize the temporal profile compactly: the instruction-tuned LLM
        # reasons over it as text (date: mean NDVI pairs + phenology summary).
        points = ", ".join(f"{p['date']}: {p['ndvi_mean']}" for p in ts["series"])
        context_parts.append(f"NDVI time series (date: mean NDVI): {points}")
        context_parts.append(f"Phenology analysis: {json.dumps(ts['phenology'])}")
    if "cropmap" in results:
        cm = results["cropmap"]
        breakdown = ", ".join(f"{c['label']}: {c['percent']}%" for c in cm["legend"] if c["percent"] > 0)
        model = cm.get("model", "rule-based seasonal")
        context_parts.append(f"Crop classification ({model}, share of area): {breakdown}")
    if "change" in results:
        ch = results["change"]
        s = ch["stats"]
        context_parts.append(
            f"NDVI change {ch['year_a']}->{ch['year_b']} (peak season): mean ΔNDVI "
            f"{s['mean_change']}, {s['greener_pct']}% of area greener, {s['browner_pct']}% browner.")

    context = "\n".join(context_parts) if context_parts else "No analysis has been run yet."

    return f"""You are AgriSense's assistant — a friendly, conversational helper inside a
satellite remote-sensing web app for Kazakhstan agriculture (Sentinel-2 imagery from
Microsoft Planetary Computer).

You SHOULD happily answer (these are all in-scope — never refuse them):
- small talk and greetings ("hi", "how are you", "thanks");
- questions about yourself ("who are you", "what can you do", "explain your
  functionality") — you're the AgriSense AI assistant;
- how the tools/features work ("how does SAM work", "how is crop classification
  done", "what is NDVI", "how does change detection work");
- interpreting the analysis results shown under "Current state";
- general remote-sensing and agriculture concepts;
- which button to click to run something.

Only politely DECLINE genuinely unrelated *tasks/trivia* — writing or debugging code,
math problems, world trivia (capitals, sports), translation, other software — e.g.
"That's outside what I do — I'm here for satellite imagery and crop analysis." Then
steer back. Identity/how-it-works/greeting questions are IN scope.

How AgriSense works (use this to answer accurately, keep it brief):
- Data: Sentinel-2 L2A, ~10 m/pixel, fetched live from Microsoft Planetary Computer.
- NDVI = (NIR-Red)/(NIR+Red), vegetation health (-1..1; healthy crops > 0.3). Also
  NDWI (water) and EVI. "NDVI index" maps it for the selected area.
- "NDVI time series" samples one cloud-free scene per month (Apr-Oct) and plots
  NDVI/NDWI/EVI to reveal crop phenology.
- "Classify crops" runs Prithvi-EO-2.0, a multi-temporal vision transformer
  fine-tuned on the European Sen4Map dataset, on the GPU. It reads 12 monthly
  Sentinel-2 images and labels ~150-600 m cells with 8 classes (cereals, root
  crops, industrial crops, pulses/vegetables, fodder, bareland, woodland,
  grassland). If it is unavailable, a rule-based seasonal classifier runs instead.
- "SAM segment" uses Meta's Segment Anything Model: it loads the image, you click
  points on the map (green=include, red=exclude), and SAM predicts the object/field
  outline as a polygon.
- "Detect change" computes the NDVI difference between two years' peak seasons.
- You can also type commands here ("classify crops in Pavlodar", "run ndvi") and a
  place name to navigate the map.

Rules:
- Base any claim about *results* on the "Current state" data. Never invent numbers.
- Be concise and warm: 1-4 sentences, plain text, no markdown, no code blocks.

Crop phenology hints: winter crops peak in May-June, spring/summer crops peak in
July-August, grassland/forest stays high and flat, bare/fallow stays low all season.

Current state:
{context}"""


def _detect_analysis_intent(text):
    """Map an explicit analysis command to a tool the UI can run, else None.
    Triggers on commands like 'classify crops', 'run ndvi', 'ndvi time series',
    'satellite preview', 'detect change' — not on questions like 'what is ndvi'."""
    import re
    t = text.lower().strip()
    # Questions / interpretation requests -> let the LLM answer from context,
    # never run a tool. ("how can I interpret these changes" must NOT re-run it.)
    if t.endswith("?") or re.search(
            r"\b(what|whats|what's|how|why|which|who|when|interpret|interpretation|"
            r"explain|describe|mean|meaning|summar|tell me|is it|is this|are these|"
            r"should i|help me understand|understand|define|difference between)\b", t):
        return None
    # Command verbs that signal "do this analysis now".
    verb = re.search(r'\b(run|show|compute|classify|segment|generate|make|create|get|'
                     r'display|map|detect|plot|visuali[sz]e|do|give me|perform)\b', t)
    # Ordered, most-specific first. Standalone phrases trigger without a verb.
    if re.search(r'change detection|ndvi difference|compare\s+\d{4}', t) \
            or (verb and re.search(r'\bchange\b|\bdiffer', t)):
        return "change"
    if re.search(r'time\s*series|seasonal profile|phenolog', t):
        return "timeseries"
    if re.search(r'crop map|crop classification', t) or (verb and re.search(r'\bcrops?\b', t)):
        return "cropmap"
    if verb and re.search(r'\bsegment\b|\bsam\b', t):
        return "sam"
    if verb and (re.search(r'\bndvi\b', t) or "vegetation index" in t):
        return "ndvi"
    if re.search(r'satellite preview', t) or (verb and re.search(r'satellite|imagery|\bpreview\b|\brgb\b', t)):
        return "preview"
    return None


async def _place_from_command(text):
    """Pull a place name out of an analysis command and geocode it, e.g.
    'classify crops in pavlodar' -> bbox of Pavlodar. (None, None) if no place."""
    import re
    cleaned = re.sub(r'\b\d{4}\b', '', text)  # drop years (e.g. "compare 2019 and 2023")
    cleaned = re.sub(
        r'\b(run|show|compute|do|classify|segment|analy[sz]e|generate|make|create|get|'
        r'display|map|detect|detection|give|plot|visuali[sz]e|perform|'
        r'me|a|an|the|in|for|of|on|over|at|near|to|and|between|vs|versus|'
        r'crops?|classification|ndvi|ndwi|evi|time|series|seasonal|profile|phenology|trend|'
        r'satellite|preview|imagery|rgb|vegetation|index|change|difference|compare|comparison|'
        r'data|results?|model|fields?|here|this|that|now|today|year|years|area|region|please)\b',
        '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^\w\s,-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) < 3:  # too short to be a real place name
        return None, None
    return await _geocode_region(cleaned, force=True)


def _get_chat_provider():
    """Determine which chat backend to use."""
    explicit = os.environ.get("CHAT_PROVIDER", "").lower()
    if explicit:
        return explicit

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("LOCAL_LLM_URL"):
        return "local"
    return None


async def _chat_anthropic(system_prompt, history):
    """Chat via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=500,
        system=system_prompt,
        messages=history,
    )
    return response.content[0].text


async def _chat_openai_compatible(system_prompt, history, base_url=None, api_key=None, model=None):
    """Chat via OpenAI-compatible API (works with OpenAI, Ollama, vLLM, LM Studio, etc.)."""
    import httpx

    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    key = api_key or os.environ.get("OPENAI_API_KEY", "none")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = [{"role": "system", "content": system_prompt}] + history

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model_name, "messages": messages, "max_tokens": 500},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _stream_openai_compatible(system_prompt, history, base_url=None, api_key=None, model=None):
    """Yield content tokens from an OpenAI-compatible streaming completion."""
    import httpx
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    key = api_key or os.environ.get("OPENAI_API_KEY", "none")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    messages = [{"role": "system", "content": system_prompt}] + history
    payload = {"model": model_name, "messages": messages, "max_tokens": 500, "stream": True}
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta


async def _stream_llm(provider, system_prompt, history):
    """Provider-agnostic token stream. Anthropic falls back to one chunk."""
    if provider == "local":
        gen = _stream_openai_compatible(
            system_prompt, history,
            base_url=os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("LOCAL_LLM_API_KEY", "none"),
            model=os.environ.get("LOCAL_LLM_MODEL", "llama3"))
    elif provider == "openai":
        gen = _stream_openai_compatible(system_prompt, history)
    else:  # anthropic — no streaming here, emit the full reply as one chunk
        yield await _chat_anthropic(system_prompt, history)
        return
    async for tok in gen:
        yield tok


def _chat_history(session_id, message):
    """Append the user message to the session history (trimmed) and return it."""
    while len(chat_sessions) > 64 and session_id not in chat_sessions:
        chat_sessions.pop(next(iter(chat_sessions)))
    history = chat_sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": message})
    if len(history) > 20:
        history = history[-20:]
        chat_sessions[session_id] = history
    return history


def _is_navigation_command(text):
    """True if the message is an explicit 'select/go to/show ... <place>' command,
    even when wrapped in greetings/politeness ('hi, select london please',
    'can you go to paris')."""
    import re
    # Strip a leading wrapper of greetings / "let's" / "i want to" / "can you" /
    # "please", then check whether the remainder starts with a navigation verb.
    stripped = re.sub(
        r"^\s*("
        r"(hi+|hello|hey|yo|hiya|ok|okay)[\s,!.]+"
        r"|(let'?s|let me|lemme)\s+"
        r"|i\s+(want|wanna|would\s+like|'?d\s+like|wish|need)\s+to\s+"
        r"|(can|could|would|will)\s+you[\s,]*"
        r"|please[\s,]*"
        r")*",
        '', text, flags=re.IGNORECASE,
    )
    return bool(re.match(
        r'^(select|show|go\s+to|navigate(\s+to)?|find|zoom(\s+to)?|display|open|'
        r'fly\s+to|take\s+me\s+to|move\s+to|center\s+on)\b',
        stripped, re.IGNORECASE,
    ))


_TOOL_NAMES = {"ndvi": "NDVI", "timeseries": "NDVI time series",
               "cropmap": "crop classification", "preview": "satellite preview",
               "sam": "SAM segmentation", "change": "change detection"}

_NO_PROVIDER_MSG = (
    "No chat backend configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
    "LOCAL_LLM_URL + LOCAL_LLM_MODEL.")


async def _resolve_chat(req):
    """Decide nav/action/error without the LLM. Returns a response dict, or
    None if the message should go to the LLM."""
    # 1. Analysis command first — so "show ndvi time series" runs an analysis
    #    instead of being mistaken for a "show <place>" navigation command.
    tool = _detect_analysis_intent(req.message)
    if tool:
        pbbox, pname = await _place_from_command(req.message)
        resp = {"response": f"Running {_TOOL_NAMES.get(tool, tool)}"
                            + (f" for {pname}…" if pname else "…"),
                "action": tool}
        if pbbox:
            resp["bbox"] = pbbox
            resp["region_name"] = pname
        return resp

    # 2. Navigation command ("select/go to <place>").
    nav = _is_navigation_command(req.message)
    bbox, region_name = await _geocode_region(req.message, force=nav)
    if nav:
        if bbox:
            return {"response": f"Selected {region_name}. Run any analysis tool, or ask me about it.",
                    "action": None, "bbox": bbox, "region_name": region_name}
        return {"response": "I couldn't find that place. Try the city or region name again.",
                "action": None}

    # 3. Bare place name (opportunistic).
    if bbox:
        return {"response": f"Selected {region_name}. Run any analysis tool, or ask me about it.",
                "action": None, "bbox": bbox, "region_name": region_name}

    # 4. Otherwise -> LLM.
    return None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    meta = await _resolve_chat(req)
    if meta is not None:
        return meta
    provider = _get_chat_provider()
    if not provider:
        return {"response": _NO_PROVIDER_MSG, "action": None}
    system_prompt = _build_system_prompt(req.bbox, latest_results)
    history = _chat_history(req.session_id, req.message)
    try:
        full = "".join([tok async for tok in _stream_llm(provider, system_prompt, history)])
        history.append({"role": "assistant", "content": full})
        return {"response": full, "action": None}
    except Exception as e:
        return {"response": f"Chat error: {str(e)}", "action": None}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-sent events: a single 'meta' event for nav/action, else streamed
    'token' events followed by 'done'."""
    meta = await _resolve_chat(req)

    async def gen():
        if meta is not None:
            yield f"data: {json.dumps({'type': 'meta', **meta})}\n\n"
            return
        provider = _get_chat_provider()
        if not provider:
            yield f"data: {json.dumps({'type': 'meta', 'response': _NO_PROVIDER_MSG, 'action': None})}\n\n"
            return
        system_prompt = _build_system_prompt(req.bbox, latest_results)
        history = _chat_history(req.session_id, req.message)
        full = ""
        try:
            async for tok in _stream_llm(provider, system_prompt, history):
                full += tok
                yield f"data: {json.dumps({'type': 'token', 't': tok})}\n\n"
            history.append({"role": "assistant", "content": full})
        except Exception as e:
            yield f"data: {json.dumps({'type': 'token', 't': f'Chat error: {e}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


class GeocodeRequest(BaseModel):
    query: str


@app.post("/api/geocode")
async def geocode(req: GeocodeRequest):
    """Look up a place by name using OpenStreetMap and return its bbox."""
    bbox, name = await _geocode_region(req.query)
    if bbox:
        return {"found": True, "name": name, "bbox": bbox}
    return {"found": False, "name": None, "bbox": None}


@app.get("/api/chat/provider")
async def chat_provider_info():
    """Return current chat provider configuration."""
    provider = _get_chat_provider()
    return {
        "provider": provider,
        "configured": provider is not None,
        "local_url": os.environ.get("LOCAL_LLM_URL", ""),
        "local_model": os.environ.get("LOCAL_LLM_MODEL", ""),
    }


@app.on_event("startup")
async def _warmup_llm():
    """Fire a couple of throwaway generations at boot so the first real user
    message doesn't pay vLLM's one-time kernel-JIT latency spike."""
    async def run():
        provider = _get_chat_provider()
        if provider not in ("local", "openai"):
            return
        for attempt in range(6):  # vLLM may still be loading right after boot
            try:
                if provider == "local":
                    await _chat_openai_compatible(
                        "warmup", [{"role": "user", "content": "hi"}],
                        base_url=os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1"),
                        api_key=os.environ.get("LOCAL_LLM_API_KEY", "none"),
                        model=os.environ.get("LOCAL_LLM_MODEL", "llama3"),
                    )
                else:
                    await _chat_openai_compatible("warmup", [{"role": "user", "content": "hi"}])
                print("LLM warmup complete")
                return
            except Exception:
                await asyncio.sleep(5)
    asyncio.create_task(run())


@app.on_event("startup")
async def _precache_presets():
    """Warm the result cache for the preset regions (satellite preview + NDVI for
    the 2023 peak season) so clicking a Quick region returns instantly. Runs in a
    daemon thread well after boot so it never blocks startup or the LLM warmup."""
    import threading
    import time

    def warm():
        time.sleep(30)
        dr = _seasonal_ranges(2023)
        for region in REGION_PRESETS:
            for runner, kind in ((_run_preview, "preview"), (_run_ndvi, "ndvi")):
                jid = f"precache-{kind}-{region['name']}"
                jobs[jid] = {"status": "queued", "type": kind}
                try:
                    runner(jid, region["bbox"], dr)
                except Exception:
                    pass
                jobs.pop(jid, None)
        print(f"Preset pre-cache complete ({len(REGION_PRESETS)} regions)")

    threading.Thread(target=warm, daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_bbox(bbox):
    if len(bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must have 4 values: [west, south, east, north]")
    west, south, east, north = bbox
    width_km = abs(east - west) * 111 * abs(np.cos(np.radians((south + north) / 2)))
    height_km = abs(north - south) * 111
    if width_km > 30 or height_km > 30:
        raise HTTPException(status_code=400, detail=f"Area too large ({width_km:.0f}x{height_km:.0f} km). Max ~25x25 km.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  AgriSense — Remote Sensing Platform")
    print("  http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
