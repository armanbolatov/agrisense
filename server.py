"""AgriSense — Remote Sensing Platform Backend.

FastAPI server providing satellite imagery download, crop classification,
SAM interactive segmentation, vegetation indices, and AI chat.

Usage:
    python server.py
    # Then open http://localhost:8000
"""

import os
import sys
import json
import asyncio
import traceback
import warnings
import numpy as np
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
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
gpu_lock = asyncio.Lock()

# Lazy-loaded models
_sam_segmenter = None

# Chat session history
chat_sessions: dict = {}  # session_id -> list of messages

# Latest results for chat context
latest_results: dict = {}  # type -> result data


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

async def _geocode_region(text):
    """Geocode a place name using OpenStreetMap Nominatim.

    Extracts the place name from the text (strips command words like
    'select', 'show', 'go to'), queries Nominatim, and returns a ~25km bbox
    centered on the result.

    Returns (bbox, name) or (None, None).
    """
    import re
    import httpx

    # Skip messages that are clearly questions or long sentences, not place names
    text_lower = text.lower().strip()
    if any(w in text_lower for w in ['what', 'how', 'why', 'when', 'which', 'where',
                                      'can you', 'tell me', 'explain', 'help',
                                      'classify', 'ndvi', 'analysis', 'satellite']):
        return None, None
    # Skip very long messages (likely questions, not place names)
    if len(text.split()) > 8:
        return None, None

    # Strip common command words to extract the place name
    cleaned = re.sub(
        r'\b(select|show|go\s+to|navigate|find|zoom|display|me|the|region|area|oblast|province)\b',
        '', text, flags=re.IGNORECASE,
    ).strip().strip('.,!?')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return None, None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": cleaned + ", Kazakhstan",
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "AgriSense/1.0"},
            )
            results = resp.json()
            if not results:
                # Try without appending Kazakhstan
                resp = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": cleaned,
                        "format": "json",
                        "limit": 1,
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": "AgriSense/1.0"},
                )
                results = resp.json()

            if not results:
                return None, None

            hit = results[0]
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
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "type": "preview"}
    background_tasks.add_task(_run_preview, job_id, req.bbox, req.date_ranges)
    return {"job_id": job_id}


def _run_preview(job_id, bbox, date_ranges):
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = "Downloading satellite imagery..."

        from satellite import download_rgb
        from geo_utils import rgb_to_png

        date_range = date_ranges[1] if len(date_ranges) > 1 else date_ranges[0]
        rgb, geo_info = download_rgb(bbox, date_range)
        overlay = rgb_to_png(rgb)

        jobs[job_id] = {
            "status": "complete",
            "type": "preview",
            "result": {
                "overlay_image": overlay,
                "bbox": bbox,
                "shape": list(geo_info["shape"]),
            },
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e), "type": "preview"}


# ---------------------------------------------------------------------------
# SAM Segmentation
# ---------------------------------------------------------------------------

@app.post("/api/sam/load")
async def sam_load(req: SamLoadRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "type": "sam_load"}
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
        jobs[job_id]["message"] = "Computing SAM embeddings (this takes 10-30s)..."

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
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "type": "ndvi"}
    background_tasks.add_task(_run_ndvi, job_id, req.bbox, req.date_ranges)
    return {"job_id": job_id}


def _run_ndvi(job_id, bbox, date_ranges):
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = "Downloading bands for NDVI..."

        from satellite import download_bands_for_indices
        from indices import compute_all_indices
        from geo_utils import index_to_colored_png

        date_range = date_ranges[1] if len(date_ranges) > 1 else date_ranges[0]
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
        jobs[job_id] = {"status": "complete", "type": "ndvi", "result": result}

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "ndvi"}


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

    context = "\n".join(context_parts) if context_parts else "No analysis has been run yet."

    return f"""You are AgriSense, an AI assistant for remote sensing and crop analysis.
You help users analyze satellite imagery, understand crop classification results, and interpret vegetation indices.

Available tools the user can trigger from the UI:
- Satellite Preview: View RGB satellite imagery for an area
- SAM Segmentation: Interactive click-to-segment using Segment Anything Model
- NDVI: Compute vegetation health index

The platform uses Sentinel-2 satellite data from Microsoft Planetary Computer.

Current state:
{context}

Be concise and helpful. If the user asks to run an analysis, tell them which button to click.
Respond in 2-4 sentences unless more detail is needed."""


def _detect_action(text):
    """Detect if the AI response suggests triggering a tool."""
    lower = text.lower()
    if "ndvi" in lower and "click" in lower:
        return "ndvi"
    if "sam" in lower and "click" in lower:
        return "sam"
    return None


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


def _is_navigation_command(text):
    """Check if the message is a navigation/selection command."""
    import re
    return bool(re.match(
        r'^\s*(select|show|go\s+to|navigate|find|zoom|display|fly\s+to|open)\b',
        text, re.IGNORECASE,
    ))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # If it looks like a navigation command, geocode directly — skip the LLM
    if _is_navigation_command(req.message):
        bbox, region_name = await _geocode_region(req.message)
        if bbox:
            return {
                "response": f"Navigating to {region_name}.",
                "action": None,
                "bbox": bbox,
                "region_name": region_name,
            }
        else:
            return {"response": "Could not find that location. Try a different name.", "action": None}

    # Otherwise run through the LLM
    provider = _get_chat_provider()
    if not provider:
        return {
            "response": "No chat backend configured. Set one of:\n"
                        "- ANTHROPIC_API_KEY (for Claude)\n"
                        "- OPENAI_API_KEY (for OpenAI)\n"
                        "- LOCAL_LLM_URL=http://localhost:11434/v1 + LOCAL_LLM_MODEL=llama3 (for Ollama/vLLM/LM Studio)",
            "action": None,
        }

    system_prompt = _build_system_prompt(req.bbox, latest_results)

    # Get or create session history
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []

    history = chat_sessions[req.session_id]
    history.append({"role": "user", "content": req.message})

    # Keep last 20 messages
    if len(history) > 20:
        history = history[-20:]
        chat_sessions[req.session_id] = history

    try:
        if provider == "anthropic":
            assistant_msg = await _chat_anthropic(system_prompt, history)

        elif provider == "openai":
            assistant_msg = await _chat_openai_compatible(system_prompt, history)

        elif provider == "local":
            assistant_msg = await _chat_openai_compatible(
                system_prompt, history,
                base_url=os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("LOCAL_LLM_API_KEY", "none"),
                model=os.environ.get("LOCAL_LLM_MODEL", "llama3"),
            )
        else:
            return {"response": f"Unknown chat provider: {provider}", "action": None}

        history.append({"role": "assistant", "content": assistant_msg})

        # Also try geocoding the user message for any place mentions
        result = {"response": assistant_msg, "action": _detect_action(assistant_msg)}
        bbox, region_name = await _geocode_region(req.message)
        if bbox:
            result["bbox"] = bbox
            result["region_name"] = region_name

        return result

    except Exception as e:
        return {"response": f"Chat error: {str(e)}", "action": None}


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
