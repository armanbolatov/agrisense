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


class TimeSeriesRequest(BaseModel):
    bbox: list[float]
    # Monthly windows across the 2023 growing season by default.
    date_windows: list[str] = [
        f"2023-{m:02d}-01/2023-{m:02d}-28" for m in range(4, 11)
    ]


class CropMapRequest(BaseModel):
    bbox: list[float]
    date_windows: list[str] = [
        f"2023-{m:02d}-01/2023-{m:02d}-28" for m in range(4, 11)
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
        r'\b(hi+|hello|hey|yo|hiya|please|thanks|thank|you|can|could|would|will|ok|okay|'
        r'select|show|go|navigate|find|zoom|display|open|fly|take|move|center|'
        r'to|on|me|the|region|area|oblast|province)\b',
        '', text, flags=re.IGNORECASE,
    ).strip().strip('.,!?')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return None, None

    async def _search(client, q):
        resp = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": "AgriSense/1.0"},
        )
        return resp.json()

    # Retry once — Nominatim occasionally times out, and a silent miss makes a
    # navigation command look like it failed.
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                results = await _search(client, cleaned)  # worldwide
            break
        except Exception as e:
            print(f"Geocode error (attempt {attempt + 1}): {e}")
            results = None

    if not results:
        return None, None

    try:
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
# NDVI Time Series
# ---------------------------------------------------------------------------

@app.post("/api/timeseries")
async def timeseries(req: TimeSeriesRequest, background_tasks: BackgroundTasks):
    _validate_bbox(req.bbox)
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "type": "timeseries"}
    background_tasks.add_task(_run_timeseries, job_id, req.bbox, req.date_windows)
    return {"job_id": job_id}


def _run_timeseries(job_id, bbox, date_windows):
    try:
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
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "type": "cropmap"}
    background_tasks.add_task(_run_cropmap, job_id, req.bbox, req.date_windows)
    return {"job_id": job_id}


def _cropmap_terramind(job_id, bbox):
    """Try the TerraMind crop model service. Returns result dict or None."""
    tm_url = os.environ.get("TERRAMIND_URL")
    if not tm_url:
        return None
    try:
        import httpx
        jobs[job_id]["status"] = "computing"
        jobs[job_id]["message"] = "Running TerraMind crop model (GPU)..."
        # The model expects three seasonal acquisitions (spring / peak / late).
        date_ranges = [
            "2023-04-20/2023-05-31",
            "2023-07-01/2023-07-31",
            "2023-09-01/2023-09-30",
        ]
        resp = httpx.post(f"{tm_url}/classify",
                          json={"bbox": bbox, "date_ranges": date_ranges}, timeout=240.0)
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            return None
        result.setdefault("bbox", bbox)
        return result
    except Exception:
        traceback.print_exc()
        return None


def _run_cropmap(job_id, bbox, date_windows):
    try:
        # Prefer the TerraMind model; fall back to the rule-based classifier.
        tm = _cropmap_terramind(job_id, bbox)
        if tm is not None:
            latest_results["cropmap"] = {
                "legend": tm["legend"], "months": tm.get("months", []),
                "model": tm.get("model", "terramind_v1_base"),
            }
            jobs[job_id] = {"status": "complete", "type": "cropmap", "result": tm}
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
        latest_results["cropmap"] = {"legend": areas, "months": months}
        jobs[job_id] = {"status": "complete", "type": "cropmap", "result": result}

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e), "type": "cropmap"}


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

    context = "\n".join(context_parts) if context_parts else "No analysis has been run yet."

    return f"""You are the assistant inside AgriSense, a satellite remote-sensing app for
Kazakhstan agriculture (Sentinel-2 imagery from Microsoft Planetary Computer).

STRICT SCOPE — this is your ONLY purpose. You only help with AgriSense's analysis:
- interpreting its results (NDVI, vegetation indices, NDVI time series / phenology,
  and crop classification) shown under "Current state" below,
- explaining the related remote-sensing concepts,
- telling the user which button to click.

If the user asks for ANYTHING outside this scope — writing or debugging code, math,
general knowledge, translation, jokes, chit-chat, other software, etc. — you MUST
refuse and NOT fulfil it. In that case reply with exactly this and nothing else:
"I can only help with the satellite imagery and crop analysis here in AgriSense."

The app buttons are: "Satellite preview", "NDVI index", "NDVI time series",
"Classify crops" (crop classification map), and "SAM segment". To analyse an area the
user first draws a rectangle on the map or picks a Quick region, then clicks a button.

Rules:
- Base every factual claim on the "Current state" data. Never invent numbers or results.
- If no analysis has been run yet, tell the user to select an area and click a button.
- Be concise: 1-3 sentences, plain text, no markdown, no code blocks.

Crop phenology hints: winter crops peak in May-June, spring/summer crops peak in
July-August, grassland/forest stays high and flat, bare/fallow stays low all season.

Current state:
{context}"""


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
    """True if the message is an explicit 'select/go to/show ... <place>' command,
    even when wrapped in greetings/politeness ('hi, select london please',
    'can you go to paris')."""
    import re
    # Strip a leading greeting / "can you" / "please" wrapper, then check whether
    # the remainder starts with a navigation verb.
    stripped = re.sub(
        r'^\s*((hi+|hello|hey|yo|hiya|ok|okay)[\s,!.]+)*'
        r'((can|could|would|will)\s+you[\s,]*)?(please[\s,]*)?',
        '', text, flags=re.IGNORECASE,
    )
    return bool(re.match(
        r'^(select|show|go\s+to|navigate(\s+to)?|find|zoom(\s+to)?|display|open|'
        r'fly\s+to|take\s+me\s+to|move\s+to|center\s+on)\b',
        stripped, re.IGNORECASE,
    ))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Navigation: an explicit "select X" command forces a worldwide lookup; a
    # bare place name is tried opportunistically. Real questions fall through to
    # the LLM (the geocoder filters them out), so the map never jumps on a question.
    nav = _is_navigation_command(req.message)
    bbox, region_name = await _geocode_region(req.message, force=nav)
    if bbox:
        return {
            "response": f"Selected {region_name}. Run any analysis tool, or ask me about it.",
            "action": None,
            "bbox": bbox,
            "region_name": region_name,
        }
    if nav:
        # Explicit command but nothing found — tell the user instead of refusing.
        return {"response": "I couldn't find that place. Try the city or region name again.",
                "action": None}

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
        return {"response": assistant_msg, "action": _detect_action(assistant_msg)}

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
