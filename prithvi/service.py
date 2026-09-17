"""Prithvi-EO-2.0 crop-classification microservice for AgriSense.

Serves a Prithvi-EO-2.0-300M checkpoint fine-tuned with terratorch on Sen4Map
crops (8 classes). The model classifies a 15x15 px (150 m) window from 12
monthly Sentinel-2 frames, so the service slides that window over the area and
paints one label per cell.

Contract matches the old TerraMind service, so AgriSense needs no changes:
    POST /classify {bbox, date_ranges | year} -> {bbox, class_image, legend,
        months, monthly, class_array, model}
    GET  /health
"""
import base64
import calendar
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Remote COG reads fail transiently; let GDAL retry. Must precede rasterio.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("AGRISENSE_MAX_PX", "0")  # the model needs native 10 m pixels

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from satellite import _ensure_size, read_band, search_sentinel2

CKPT = os.environ.get("PRITHVI_CKPT", "/models/prithvi_sen4map_crops.ckpt")
BATCH = int(os.environ.get("PRITHVI_BATCH", "8"))          # throughput is flat in batch
STRIDE = int(os.environ.get("PRITHVI_STRIDE", "0"))        # px; 0 = auto
MAX_WINDOWS = int(os.environ.get("PRITHVI_MAX_WINDOWS", "2000"))
WORKERS = int(os.environ.get("PRITHVI_WORKERS", "12"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH, SIZE, MIN_MONTHS = 15, 224, 6
MODEL_NAME = "prithvi_eo_v2_300_sen4map_crops"

# Training band order (Sen4Map dataset_bands) with native resolution in metres.
BANDS = [("B02", 10), ("B03", 10), ("B04", 10), ("B05", 20), ("B06", 20),
         ("B07", 20), ("B08", 10), ("B8A", 20), ("B11", 20), ("B12", 20)]
# Sen4Map min-max quantiles (terratorch/datasets/sen4map.py). Training normalized
# with these and clipped to [0, 1]; mean/std input makes the model collapse.
Q_LO = np.array([67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27], np.float32)
Q_HI = np.array([2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72,
                 5140.84, 4414.45], np.float32)

CLASSES = ["Cereals", "Root crops", "Industrial crops", "Pulses, vegetables, flowers",
           "Fodder crops", "Bareland", "Woodland and shrubland", "Grassland"]
COLORS = ["#e3b505", "#f28e2b", "#9d4edd", "#e377c2",
          "#a6d96a", "#b08968", "#1f4e1f", "#4daf4a"]

app = FastAPI(title="Prithvi Crop Service")
_model = None


def get_model():
    global _model
    if _model is None:
        from terratorch.tasks import ClassificationTask
        hp = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["hyper_parameters"]
        # pretrained=False: the checkpoint already holds the fine-tuned backbone,
        # so skip downloading the base Prithvi weights first.
        task = ClassificationTask.load_from_checkpoint(
            CKPT, map_location="cpu", strict=True,
            model_args={**hp["model_args"], "pretrained": False})
        _model = task.model.eval().to(DEVICE)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.float16,
                                                     enabled=DEVICE == "cuda"):
            _model(torch.zeros(BATCH, len(BANDS), 12, SIZE, SIZE, device=DEVICE))  # warmup
    return _model


def _read_retry(item, asset, bbox, res, target_size=None, tries=4):
    """read_band with retries — remote COG reads fail transiently."""
    for attempt in range(tries):
        try:
            return read_band(item, asset, bbox, res, target_size=target_size)[0]
        except Exception:  # noqa: BLE001 (rasterio/curl transient errors)
            if attempt == tries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def _harmonize(item, arr):
    """Undo ESA's +1000 offset on scenes from processing baseline 04.00 (2022+).
    Sen4Map was built from 2018 scenes without it, and unharmonized input is so
    bright that the model calls nearly everything bare soil."""
    if float(item.properties.get("s2:processing_baseline") or 0) < 4.0:
        return arr
    return np.clip(arr - 1000, 0, None)   # new array: cached reads stay untouched


def _find_scene(bbox, window):
    """Least-cloudy scene in the window, relaxing the cloud limit if needed."""
    for max_cloud in (20, 60, 100):
        if item := search_sentinel2(bbox, window, max_cloud=max_cloud):
            return item
    return None


def download_stack(bbox, year):
    """Raw reflectance [10 bands, 12 months, H, W] plus a [H, W] valid mask."""
    windows = [f"{year}-{m:02d}-01/{year}-{m:02d}-{calendar.monthrange(year, m)[1]}"
               for m in range(1, 13)]
    with ThreadPoolExecutor(6) as pool:
        items = list(pool.map(lambda w: _find_scene(bbox, w), windows))
    present = [t for t, item in enumerate(items) if item is not None]
    if len(present) < MIN_MONTHS:
        raise ValueError(f"Only {len(present)} of 12 months have imagery for {year}.")

    H, W = _read_retry(items[present[0]], "B02", bbox, 10).shape
    if min(H, W) < PATCH:
        raise ValueError("Area too small for a 150 m classification window.")
    raw = np.zeros((len(BANDS), 12, H, W), np.float32)

    def fetch(job):
        t, b = job
        if b is None:  # scene classification: nearest-neighbour 20 m -> 10 m, never bilinear
            scl = _read_retry(items[t], "SCL", bbox, 20)
            return t, b, _ensure_size(np.repeat(np.repeat(scl, 2, 0), 2, 1), (H, W))
        band = _read_retry(items[t], BANDS[b][0], bbox, BANDS[b][1], target_size=(H, W))
        return t, b, _harmonize(items[t], band)

    jobs = [(t, b) for t in present for b in range(len(BANDS))] + [(t, None) for t in present]
    with ThreadPoolExecutor(WORKERS) as pool:
        for t, b, arr in pool.map(fetch, jobs):   # in order: every band lands before its SCL
            if b is None:
                raw[:, t, arr >= 9] = 0          # the training cloud/snow mask
            else:
                raw[b, t] = arr

    missing = [t for t in range(12) if t not in present]
    if missing:  # as in training: an empty month gets the median over the year
        raw[:, missing] = np.median(raw[:, present], axis=1, keepdims=True)
    return raw, raw.any(axis=(0, 1))


def _pick_stride(H, W):
    if STRIDE:
        return STRIDE
    return next((s for s in (15, 30, 45, 60)
                 if math.ceil(H / s) * math.ceil(W / s) <= MAX_WINDOWS), 60)


@torch.inference_mode()
def infer(stack, valid, stride):
    """Classify a 15 px window centred in every stride x stride cell; -1 = nodata."""
    model = get_model()
    C, T, H, W = stack.shape
    cls = np.full((H, W), -1, np.int16)
    src = torch.from_numpy(stack)
    cells = [(cy, cx) for cy in range(0, H, stride) for cx in range(0, W, stride)
             if valid[cy:cy + stride, cx:cx + stride].any()]
    off = (stride - PATCH) // 2
    for i in range(0, len(cells), BATCH):
        chunk = cells[i:i + BATCH]
        x = torch.stack([src[:, :, (y := min(cy + off, H - PATCH)):y + PATCH,
                             (xo := min(cx + off, W - PATCH)):xo + PATCH]
                         for cy, cx in chunk])
        x = x.reshape(len(chunk), C * T, PATCH, PATCH).to(DEVICE)
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True)
        x = x.reshape(len(chunk), C, T, SIZE, SIZE).clamp_(0, 1)  # training: resize, then clip
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            pred = model(x).output.argmax(1).cpu().numpy()
        for (cy, cx), p in zip(chunk, pred):
            cls[cy:cy + stride, cx:cx + stride] = p
    cls[~valid] = -1
    return cls, len(cells)


def _png(img):
    buf = io.BytesIO()
    plt.imsave(buf, img, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def _rgb_png(month):
    """RGB PNG from one [10, H, W] month (B04, B03, B02), 2-98 % stretch."""
    rgb = np.stack([month[2], month[1], month[0]], axis=-1).astype(float)
    valid = rgb[rgb > 0]
    p2, p98 = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
    return _png(np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1))


def _class_png(class_map):
    """Coloured RGBA PNG from an int class map (-1 = transparent nodata)."""
    rgba = np.zeros((*class_map.shape, 4), np.uint8)
    for cid, color in enumerate(COLORS):
        m = class_map == cid
        rgba[m, :3] = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        rgba[m, 3] = 255
    return _png(rgba)


class ClassifyRequest(BaseModel):
    bbox: list[float]
    date_ranges: list[str] = []
    year: int | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/classify")
def classify(req: ClassifyRequest):
    year = req.year or (int(req.date_ranges[0][:4]) if req.date_ranges else 2023)
    t0 = time.perf_counter()
    try:
        raw, valid = download_stack(req.bbox, year)
    except ValueError as e:
        return {"error": str(e)}
    t_download = time.perf_counter() - t0

    monthly = [{"month": f"{year}-{m:02d}", "image": _rgb_png(raw[:, m - 1])} for m in (4, 7, 9)]
    raw -= Q_LO[:, None, None, None]
    raw /= (Q_HI - Q_LO)[:, None, None, None]

    stride = _pick_stride(*valid.shape)
    t1 = time.perf_counter()
    class_map, windows = infer(raw, valid, stride)
    t_infer = time.perf_counter() - t1
    print(f"classify {valid.shape[1]}x{valid.shape[0]} px, cell {stride * 10} m, "
          f"{windows} windows: download {t_download:.1f}s, inference {t_infer:.1f}s", flush=True)

    n_valid = int((class_map >= 0).sum()) or 1
    legend = [{"id": cid, "label": name, "color": color,
               "percent": round(100.0 * int((class_map == cid).sum()) / n_valid, 1)}
              for cid, (name, color) in enumerate(zip(CLASSES, COLORS))]
    return {
        "bbox": req.bbox,
        "class_image": _class_png(class_map),
        "legend": legend,
        "months": [m["month"] for m in monthly],
        "monthly": monthly,
        "model": MODEL_NAME,
        "cell_m": stride * 10,
        "windows": windows,
        "timing_s": {"download": round(t_download, 1), "inference": round(t_infer, 1)},
        # Raw class map for GeoTIFF export on the app side.
        "class_array": {"dtype": "int16", "shape": list(class_map.shape),
                        "b64": base64.b64encode(class_map.tobytes()).decode()},
    }


if __name__ == "__main__":
    print(f"Loading {MODEL_NAME} from {CKPT} ...", flush=True)
    get_model()
    print("Model ready. Serving on :8601", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8601)
