"""Check the Prithvi crop map against ESA WorldCover 2021, an independent 10 m land-cover map.

Scores crop vs non-crop per pixel (Prithvi classes 0-4 vs WorldCover "Cropland"),
on two Kazakh presets and one in-domain European wheat region. Runs in the
prithvi container against the live service:
    docker compose --profile prithvi exec prithvi python3 validate_worldcover.py \
        > bench_prithvi_worldcover.json
"""
import base64, json, sys, time, urllib.request
from functools import partial
import numpy as np, rasterio, pystac_client, planetary_computer
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

REGIONS = {  # name: bbox
    "Kostanay (KZ)": [63.2, 52.35, 63.45, 52.48],
    "Akmola (KZ)": [69.0, 51.35, 69.25, 51.48],
    "Beauce (FR, in-domain)": [1.45, 48.25, 1.65, 48.35],
}
YEAR = 2021
CL = ["Cereals", "Root", "Industrial", "Pulses/veg", "Fodder", "Bareland", "Woodland", "Grassland"]
WC = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Cropland", 50: "Built", 60: "Bare", 80: "Water", 90: "Wetland"}
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=planetary_computer.sign_inplace)
log = partial(print, file=sys.stderr, flush=True)
rows = {}
for name in sys.argv[1:] or REGIONS:
    bbox = REGIONS[name]
    t0 = time.time()
    req = urllib.request.Request("http://localhost:8601/classify", json.dumps({"bbox": bbox, "year": YEAR}).encode(),
                                 {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    if "error" in d:
        log(name, "ERROR", d["error"]); continue
    ca = d["class_array"]
    pred = np.frombuffer(base64.b64decode(ca["b64"]), dtype=ca["dtype"]).reshape(ca["shape"])
    H, W = pred.shape
    wc = np.zeros((H, W), np.uint8)
    item = next(cat.search(collections=["esa-worldcover"], bbox=bbox, datetime="2021").items())
    with rasterio.open(item.assets["map"].href) as src:
        reproject(rasterio.band(src, 1), wc, dst_transform=from_bounds(*bbox, W, H),
                  dst_crs="EPSG:4326", resampling=Resampling.nearest)
    m = (pred >= 0) & (wc > 0)
    p = np.isin(pred, [0, 1, 2, 3, 4])[m]; w = (wc == 40)[m]
    tp, fp, fn, tn = (p & w).sum(), (p & ~w).sum(), (~p & w).sum(), (~p & ~w).sum()
    v, c = np.unique(wc[wc > 0], return_counts=True)
    sel = pred[m & (wc == 40)]; sv, sc = np.unique(sel, return_counts=True)
    row = {
        "seconds": round(time.time() - t0), "timing_s": d["timing_s"], "cell_m": d["cell_m"],
        "prithvi": {l["label"]: l["percent"] for l in d["legend"] if l["percent"] >= 0.5},
        "worldcover": {WC.get(int(k), int(k)): round(100 * n / c.sum(), 1) for k, n in zip(v, c) if 100 * n / c.sum() >= 0.5},
        "crop_share_worldcover": round(100 * w.mean(), 1), "crop_share_prithvi": round(100 * p.mean(), 1),
        "agreement": round(100 * (tp + tn) / m.sum(), 1),
        "majority_baseline": round(100 * max(w.mean(), 1 - w.mean()), 1),
        "crop_recall": round(100 * tp / max(tp + fn, 1), 1),
        "crop_precision": round(100 * tp / max(tp + fp, 1), 1),
        "worldcover_cropland_labelled_as": {CL[int(k)]: round(100 * n / sc.sum()) for k, n in zip(sv, sc) if 100 * n / sc.sum() >= 1},
    }
    rows[name] = row
    log(f"{name}: crop recall {row['crop_recall']}%, precision {row['crop_precision']}%, "
        f"agreement {row['agreement']}% (majority baseline {row['majority_baseline']}%)")
print(json.dumps(rows, indent=2))
