"""AgriSense performance benchmark.

Times the satellite-data paths that dominate every analysis and reports the
work they do (band reads, pixels, overlay payload size).

    docker run --rm --network host -v $PWD:/app -w /app agrisense:latest \
        python bench.py --out bench_after.json

Two sections:
  cold  — caches cleared before each op (measures bytes fetched per analysis)
  warm  — preview -> ndvi -> preview on one area (measures cross-tool reuse)
"""

import argparse
import json
import statistics
import time

BBOX = [63.2, 52.35, 63.45, 52.48]      # Kostanay preset
DATE = "2023-07-01/2023-07-31"          # peak season
YEAR = 2023

import satellite
from geo_utils import index_to_colored_png, rgb_to_png
from indices import compute_ndvi

_reads = {"n": 0, "px": 0}
_real_read_band = satellite.read_band


def _counting_read_band(*a, **kw):
    out = _real_read_band(*a, **kw)
    _reads["n"] += 1
    _reads["px"] += int(out[0].size)
    return out


satellite.read_band = _counting_read_band


def _clear():
    """Drop any caches the optimized build added (no-op on the baseline)."""
    for name in ("clear_caches",):
        fn = getattr(satellite, name, None)
        if fn:
            fn()


def timed(fn, clear=True):
    if clear:
        _clear()
    _reads["n"] = _reads["px"] = 0
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return dt, {"reads": _reads["n"], "read_mpx": round(_reads["px"] / 1e6, 2)}, out


# --- the operations, as the server runs them --------------------------------

def op_preview():
    rgb, _ = satellite.download_rgb(BBOX, DATE)
    return {"png_kb": round(len(rgb_to_png(rgb)) / 1024, 1),
            "out_mpx": round(rgb.shape[0] * rgb.shape[1] / 1e6, 2)}


def op_ndvi():
    bands, _ = satellite.download_bands_for_indices(BBOX, DATE)
    ndvi = compute_ndvi(bands["nir"], bands["red"])
    return {"png_kb": round(len(index_to_colored_png(ndvi, BBOX)) / 1024, 1),
            "out_mpx": round(ndvi.size / 1e6, 2)}


def op_timeseries():
    windows = [f"{YEAR}-{m:02d}-01/{YEAR}-{m:02d}-28" for m in range(4, 11)]
    return {"points": len(satellite.download_ndvi_series(BBOX, windows))}


def op_change():
    for year in (2020, 2023):
        kw = {"bands": ("red",)} if _supports_subset() else {}
        satellite.download_bands_for_indices(BBOX, f"{year}-07-01/{year}-07-31", **kw)
    return {}


def _supports_subset():
    import inspect
    return "bands" in inspect.signature(satellite.download_bands_for_indices).parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    report = {"cold": {}, "warm": {}}

    for name, fn in (("preview", op_preview), ("ndvi", op_ndvi),
                     ("timeseries", op_timeseries), ("change", op_change)):
        runs, meta, extra = [], None, None
        for _ in range(args.repeats):
            dt, meta, extra = timed(fn)          # caches cleared -> always cold
            runs.append(dt)
        report["cold"][name] = {"median_s": round(statistics.median(runs), 2),
                                "runs_s": [round(r, 2) for r in runs], **meta,
                                **(extra or {})}
        print(f"cold  {name:11s} {report['cold'][name]['median_s']:6.2f}s  "
              f"reads={meta['reads']:2d}  read_mpx={meta['read_mpx']}")

    # Warm: one user session on one area — preview, then NDVI, then preview again.
    _clear()
    for step, fn in (("preview", op_preview), ("ndvi", op_ndvi), ("preview_again", op_preview)):
        dt, meta, _ = timed(fn, clear=False)
        report["warm"][step] = {"s": round(dt, 2), **meta}
        print(f"warm  {step:13s} {dt:6.2f}s  reads={meta['reads']:2d}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
