"""Model precision benchmark: Qwen via vLLM (bf16 vs FP8) and SAM ViT-H (fp32/bf16/fp16).

LLM — runs on the host against the running `vllm` container:
    VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 docker compose up -d --wait vllm
    .venv/bin/python bench_models.py --target llm --out bench_llm_bf16.json
    docker compose up -d --wait vllm        # FP8 is the compose default
    .venv/bin/python bench_models.py --target llm --out bench_llm_fp8.json

SAM — runs inside the app image on the GPU, no vLLM needed:
    docker compose run --rm --no-deps -v $PWD:/app agrisense \
        python bench_models.py --target sam --out bench_sam.json
"""

import argparse
import json
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

BBOX = [63.2, 52.35, 63.45, 52.48]      # Kostanay preset
DATE = "2023-07-01/2023-07-31"          # peak season


def med(xs, nd=2):
    return round(statistics.median(xs), nd)


# --- LLM ----------------------------------------------------------------------

TIMED_PROMPT = "Explain NDVI, NDWI and EVI and when to use each for crop monitoring."
QUALITY_PROMPTS = ["hi", "What is NDVI?", "How does change detection work?"]
LOG_FACTS = {                            # last match wins (the most recent start)
    "model": r"model='([^']+)'",
    "quantization": r"quantization=(\w+)",
    "weights_gib": r"Model loading took ([\d.]+) GiB",
    "load_s": r"Model loading took [\d.]+ GiB (?:memory )?and ([\d.]+) seconds",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
}


def vllm_facts(container):
    p = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
    text = p.stdout + p.stderr
    return {k: hits[-1].replace(",", "") for k, pat in LOG_FACTS.items()
            if (hits := re.findall(pat, text))}


def stream(url, system, user, max_tokens, ignore_eos=False):
    import httpx
    body = {"model": "qwen", "max_tokens": max_tokens, "temperature": 0, "seed": 0,
            "ignore_eos": ignore_eos, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    t0, t_first, text, usage = time.perf_counter(), None, [], {}
    with httpx.stream("POST", f"{url}/chat/completions", json=body, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices", []):
                if tok := c.get("delta", {}).get("content"):
                    t_first = t_first or time.perf_counter()
                    text.append(tok)
    t_end = time.perf_counter()
    n = usage.get("completion_tokens", len(text))
    return {"ttft_ms": (t_first - t0) * 1000, "tps": (n - 1) / (t_end - t_first),
            "tokens": n, "prompt_tokens": usage.get("prompt_tokens"), "text": "".join(text)}


def run_llm(args):
    import server                        # the app's real system prompt
    ip = subprocess.run(
        ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
         args.container], capture_output=True, text=True, check=True).stdout.strip()
    url = f"http://{ip}:8000/v1"         # never localhost:8000 — that is triton
    system = server._build_system_prompt(BBOX, {})

    report = {"load": vllm_facts(args.container)}
    print("load ", report["load"])
    stream(url, system, "hi", 16)        # warmup: absorbs the first-request kernel JIT

    runs = [stream(url, system, TIMED_PROMPT, 200, ignore_eos=True) for _ in range(args.repeats)]
    report["batch1"] = {"ttft_ms": med([r["ttft_ms"] for r in runs], 1),
                        "decode_tok_s": med([r["tps"] for r in runs], 1),
                        "prompt_tokens": runs[0]["prompt_tokens"]}
    print("b=1  ", report["batch1"])

    n = args.concurrency
    with ThreadPoolExecutor(n) as pool:
        t0 = time.perf_counter()
        rs = list(pool.map(lambda _: stream(url, system, TIMED_PROMPT, 200, ignore_eos=True),
                           range(n)))
        wall = time.perf_counter() - t0
    report[f"batch{n}"] = {"total_tok_s": round(sum(r["tokens"] for r in rs) / wall, 1),
                           "ttft_ms": med([r["ttft_ms"] for r in rs], 1)}
    print(f"b={n}  ", report[f"batch{n}"])

    report["answers"] = {p: stream(url, system, p, 120)["text"] for p in QUALITY_PROMPTS}
    return report


# --- SAM ----------------------------------------------------------------------

def run_sam(args):
    import gc
    import importlib
    import os
    import torch
    from shapely.geometry import shape
    import satellite

    rgb, geo = satellite.download_rgb(BBOX, DATE)   # the same call as _run_sam_load
    w, s, e, n = BBOX
    points = [[w + fx * (e - w), n - fy * (n - s)]
              for fx, fy in [(.3, .3), (.7, .3), (.5, .5), (.3, .7), (.7, .7)]]

    report, ref = {"image": list(rgb.shape)}, None
    for name in args.dtypes.split(","):
        os.environ["SAM_DTYPE"] = name
        import sam_segmenter
        importlib.reload(sam_segmenter)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

        seg = sam_segmenter.SamSegmenter()
        t0 = time.perf_counter(); seg.load(); load_s = time.perf_counter() - t0
        weights = sum(p.numel() * p.element_size() for p in seg.predictor.model.parameters())

        seg.set_image(rgb, geo); torch.cuda.synchronize()           # warmup
        enc = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            seg.set_image(rgb, geo); torch.cuda.synchronize()
            enc.append(time.perf_counter() - t0)

        clicks, polys = [], []
        for p in points:
            t0 = time.perf_counter()
            r = seg.predict(points=[p], labels=[1])
            clicks.append((time.perf_counter() - t0) * 1000)
            polys.append(shape(r["geojson"]["geometry"]).buffer(0) if r["geojson"] else None)

        row = {"weights_gib": round(weights / 2**30, 2), "load_s": round(load_s, 2),
               "encode_s": med(enc, 3), "click_ms": med(clicks, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if ref is None:
            ref = polys
        else:
            ious = [1.0 if a is None and b is None else
                    0.0 if a is None or b is None else
                    a.intersection(b).area / a.union(b).area
                    for a, b in zip(ref, polys)]
            row["iou_vs_fp32_mean"] = round(sum(ious) / len(ious), 4)
            row["iou_vs_fp32_min"] = round(min(ious), 4)
        report[name] = row
        print(f"{name:5s}", row, flush=True)
        del seg

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["llm", "sam"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--container", default="agrisense-vllm")
    ap.add_argument("--dtypes", default="fp32,bf16,fp16")   # fp32 first: the reference
    args = ap.parse_args()

    report = run_llm(args) if args.target == "llm" else run_sam(args)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
