# AgriSense

Remote sensing analysis platform built on Sentinel-2 satellite data from Microsoft Planetary Computer.

## Features

- **Satellite Preview** — RGB imagery for any region
- **NDVI / NDWI / EVI** — Vegetation and water index computation
- **NDVI Time Series** — Seasonal vegetation dynamics (one scene per month) as an interactive **Plotly** chart (NDVI + toggleable NDWI/EVI), with a rule-based phenology interpretation (crop-type hint)
- **Crop Classification** — Crop map from **Prithvi-EO-2.0** (IBM/NASA foundation model) fine-tuned on **Sen4Map** crops: 8 classes from 12 monthly Sentinel-2 frames, GPU, when its optional service is running; automatic fallback to a built-in rule-based seasonal classifier
- **Season Slider** — Scrub the multi-temporal imagery (Spring / Peak / Late) over the classified map to see the season change
- **Change Detection** — NDVI difference between two years' peak seasons (greener / browner map + stats)
- **SAM Segmentation** — Interactive click-to-segment using Segment Anything Model (GPU)
- **Any year** — pick the analysis year (2018–2024); all tools use it
- **Export** — download results as **GeoTIFF** (NDVI / crop map / change), **CSV** (time series), or **GeoJSON** (SAM segments)
- **AI Chat (agentic)** — an instruction-tuned **Qwen3** (via vLLM) that *runs analyses on command* ("classify crops in Pavlodar", "show NDVI time series"), navigates to **any place worldwide** (with a Kazakhstan preference), interprets results, explains how the tools work, and answers general questions — streamed token-by-token
- **Result caching** — re-running an analysis on the same area + dates is instant (no re-download); the preset regions are pre-cached at startup
- **MCP Server** — Claude Code integration via Model Context Protocol

## Quick Start (Docker, recommended)

Requires Docker + an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(one-time setup: `sudo ./install-gpu.sh`).

```bash
git clone https://github.com/armanbolatov/agrisense.git
cd agrisense
docker compose up -d --wait
```

Open http://localhost:8501 in your browser.

The default `up` brings up a **self-contained** two-container stack:

| Service | What it does |
|---|---|
| `vllm` | Serves **Qwen3-4B-Instruct** in FP8 (OpenAI-compatible API) on the GPU for chat |
| `agrisense` | The web app; SAM on the GPU, satellite data, indices, rule-based crop map |

**SAM checkpoint** — mount it from the host (read-only). Fetch one once:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
# vit_b (light, 375 MB) — set SAM_MODEL_TYPE=vit_b SAM_CHECKPOINT=/models/sam_vit_b_01ec64.pth
curl -L -o ~/.cache/torch/hub/checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# vit_h (best, 2.4 GB, default) :
curl -L -o ~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Change the chat model: `VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct docker compose up -d vllm`
(default is `Qwen/Qwen3-4B-Instruct-2507-FP8`; see [Model precision](#model-precision)).

### Crop model: Prithvi-EO-2.0 on Sen4Map (optional)

"Classify crops" uses a Prithvi-EO-2.0-300M checkpoint fine-tuned with terratorch
on the Sen4Map crop task when its service is up; otherwise the app runs the
built-in rule-based classifier, so the base stack works on its own.

The model labels a 150 m window from 12 monthly frames of 10 Sentinel-2 bands
with one of 8 classes: cereals, root crops, industrial crops, pulses/vegetables,
fodder, bareland, woodland, grassland. The service slides that window over the
area, one label per 150-600 m cell (`PRITHVI_STRIDE`, auto by area size).
Validation accuracy of the checkpoint: 84.3% overall, 64.9% class-balanced.

Put the checkpoint in `~/models/prithvi-sen4map/` (or set `PRITHVI_CKPT_DIR`),
then start with the profile. The image builds on `agrisense-terramind:latest`.

```bash
docker compose --profile prithvi up -d --wait
```

A request takes 60-90 s: 35-65 s to download 12 months x 11 assets from
Planetary Computer, 15-32 s of GPU inference. The app caches the result.

Two things matter for correct input. Scenes from ESA processing baseline 04.00
(2022 onward) carry a +1000 offset, removed in `satellite.read_band` for every
tool; without it the model calls most land bare soil. Inputs are min-max scaled
with the Sen4Map quantiles, not mean/std.

**How well it transfers.** Sen4Map is European (LUCAS 2018), so the service was
checked against ESA WorldCover 2021, an independent 10 m land-cover map
(`prithvi/validate_worldcover.py`, results in `bench_prithvi_worldcover.json`):

| Region | WorldCover cropland | Cropland found as crop | Prithvi crop that is cropland |
|---|---|---|---|
| Beauce, France (in-domain) | 87% | 96% | 88% |
| Akmola, Kazakhstan | 65% | 69% | 76% |
| Kostanay, Kazakhstan (mostly steppe) | 14% | 10% | 22% |

In Europe it works as trained. In dense Kazakh cropland it finds most fields,
mostly as cereals. Kazakh steppe is the weak spot: it is often labelled woodland
or bareland, and scattered fields inside it are missed. Treat the Kazakhstan
output as indicative, not validated.

The older TerraMind service (`--profile terramind`) is still wired as a second
choice but needs its fine-tuned checkpoint in `../terramind/output`.

## Hosting & remote access

Run it on any machine with an NVIDIA GPU + the toolkit. The app binds
`0.0.0.0:${AGRISENSE_PORT:-8501}`, so it's reachable on the host's network.

- **Same LAN:** others open `http://<host-LAN-IP>:8501`.
- **Public IP / cloud VM:** open the port in the firewall / security group, then
  `http://<public-ip>:8501`. Change the port with `AGRISENSE_PORT=80`.
- **Tunnel (easiest, no firewall changes):**
  `cloudflared tunnel --url http://localhost:8501` or `ngrok http 8501` → a public
  HTTPS URL.

⚠️ **The app has no authentication.** Anyone who can reach the port can run
analyses that use your GPU, the LLM, and Planetary Computer bandwidth. Before
exposing it publicly, put it behind a reverse proxy with auth (nginx/Caddy basic
auth, or Cloudflare Access), or restrict the firewall to known IPs. Keep the
`vllm`/`prithvi`/`terramind` ports internal (they are, by default — don't publish them).

### Small GPU

The full stack (Qwen3-4B + SAM `vit_h` + Prithvi) needs ~22 GB VRAM. For an
8–16 GB GPU, create a `.env` next to `docker-compose.yml`:

```env
VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
VLLM_GPU_UTIL=0.25
SAM_MODEL_TYPE=vit_b
SAM_CHECKPOINT=/models/sam_vit_b_01ec64.pth
```

and run the default `docker compose up -d` (skip Prithvi). That's ~4 GB VRAM
(Qwen-0.5B ≈ 2 GB + SAM `vit_b` ≈ 1.5 GB). For no GPU at all, build the CPU image
(`docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t agrisense .`)
and point chat at a hosted API (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).

## Quick Start (local, without Docker)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync              # install dependencies
./start.sh           # start with Ollama (default)
```

Open http://localhost:8000.

```bash
./start.sh                                     # Ollama (qwen2.5:3b)
ANTHROPIC_API_KEY=sk-... ./start.sh anthropic  # Anthropic
OPENAI_API_KEY=sk-... ./start.sh openai        # OpenAI
```

## Usage

1. Draw a rectangle on the map or type a region name in the chat (e.g. `taldykorgan`, `pavlodar`)
2. Click any analysis button: Satellite Preview, NDVI, NDVI Time Series, or SAM Load Area
3. For SAM: after loading, click on the map to segment objects (green = include, red = exclude)
4. Ask the AI chat questions about the analysis results — e.g. after running a time
   series: *"When does vegetation peak and what crop type does the profile suggest?"*

## Performance

Every analysis is dominated by network I/O against Planetary Computer, not by GPU
work. Three changes cut that I/O (see `bench.py`, Kostanay preset, 17x14 km,
median of 3 runs):

- **Scene + band memoization** (`satellite.py`) — one signed STAC client per
  process, and scene searches and band reads cached for 30 min. Preview, NDVI,
  SAM and change detection over the same area and date now share one download.
- **Resolution cap** — native reads are capped at 1024 px on the long edge, which
  rasterio serves from the COG overviews. Map overlays and SAM (which resizes to
  1024 internally) look identical, and the payloads shrink by ~60%.
- **Band subsetting** — change detection asks for red + NIR instead of all four
  bands, halving its downloads.

| Analysis (cold cache) | Before | After |
|---|---|---|
| Satellite preview | 8.2 s | **5.5 s** |
| NDVI | 7.8 s | **6.6 s** |
| NDVI time series (7 months) | 12.5 s | **10.1 s** |
| Change detection (2 years) | 14.5 s | **8.8 s** |
| Preview -> NDVI -> preview, one area | 23.1 s | **7.9 s** |

Overlay payloads: preview 7.0 MB -> 2.5 MB, NDVI 3.2 MB -> 1.2 MB base64 PNG.

The cap also applies to GeoTIFF export, so a large area exports at ~15 m/px
instead of 10 m. Set `AGRISENSE_MAX_PX=0` for full native resolution, or any
pixel count to trade payload size against detail.

Reproduce the measurement:

```bash
docker run --rm --network host -v $PWD:/app -w /app agrisense:latest \
    python bench.py --out bench_after.json
```

### Model precision

Both GPU models now run at reduced precision by default (`bench_models.py`,
RTX 5090, median of 5 runs after warmup).

**Qwen3-4B-Instruct via vLLM: bf16 -> FP8.** The official
`Qwen/Qwen3-4B-Instruct-2507-FP8` checkpoint (W8A8, block-wise weight scales)
under the same `--gpu-memory-utilization 0.4` budget and the app's real
657-token system prompt:

| | bf16 | **FP8** | Change |
|---|---|---|---|
| Weights in GPU memory | 7.64 GiB | **4.27 GiB** | -44% |
| KV cache | 32,608 tokens | **44,032 tokens** | +35% |
| Parallel 4k-token requests that fit | 7.96 | **10.75** | +35% |
| Decode, 1 request | 166 tok/s | **177 tok/s** | +6% |
| Decode, 8 concurrent requests | 908 tok/s | **1,088 tok/s** | +20% |
| Time to first token | 51 ms | 62 ms | +11 ms |

The main win is memory: the same budget holds 35% more context. Answers to the
same prompts at temperature 0 are equivalent in content (stored under `answers`
in the JSON files). Back to bf16: `VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507`.

**SAM ViT-H: fp32 -> fp16 image encoder.** The ViT encoder is 99% of the weights
and the whole 1024 px encode; the prompt decoder stays fp32. Measured on a real
Kostanay tile with 5 point prompts, agreement as polygon IoU against fp32:

| | fp32 | bf16 | **fp16** |
|---|---|---|---|
| Weights | 2.39 GiB | 1.20 GiB | **1.20 GiB** |
| Peak GPU memory | 5.61 GiB | 3.80 GiB | **2.82 GiB** |
| Image encode | 182 ms | 87 ms | **77 ms** |
| Click to polygon | 99 ms | 83 ms | **74 ms** |
| Mask IoU vs fp32, mean / min | 1 | 0.992 / 0.972 | **0.9998 / 0.9996** |

fp16 beats bf16 on every column on this GPU, so it is the default. Set
`SAM_DTYPE=fp32` for the original model.

Reproduce:

```bash
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 docker compose up -d --wait vllm
.venv/bin/python bench_models.py --target llm --out bench_llm_bf16.json
docker compose up -d --wait vllm
.venv/bin/python bench_models.py --target llm --out bench_llm_fp8.json
docker compose run --rm --no-deps -v $PWD:/app agrisense \
    python bench_models.py --target sam --out bench_sam.json
```

## Architecture

```
server.py          FastAPI backend — routes, job system, chat, geocoding
satellite.py       Sentinel-2 data access via Planetary Computer (parallel band reads)
geo_utils.py       Coordinate transforms, raster-to-vector, PNG encoding, charts
indices.py         NDVI, NDWI, EVI computation + phenology heuristics
cropmap.py         Rule-based seasonal crop classifier (fallback) + season stack
sam_segmenter.py   SAM wrapper with georeferencing
mcp_server.py      MCP tools for Claude Code
static/            Frontend (Leaflet map + vendored Leaflet/Plotly, neutral UI)
docker-compose.yml vllm + agrisense; prithvi / terramind behind their profiles
Dockerfile         App image (CUDA torch by default, TORCH_INDEX arg for CPU)
prithvi/           Prithvi-EO-2.0 crop service + WorldCover validation script
../terramind/      Optional TerraMind crop service (separate repo + checkpoint)
install-gpu.sh     One-time NVIDIA Container Toolkit setup
start.sh           Local (non-Docker) launch script
```

## Environment Variables

| Variable | Description |
|---|---|
| `CHAT_PROVIDER` | `local`, `anthropic`, or `openai` |
| `LOCAL_LLM_URL` | OpenAI-compatible endpoint (Docker default: `http://vllm:8000/v1`) |
| `LOCAL_LLM_MODEL` | Model name for local LLM (Docker default: `qwen`) |
| `VLLM_MODEL` | HuggingFace model served by the vLLM container (default `Qwen/Qwen3-4B-Instruct-2507-FP8`) |
| `VLLM_EXTRA_ARGS` | Extra vLLM flags, e.g. `--quantization fp8` for online FP8 on bf16 weights |
| `VLLM_GPU_UTIL` | Fraction of GPU memory vLLM reserves (default `0.4`; lower for small GPUs) |
| `AGRISENSE_PORT` | Host port for the web app (default `8501`) |
| `SAM_MODEL_TYPE` | `vit_h` (default) or `vit_b` (lighter) |
| `SAM_DTYPE` | SAM image-encoder precision on GPU: `fp16` (default), `bf16`, or `fp32` |
| `SAM_CHECKPOINT` | In-container path to SAM weights (default `/models/sam_vit_h_4b8939.pth`) |
| `SAM_CHECKPOINT_DIR` | Host dir mounted to `/models` (default `~/.cache/torch/hub/checkpoints`) |
| `PRITHVI_URL` | Prithvi crop service, tried first (default `http://prithvi:8601`) |
| `PRITHVI_CKPT_DIR` | Host dir holding `prithvi_sen4map_crops.ckpt` (default `~/models/prithvi-sen4map`) |
| `PRITHVI_STRIDE` | Pixels per label cell, multiple of 15 (default `0` = auto) |
| `PRITHVI_MAX_WINDOWS` | Auto stride keeps the window count under this (default `2000`) |
| `PRITHVI_BATCH` | Windows per GPU batch (default `8`; ~6 GB) |
| `TERRAMIND_URL` | TerraMind crop service, tried second (default `http://terramind:8600`) |
| `TERRAMIND_DIR` | Host path to the terramind repo (for the checkpoint mount) |

## License

MIT
