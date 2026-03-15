# AgriSense

Remote sensing analysis platform built on Sentinel-2 satellite data from Microsoft Planetary Computer.

## Features

- **Satellite Preview** — RGB imagery for any region
- **NDVI / NDWI / EVI** — Vegetation and water index computation
- **SAM Segmentation** — Interactive click-to-segment using Segment Anything Model
- **AI Chat** — Natural language interface with geocoding (type a place name to navigate)
- **MCP Server** — Claude Code integration via Model Context Protocol

## Quick Start

```bash
# Run (uv will install dependencies automatically)
uv run python server.py

# Or use the start script with AI chat (Ollama)
./start.sh

# Or specify a chat provider
ANTHROPIC_API_KEY=sk-... ./start.sh anthropic
OPENAI_API_KEY=sk-... ./start.sh openai
```

Open http://localhost:8000 in your browser.

## Usage

1. Draw a rectangle on the map or type a region name in the chat (e.g. `taldykorgan`, `pavlodar`)
2. Click any analysis button: Satellite Preview, NDVI, or SAM Load Area
3. For SAM: after loading, click on the map to segment objects (green = include, red = exclude)
4. Ask the AI chat questions about the analysis results

## Architecture

```
server.py          FastAPI backend — routes, job system, chat, geocoding
satellite.py       Sentinel-2 data access via Planetary Computer
geo_utils.py       Coordinate transforms, raster-to-vector, PNG encoding
indices.py         NDVI, NDWI, EVI computation
sam_segmenter.py   SAM wrapper with georeferencing
mcp_server.py      MCP tools for Claude Code
static/            Frontend (Leaflet map, dark UI)
start.sh           Launch script with provider selection
```

## Environment Variables

| Variable | Description |
|---|---|
| `CHAT_PROVIDER` | `local`, `anthropic`, or `openai` |
| `LOCAL_LLM_URL` | OpenAI-compatible endpoint (default: `http://localhost:11434/v1`) |
| `LOCAL_LLM_MODEL` | Model name for local LLM (default: `llama3`) |
| `SAM_CHECKPOINT` | Path to SAM weights (default: `~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth`) |

## License

MIT
