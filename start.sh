#!/bin/bash
# AgriSense — Start the remote sensing platform
#
# Usage:
#   ./start.sh              # Uses Ollama (qwen2.5:0.5b) by default
#   ./start.sh anthropic    # Uses Anthropic API (needs ANTHROPIC_API_KEY)
#   ./start.sh openai       # Uses OpenAI API (needs OPENAI_API_KEY)

set -e
cd "$(dirname "$0")"

PROVIDER="${1:-local}"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo "$HOME/.local/bin/ollama")}"

case "$PROVIDER" in
    local|ollama)
        # Start Ollama if not running
        if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "Starting Ollama server..."
            "$OLLAMA_BIN" serve &>/tmp/ollama-serve.log &
            sleep 3
        fi

        # Check model exists
        if ! "$OLLAMA_BIN" list 2>/dev/null | grep -q "qwen2.5:3b"; then
            echo "Pulling qwen2.5:3b model..."
            "$OLLAMA_BIN" pull qwen2.5:3b
        fi

        export CHAT_PROVIDER=local
        export LOCAL_LLM_URL=http://localhost:11434/v1
        export LOCAL_LLM_MODEL=qwen2.5:3b
        echo "Chat: Ollama (qwen2.5:3b)"
        ;;
    anthropic)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "Error: ANTHROPIC_API_KEY not set"
            exit 1
        fi
        export CHAT_PROVIDER=anthropic
        echo "Chat: Anthropic API"
        ;;
    openai)
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "Error: OPENAI_API_KEY not set"
            exit 1
        fi
        export CHAT_PROVIDER=openai
        echo "Chat: OpenAI API"
        ;;
    *)
        echo "Unknown provider: $PROVIDER"
        echo "Usage: ./start.sh [local|anthropic|openai]"
        exit 1
        ;;
esac

echo "=================================================="
echo "  AgriSense — Remote Sensing Platform"
echo "  http://localhost:8000"
echo "=================================================="

exec uv run python server.py
