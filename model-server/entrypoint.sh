#!/usr/bin/env sh

echo "⏳ Loading model..."

export USE_LLAMACPP=1

export MODEL_NAME="DeepSeek-R1-Distill-Qwen-14B-Q6_K_L.gguf"

exec uvicorn model_api:app --host 0.0.0.0 --port 8002
