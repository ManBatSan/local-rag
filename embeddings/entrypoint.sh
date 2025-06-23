#!/usr/bin/env sh
set -e

RAWDIR=/data/raw
EMB_OUTPUT=/app/output/embeddings.npy

# Esperar a que exista al menos el corpus
echo "⏳ Waiting for raw corpus..."
while [ -z "$(ls ${RAWDIR}/text-corpus_*.jsonl 2>/dev/null)" ]; do
  sleep 5
done

# Si ya existe embeddings.npy, saltar generación
if [ -f $EMB_OUTPUT ]; then
  echo "✅ Embeddings already generated, skipping."
else
  echo "⏳ Generating embeddings (using GPU if available)…"
  python embed.py \
    --input_dir $RAWDIR \
    --output_dir /app/output \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --device cuda
  echo "✅ Embeddings generation complete."
fi
