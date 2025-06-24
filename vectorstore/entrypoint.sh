#!/usr/bin/env sh
echo "⏳ Waiting for embeddings..."
while [ ! -f /data/embeddings/embeddings.npy ]; do
  sleep 5
done
echo "✅ Embeddings ready, launching FAISS service..."
exec uvicorn vector_api:app --host 0.0.0.0 --port 8001
