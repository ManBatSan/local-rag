#!/usr/bin/env sh
set -e

# Rutas de output
RAWDIR=/data/raw
QAP_FILES="${RAWDIR}/question-answer-passages_train.jsonl ${RAWDIR}/question-answer-passages_test.jsonl"
CORPUS_FILE="${RAWDIR}/text-corpus_test.jsonl"

# Comprueba si YA existen ambos QAP y el corpus
if [ -f $CORPUS_FILE ] && [ -f $(echo $QAP_FILES | awk '{print $1}') ] && [ -f $(echo $QAP_FILES | awk '{print $2}') ]; then
  echo "✅ Raw data already present, skipping download."
else
  echo "⏳ Downloading raw data..."
  python download_data.py --config question-answer-passages
  python download_data.py --config text-corpus
  echo "✅ Raw data download complete."
fi
