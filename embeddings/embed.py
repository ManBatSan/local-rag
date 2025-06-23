import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


def main(input_dir: str, output_dir: str, model_name: str, text_key: str, device: str):
    os.makedirs(output_dir, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    embeddings = []
    ids = []

    # assume each JSONL has a field 'passage' and unique 'id'
    for fname in os.listdir(input_dir):
        if not (fname.startswith("text-corpus_") and fname.endswith(".jsonl")):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                ids.append(doc["id"])
                embeddings.append(model.encode(doc[text_key], show_progress_bar=False))

    embeddings = np.vstack(embeddings)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "ids.json"), "w") as f:
        json.dump(ids, f)

    print(f"Saved {len(ids)} embeddings to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, help="data/raw/text-corpus_train.jsonl etc"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_key", default="passage")
    parser.add_argument(
        "--device", default="cuda", help='torch device: "cpu" or "cuda"'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model, args.text_key, args.device)
