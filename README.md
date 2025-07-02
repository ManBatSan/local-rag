## Project: Local RAG Pipeline

This repository provides a modular, containerized retrieval-augmented generation (RAG) pipeline, designed for local development and easy migration to cloud environments (VMs, Kubernetes). While we use the BioASQ13 dataset as an example, the pipeline supports any JSON/JSONL-based text collection.

### 📖 Dataset

**BioASQ13 Task B** benchmark (via Hugging Face):

- Contains development and test questions, titles, abstracts, and gold-standard answers (exact and ideal).
- Originates from the BioASQ challenge (phases A and B), focusing on biomedical question answering. See [BioASQ Task 13B Guidelines](http://bioasq.org).
- Our example dataset: `enelpol/rag-mini-bioasq` on Hugging Face, which provides two configs:

  - **question-answer-passages**: includes questions, answers, and associated passage IDs (test/development splits).
  - **text-corpus**: contains the full text passages referenced by the IDs.

To build and evaluate the RAG pipeline, you should download both configs to:

```bash
python3 scripts/download_data.py --dataset enelpol/rag-mini-bioasq --config question-answer-passages
python3 scripts/download_data.py --dataset enelpol/rag-mini-bioasq --config text-corpus
```

### 🗂️ Repository Structure

```
my-local-rag/
├── data/
│   ├── raw/                # Downloaded dataset files (config-specific JSONL)
│   └── processed/          # Split & cleaned passages
├── scripts/
│   └── download_data.py    # Example data download script (supports --config)
├── embeddings/             # Embedding generator service
├── vectorstore/            # FAISS-based similarity search service
├── model-server/           # LLM inference service
├── api/                    # FastAPI RAG orchestrator
└── docker-compose.yml      # Compose configuration
```

## Development setup

We provide scripts to bootstrap code quality tools:

- **Linux/macOS/WSL2**
  ```bash
  ./dev-setup/install_precommit.sh
  ```
- **Windows**
  ```bat
  .\dev-setup\install_precommit.bat
  ```

### ⚙️ How to Run

1. **Clone repository** and `cd local-rag/`.
2. **Download both configs**:

   ```bash
   python scripts/download_data.py --config question-answer-passages
   python scripts/download_data.py --config text-corpus
   ```

3. **Download main model**: 

    You can download any model for llama_cpp or Transformers you wish to use. Good results were obtained for 16GB of VRAM using DeepSeek-R1-Distill-Qwen-14B-Q6_K_L

    For ease of use, you can download this model from huggingface using the following link: [DeepSeek-R1-Distill-Qwen-14B-Q6_K_L](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-14B-Q6_K_L.gguf)
4. **Process and clean** the raw data in `data/processed/` (see ingestion scripts).
5. **Docker Compose**:

   ```bash
   docker-compose up --build
   ```

6. **Access API** Example call:

   ```bash
    curl --location 'http://localhost:8000/chat' \
    --header 'Content-Type: application/json' \
    --data '{"question": "What is the role of STAG1/STAG2 proteins in differentiation?"}'
   ```

---

_This README uses the BioASQ13 dataset as an example. Replace dataset and config names to adapt the pipeline to any JSON/JSONL-based collection._
