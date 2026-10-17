# EWOC: consolidated package

Five source/configuration files; all previous features retained.

| File | Responsibility |
|---|---|
| ewoc.py | One command runner: data preparation, indexing, prediction, evaluation, fine-tuning, checkpoint selection, merge, API and checks |
| ewoc_core.py | Shared data handling, prompts, leakage checks, retrieval, embedding adapters and Qwen inference |
| labels.json | Class names and optional verified business definitions in one configuration |
| requirements.txt | One complete dependency list for every command |
| README.md | This single setup and usage guide |

Training checkpoints, indexes, predictions and logs are generated runtime outputs;
they are not additional source files you need to assemble.

## Setup

Use a fresh Linux/Python 3.11 environment on the H100 server and your approved
package index. This single requirements file installs the dependencies for all
features. Embedding models remain optional: no second model loads unless selected.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python ewoc.py check
```

The dependency ranges are API targets, not a GPU-tested lockfile. Do not force a
different torch build over the one required by vLLM. Keep your existing working
environment available while validating this new environment.

Place your CSV here and your existing weights under `models/Qwen3-8B`, or pass
the actual local model directory with `--model`. No weights or user CSV are bundled.
Stop any other model server using the same allocated GPU before an offline run.

## Commands

Use `python ewoc.py COMMAND --help` for every available option.

| Command | Function |
|---|---|
| prepare | Audit labels, quarantine conflicts, deduplicate and create grouped holdouts |
| index | Build lexical RAG with optional MiniLM/BERT/Qwen embeddings |
| predict | Classify a description or multiline text file |
| evaluate | Accuracy, macro/weighted F1, confusion matrix, errors and retrieval evidence |
| train | H100 BF16 LoRA or NF4 QLoRA, optional mixed RAG prompts, epoch saves and resume |
| select | Choose an epoch by generated validation macro F1 |
| merge | Optionally merge the adapter into full BF16 weights |
| serve | Local FastAPI `/health` and `/classify` endpoints |
| check | All 23 CPU integrity/adapter/consolidation checks |

## Data and default retrieval

Required CSV columns: DESCRIPTION and integer TYPE_ID. EWOC_ID is recommended.
Synthetic rows are rejected because their original source lineage is unknown.

```bash
python ewoc.py prepare --csv whole_ewoc_tickets.csv
python ewoc.py index --out indexes/lexical
CUDA_VISIBLE_DEVICES=0 python ewoc.py evaluate \
  --model models/Qwen3-8B --index indexes/lexical \
  --eval-csv data/validation.csv --out results/base_lexical_val
```

The default split is approximately 80/10/10, requiring ten independent groups per
class. `--folds 5` gives approximately 60/20/20 when rare classes have fewer groups.
Inspect `data/split_audit.json` and `data/label_conflicts.csv`. Exact normalized
description and ticket-ID connections are grouped transitively; optional
`--group-templates` adds heuristic identifier grouping. General semantic duplicate
detection and time-based splits are not implemented.

The default taxonomy contains your 13 evaluated classes. Edit `labels.json` to
change names, add classes or provide verified internal definitions, then rebuild
splits/indexes and retrain as appropriate. Each entry has `name` and `definition`.
Empty definitions are omitted. Inference validates the adapter's taxonomy signature.
Legacy separate `--labels` and `--definitions` inputs remain supported.

For direct Qwen classification without any RAG, add `--no-rag`.
For one ticket:

```bash
CUDA_VISIBLE_DEVICES=0 python ewoc.py predict \
  --model models/Qwen3-8B --index indexes/lexical \
  --text-file ticket.txt --out results/one_ticket.json
```

`--description "..."` or an interactive prompt is also supported. Use a text file
for multiline tickets or shell-special characters. Qwen produces a constrained
allowed ID; Python supplies the type name. No confidence probability is invented.

## Optional MiniLM / BERT embeddings

No embedding model is needed for the default BM25 + character TF-IDF retrieval.
To add semantic retrieval, build a separate index and select it:

```bash
python ewoc.py index --embedding-model models/all-MiniLM-L6-v2 \
  --embedding-kind minilm --out indexes/minilm
python ewoc.py evaluate --model models/Qwen3-8B --index indexes/minilm
```

For BERT, use `--embedding-model models/bert-base-uncased --embedding-kind bert`
and a new output such as `indexes/bert`. Saved SentenceTransformer pooling is
retained; raw supported BERT-family encoders receive masked mean pooling. Raw BERT
is not assumed to have been trained for sentence similarity. This loader expects
a Hugging Face checkpoint usable by PyTorch, not a standalone TF SavedModel.

`--embedding-kind auto` detects saved pipelines and supported raw encoders.
`qwen` requires the actual Qwen embedding checkpoint, distinct from Qwen3-8B.
MiniLM/BERT receive plain text; Qwen embeddings receive their query instruction.
Saved/native token limits are respected; optional `--embedding-max-length` only
lowers that limit. Long inputs retain both ends. Encoder settings are persisted.

To turn embeddings off again, select `--index indexes/lexical`. `--no-rag` turns
all retrieval off. Model changes require rebuilding a new index. Optional
`--reranker models/bge-reranker-v2-m3` uses separate local reranker weights.
Retrieval runs on CPU by default; batch evaluation can precompute it using
`--retrieval-device cuda:0` before releasing retrieval models and loading vLLM.

**Storage:** no ChromaDB, FAISS or database service. Lexical sparse matrices and
metadata use `lexical.joblib`; optional normalized dense vectors use
`embeddings.npy`, searched by exact cosine similarity with NumPy. Load only trusted
local indexes. Index format remains version 2 from the optional-embedding package.

## H100 fine-tuning and resume

```bash
CUDA_VISIBLE_DEVICES=0 python ewoc.py train \
  --model models/Qwen3-8B --output runs/ewoc_lora \
  --quantization none --epochs 3 \
  --batch-size 2 --gradient-accumulation 16
```

BF16 LoRA updates adapters, keeping base weights frozen. `--quantization nf4`
enables 4-bit QLoRA. Only answer-ID/EOS tokens contribute to loss; prompt and
padding labels are masked. SDPA and gradient checkpointing are enabled.

Every epoch's resumable checkpoint is retained under `runs/ewoc_lora/checkpoints`.
`runs/ewoc_lora/adapter` is the best epoch by validation loss. Training history and
metadata are saved. Early stopping patience is two evaluations by default.
Resume using the original arguments plus `--resume latest`; changed run settings,
data, taxonomy or base-file stamps are rejected. These are trusted local checkpoints.

For mixed direct/RAG training, use a new output directory and add
`--rag-index indexes/minilm --rag-probability 0.5 --rag-examples 4` (or lexical).
The query's own group is excluded; demonstrations come only from training rows.
Use a lower batch size or NF4 in a new run if GPU memory is insufficient. Capacity
and runtime have not been measured on your server.

## Select, test, merge and serve

```bash
CUDA_VISIBLE_DEVICES=0 python ewoc.py select \
  --model models/Qwen3-8B --run runs/ewoc_lora \
  --validation-csv data/validation.csv --no-rag \
  --out results/select_direct
```

Selection generates predictions from each epoch in a separate subprocess and
exports `runs/ewoc_lora/adapter_best_f1`. It requires the original full validation
partition and refuses overwrite. Omit `--no-rag` and supply the chosen index and
example/reranker settings when selecting a RAG configuration. Preserve an earlier
selection before deliberately selecting again with different inference settings.

Fix the winning configuration on validation; evaluate test.csv once. Example if
direct classification wins:

```bash
CUDA_VISIBLE_DEVICES=0 python ewoc.py evaluate \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 --no-rag \
  --eval-csv data/test.csv --out results/final_test
CUDA_VISIBLE_DEVICES=0 python ewoc.py merge \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 \
  --out models/Qwen3-8B-EWOC
CUDA_VISIBLE_DEVICES=0 python ewoc.py serve \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 \
  --no-rag --port 8000
```

Merging is optional and requires more disk space. Use a merged model via `--model`
without `--adapter`. API defaults to localhost, one worker and CPU retrieval.
POST `/classify` with `{"description":"Verify E911 test calls","include_evidence":false}`.
GET `/health` reports readiness. External authentication belongs to your deployment.

## Verification

23 CPU checks pass, including the original leakage/retrieval/completion-masking
checks, optional encoder settings, unified configuration, checkpoint subprocess
dispatch and mocked API route registration. CLI help works for all nine commands;
a synthetic fixture exercised prepare/index with disjoint holdouts.
Neural encoder inference, CUDA generation, training, real merge and live HTTP API
still require verification with your model weights/CSV/CUDA stack. Dependencies
were not installed and tested as a combined GPU environment here. No real-data
accuracy improvement, GPU memory figure or runtime is claimed.
