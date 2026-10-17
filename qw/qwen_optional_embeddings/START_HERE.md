# MiniLM/BERT option added

Read OPTIONAL_EMBEDDINGS.md first for optional MiniLM/BERT settings and storage details. Dense retrieval is OFF by default. Index format is now version 2: rebuild old indexes in a new directory.

# Use your existing Qwen3-8B: no embedding model required

This edition defaults to lexical RAG: BM25 + character TF-IDF + reciprocal rank
fusion retrieve labeled historical tickets. Your existing Qwen3-8B reads those
examples and predicts the ticket type. No 0.6B model or separate reranker is
needed. Fine-tuning also uses your existing Qwen3-8B.

## The model-name distinction

| Exact checkpoint | Purpose |
|---|---|
| Qwen3-0.6B | General causal language model; the model in your screenshot |
| Qwen3-8B | General causal language model; use your existing local copy |
| Qwen3-Embedding-0.6B | Separate checkpoint trained for text embeddings; optional |
| Qwen3-Embedding-8B | Separate embedding checkpoint; not the ordinary Qwen3-8B |

The word `Embedding` is part of the checkpoint name, not a claim based on its
parameter count. Both general and embedding models contain token-embedding
layers, and may even use the same architecture class; that does not make their
trained weights or purposes interchangeable. A `Non-Embedding Parameters`
count refers to parameters outside the token-embedding layers.

It is possible to extract vectors from ordinary Qwen3-8B hidden states, but
that alone does not establish a good semantic-retrieval encoder. This package
does not pretend those vectors are equivalent to a trained embedding model.
It uses Qwen3-8B directly for classification and optionally fine-tunes it.

## Setup

Use Linux/Python 3.11 on your GPU server and your approved package index.
Keep all supplied Python and JSON files together. Copy your CSV into this folder
and place your existing Qwen3-8B under `models/Qwen3-8B`, or pass its actual path
using `--model`. The archive contains code, not weights or your CSV.

```bash
python3.11 -m venv .venv-infer
source .venv-infer/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip check
python selfcheck.py
```

## 1. Prepare the data

```bash
python prepare_data.py --csv whole_ewoc_tickets.csv --out data
```

Required columns: DESCRIPTION and integer TYPE_ID. EWOC_ID is recommended.
Default split is approximately 80/10/10. If a class has fewer than ten independent
groups, use `--folds 5` for approximately 60/20/20. Inspect `data/split_audit.json`
and `data/label_conflicts.csv`. The default taxonomy has the 13 evaluated types.

## 2. Build retrieval without another model

```bash
python build_index.py --train-csv data/train.csv
```

Default output: `indexes/lexical`. No GPU, embedding weights, or
sentence-transformers installation is used by this command. Do not pass
`--embedding-model` for this workflow.

## 3. Evaluate using Qwen3-8B

Stop another model server using the same allocated GPU before these runs.
Choose your allocated GPU instead of 0 if needed.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B \
  --eval-csv data/validation.csv \
  --out results/qwen8b_lexical_val
```

To compare direct Qwen classification without retrieval:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B --no-rag \
  --eval-csv data/validation.csv \
  --out results/qwen8b_direct_val
```

Reports include accuracy, macro/weighted F1, confusion matrix, error rows and
retrieval evidence. RRF scores are not confidence probabilities.

## 4. Classify one description

```bash
CUDA_VISIBLE_DEVICES=0 python classify_description.py \
  --model models/Qwen3-8B \
  --description "Please validate completed E911 test calls." \
  --out results/one_ticket.json
```

For multiline tickets or text containing shell-special characters, use
`--text-file ticket.txt`. Add `--no-rag` for direct classification.

## 5. Fine-tune the existing Qwen3-8B on H100

Use a separate training environment to avoid vLLM/torch dependency conflicts:

```bash
deactivate
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-training.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 python finetune_qwen.py \
  --model models/Qwen3-8B \
  --output runs/ewoc_lora \
  --quantization none --epochs 3 \
  --batch-size 2 --gradient-accumulation 16
```

This is BF16 LoRA. Use `--quantization nf4` in a new run directory for QLoRA.
Both train adapters on the 8B model. Neither requires an embedding model.
Every epoch checkpoint includes optimizer/scheduler state and is retained.
`runs/ewoc_lora/adapter` contains the best checkpoint by validation loss.

Resume with the original arguments and `--resume latest`.
To fine-tune with a mix of direct and retrieved training prompts, add
`--rag-index indexes/lexical --rag-examples 4` in a separate run directory.

## 6. Select the best epoch by validation macro F1

```bash
deactivate
source .venv-infer/bin/activate
CUDA_VISIBLE_DEVICES=0 python select_checkpoint.py \
  --model models/Qwen3-8B --run runs/ewoc_lora \
  --validation-csv data/validation.csv --no-rag \
  --out results/select_lora_direct
```

The selected adapter is `runs/ewoc_lora/adapter_best_f1`. Omit `--no-rag` to
select using lexical RAG. Fix the winning inference configuration on validation,
then run `evaluate_classifier.py` once on `data/test.csv` with that configuration.
Use `--adapter runs/ewoc_lora/adapter_best_f1` to load the selected adapter.

## Optional semantic retrieval later

Only if you obtain the actual **Qwen3-Embedding-0.6B** checkpoint:

```bash
python -m pip install -r requirements-semantic-optional.txt
CUDA_VISIBLE_DEVICES=0 python build_index.py \
  --train-csv data/train.csv \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda:0 --out indexes/semantic
```

Then explicitly use `--index indexes/semantic` at inference/evaluation. Download
the complete model repository including modules.json and its pooling configuration.
The code now refuses to silently create a pooling encoder from an ordinary LLM.

The same optional dependency file enables the BGE cross-encoder reranker if you
later obtain those separate model weights. Neither option is required here.

`ADVANCED_REFERENCE.md` retains the full earlier reference for optional dense
retrieval, checkpoint selection, merge and API commands. Its explicit
`indexes/hybrid` examples refer to optional semantic experiments. This edition's
default index is `indexes/lexical`; install the optional dependencies first when
following any dense/reranker examples.

## Verification

14 CPU checks passed on this edition, including ordinary-LM rejection for the
optional embedding loader. Neural inference, fine-tuning, merge and the live API
still require verification with your model files, actual CSV and CUDA stack.
No real-data accuracy improvement or H100 memory/runtime result is claimed.

Model references:
https://huggingface.co/Qwen/Qwen3-0.6B
https://huggingface.co/Qwen/Qwen3-8B
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
