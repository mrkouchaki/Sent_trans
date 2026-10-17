# Optional MiniLM / BERT / Qwen embeddings

Your Qwen3-8B remains the classifier. Dense embeddings are an optional retrieval
channel. No second model is loaded unless you build and select a dense index.

## OFF: default, no additional model or dependency

```bash
python prepare_data.py --csv whole_ewoc_tickets.csv
python build_index.py --out indexes/lexical
python evaluate_classifier.py --model models/Qwen3-8B --index indexes/lexical
```

This uses BM25 + character TF-IDF + reciprocal rank fusion. To disable all RAG
and use Qwen directly, pass `--no-rag` to prediction/evaluation.

## ON: use your local MiniLM sentence embedding model

Install the optional dependency from your approved package index:

```bash
python -m pip install -r requirements-semantic-optional.txt
python build_index.py \
  --embedding-model models/all-MiniLM-L6-v2 \
  --embedding-kind minilm \
  --out indexes/minilm
python evaluate_classifier.py \
  --model models/Qwen3-8B --index indexes/minilm
```

Replace the model path with your actual local folder. A complete
SentenceTransformer MiniLM checkpoint retains its saved pooling and sequence
limit (commonly 256 for all-MiniLM-L6-v2). No Qwen query instruction is added.
Long ticket texts are shortened at the token level, retaining both ends.

## ON: use your local BERT encoder

```bash
python build_index.py \
  --embedding-model models/bert-base-uncased \
  --embedding-kind bert \
  --out indexes/bert
python evaluate_classifier.py \
  --model models/Qwen3-8B --index indexes/bert
```

If BERT is already a complete SentenceTransformer, its saved pooling is used.
Otherwise a supported BERT-family encoder is wrapped with masked mean pooling.
Raw BERT is not assumed to be trained for semantic sentence similarity; validate
it against MiniLM and lexical retrieval. This loader expects a local Hugging Face
Transformers checkpoint usable by PyTorch, not a standalone TF SavedModel.

`--embedding-kind auto` is the default when you supply a model path. Explicit
`minilm` and `bert` options make your intent visible. Other supported choices
are `sentence-transformer` (complete saved pipeline required) and `qwen`
(actual Qwen embedding checkpoint required).

## Optional lower token cap

Add `--embedding-max-length 128` when building an index. The chosen cap never
increases the model's saved/native limit. Pooling, model path/stamp, query
instruction and effective length are persisted in the index; subsequent queries
reuse those settings. Switching encoders requires rebuilding in a NEW directory.

## How to turn embeddings off later

Select `--index indexes/lexical` again. This disables dense retrieval while
retaining lexical RAG. `--no-rag` disables all retrieval. The optional
sentence-transformers package need not be installed for lexical-only workflows.

The same `--index` works with classify_description.py and serve_api.py.
For mixed RAG fine-tuning, choose the reference index using
`--rag-index indexes/minilm` or `--rag-index indexes/bert`. Retrieval still uses
training rows only; validation/test rows are excluded from the references.

## Are we using ChromaDB or another vector database?

No. This package uses a local file-backed exact search implementation:

| Component | Storage / search |
|---|---|
| BM25 and character TF-IDF | Sparse matrices + vectorizers in lexical.joblib |
| Optional dense embeddings | Normalized NumPy array in embeddings.npy |
| Metadata / encoder settings | index_manifest.json and lexical.joblib |
| Dense nearest neighbors | Exact cosine similarity via NumPy matrix-vector multiplication |

It uses neither ChromaDB nor FAISS and does not run a database service. RAG does
not require a vector database. With dense embeddings off, embeddings.npy is not
created. An index loads saved references and matrices; the model-generated class
ID is still produced by Qwen3-8B.

This index format is version 2. Rebuild previous indexes rather than mixing the
old Qwen-only formatting with MiniLM/BERT.

## Verification

Run `python -m unittest selfcheck test_optional_embeddings -v`.
20 CPU checks passed during preparation. Adapter-format and dense persistence
checks use mocked encoders; they do not claim a tested neural model forward pass.
Actual MiniLM/BERT weights, Qwen generation and H100 training remain to be verified
on your machine with your CSV. No accuracy improvement is claimed yet.

References:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
https://sbert.net/docs/sentence_transformer/usage/custom_models.html
