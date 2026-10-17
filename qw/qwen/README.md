# EWOC Qwen3-8B: configuration-first, local models

Edit the CONFIG section immediately after the imports in `ewoc.py`. Set ACTION,
then run `python ewoc.py`. No command-line settings are required. Optional flags
remain available for compatibility. All previous workflows remain, in six files:

| File | Purpose |
|---|---|
| ewoc.py | Top CONFIG plus preparation, indexing, prediction, evaluation, training, checkpoint selection, merge, API and code checks |
| test_evaluate.py | Independent model-test runner; inherits CONFIG with optional test-only settings after imports |
| ewoc_core.py | Real retrieval/classifier classes, embedding adapters, prompts and leakage checks |
| labels.json | 13 class names and definitions; marked draft definitions require domain review |
| requirements.txt | One dependency list; Transformers/PyTorch by default, vLLM optional |
| README.md | This guide |

## Your company network

Transformers is a Python library and can load local PyTorch model files without
contacting the Hugging Face website. This code forces HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1 and disables Hub/vLLM telemetry before model imports.
Pretrained Transformers/PEFT/SentenceTransformers loaders use local files only;
remote custom code is disabled. Missing files produce a local error. No Hub ID,
login or automatic download is used by the application.

Obtain complete model folders through your company-approved source. MODEL_PATH
must contain config.json, tokenizer files/chat template, and all weight shards
plus their index when sharded. An adapter also needs its original base model.
A single weights file or a repository name is not a complete local model folder.
GGUF is not supported by these Transformers/PEFT loaders.

SentenceTransformer checkpoints need modules.json, pooling/normalization files
and all referenced subfolders. Raw BERT needs its config, tokenizer and PyTorch-
compatible weights; it uses explicit masked mean pooling. A plain BERT checkpoint
is not assumed trained for sentence similarity. General Qwen3-0.6B/Qwen3-8B are
not Qwen3-Embedding checkpoints. Do not use generation weights as embeddings.

Use Linux/Python 3.11 on the H100. Install through your company's approved pip
index, or use a compatible offline wheelhouse:

```bash
python -m pip install -r requirements.txt
# Alternatively, when approved wheels are available locally:
python -m pip install --no-index --find-links=/approved/wheelhouse -r requirements.txt
python -m pip check
python ewoc.py check
```

The first install command uses your pip configuration; point it at your approved
internal mirror beforehand. Packages and models are not bundled. H100 requires
CUDA-enabled PyTorch and a compatible NVIDIA driver. Keep your working environment
while validating a fresh one. Dependency ranges are API targets, not a GPU-tested
lockfile. vLLM is optional and not installed by this requirements file. If selected,
install vllm>=0.12,<0.13 from an approved source with its compatible torch build.

## Main settings

Defaults: unchanged local Qwen3-8B, Transformers/PyTorch BF16 on your allocated GPU,
lexical RAG, no dense embeddings, no adapter, up to 200 test samples per class.
These are starting values, not measured optima for your data.

| Setting in ewoc.py | Options |
|---|---|
| ACTION | prepare/index/predict/evaluate/test/train/select/merge/serve/check |
| MODEL_PATH | Complete local Qwen folder |
| ADAPTER_PATH | None=base; local saved adapter=tuned inference |
| USE_RAG | True/False; independent of fine-tuning |
| EMBEDDING['embedding_model'] | None, or a local BERT/MiniLM/Qwen embedding folder |
| EMBEDDING['embedding_kind'] | auto/bert/minilm/sentence-transformer/qwen |
| INDEX_PATH | Defaults to lexical, or the selected encoder's folder name |
| INFERENCE['backend'] | transformers (default) / vllm |
| FINETUNE['method'] | lora=BF16 LoRA; qlora=NF4 4-bit LoRA with BF16 compute |
| TEST/EVALUATION['max_per_class'] | None=all; positive count=sample up to that many/class |
| TEST/EVALUATION['samples_by_class'] | Optional dictionary, e.g. {1:100, 40:80} |

Paths default relative to the script, even if launched elsewhere. GPU_ID applies
unless your scheduler already set CUDA_VISIBLE_DEVICES. cuda:0 is the first visible
GPU. Stop competing GPU model servers. Transformers processes one prompt at a
time to limit memory; optional vLLM retains batching. Both constrain output to
configured integer class IDs. Retrieval scores are not calibrated probabilities.

## RAG without fine-tuning

1. Set PREPARE['csv'] to your original labeled CSV. Set ACTION='prepare'; run
   `python ewoc.py`. Required columns: DESCRIPTION and integer TYPE_ID. EWOC_ID
   is recommended.
2. Leave EMBEDDING['embedding_model']=None. Set ACTION='index'; run again.
3. Leave ADAPTER_PATH=None and USE_RAG=True. Set ACTION='evaluate'; run again
   for validation metrics. ACTION='predict' accepts one ticket interactively, or
   uses the PREDICT description/text_file setting.

RAG adds retrieved historical labeled tickets to the prompt; model weights do
not change. USE_RAG=False gives direct Qwen classification without needing an
index. Default retrieval is BM25 + character TF-IDF with reciprocal rank fusion.
There is no ChromaDB/FAISS/database server. Lexical matrices and metadata live
in lexical.joblib; optional normalized embeddings.npy vectors use exact NumPy
cosine scoring fused with lexical results.

To add local MiniLM later, edit the existing CONFIG entries:

```python
EMBEDDING = {
    'embedding_model': ROOT / 'models/all-MiniLM-L6-v2',
    'embedding_kind': 'minilm',
    'embedding_max_length': None,
    'device': 'cpu',
    'batch_size': 32,
}
```

For BERT, select your local BERT directory and embedding_kind='bert'. Rebuild
with ACTION='index' into a NEW index directory, then evaluate with that INDEX_PATH.
Changing the encoder requires rebuilding; editing an option cannot change an
existing index. Existing indexes are never silently overwritten. Saved pooling
and native input limits are respected. embedding_max_length can lower the native
limit. Only Qwen embedding models receive the Qwen query instruction.
RAG['reranker'] optionally enables a local BGE reranker; None disables it.

## Fine-tuning

Prepare the data first. Set ACTION='train'. Edit the existing FINETUNE entries:

```python
# 'method': 'lora',              # BF16 LoRA starting configuration for H100
# 'output': ROOT / 'runs/experiment_01',
# 'epochs': 3,
# 'batch_size': 2,
# 'gradient_accumulation': 16,
# 'learning_rate': 1e-4,
```

Run `python ewoc.py`. For QLoRA, set method='qlora', use a new output folder,
and ensure bitsandbytes is installed. Both methods train adapters. Full-weight,
DPO and PPO training are not implemented. FINETUNE exposes LoRA settings, target
modules, token length, optimizer/scheduler, early stopping and resume with notes.
Actual H100 capacity/runtime depend on memory capacity and settings; reduce batch
size or use QLoRA if needed.

Training uses Transformers Trainer + PyTorch + PEFT on one BF16-capable GPU.
Only target ID/EOS tokens contribute to loss; shared prompts and grouped leakage
checks apply. FINETUNE['rag_index']=None gives ordinary SFT; set it to INDEX_PATH
to mix retrieved and zero-shot prompts using rag_probability. Each training
row's own group is excluded; references must come from the source train partition.

All epoch checkpoints retain optimizer/scheduler/RNG state. Early stopping uses
validation completion loss; best loss adapter is saved under output/adapter.
To resume the SAME run/config, set FINETUNE['resume']='latest'. Different settings
need a new output folder. To select by classification macro F1, set SELECTION['run'],
keep ADAPTER_PATH=None, and set ACTION='select'. This evaluates every saved epoch
on the full original validation set, ignoring sampling caps, and exports
adapter_best_f1. Selection refuses to overwrite an existing result.

For tuned inference, set ADAPTER_PATH to the saved adapter and ACTION='evaluate'.
USE_RAG can independently be True or False. Choose settings on validation, then
run the final test. ACTION='merge' optionally exports merged BF16 weights using
MERGE settings. ACTION='serve' starts the localhost API using SERVER settings:
GET /health; POST /classify with description and optional include_evidence.
It uses one worker and CPU retrieval.

## Independent testing and metrics

Set TEST['eval_csv'] to your independent labeled CSV, then ACTION='test'; run
`python ewoc.py`. Or run `python test_evaluate.py` regardless of ACTION. Its top
settings inherit the shared CONFIG and allow test-only overrides. Neither entry
point starts training. N_ROWS_PER_TYPE=None uses all available test rows.

Sampling is seeded without replacement. Shortages are reported; strict_samples=True
stops instead. All configured classes appear; classes with no selected support
affect macro averages. Console and files include accuracy, macro/weighted
precision, recall, F1, per-class support and confusion matrix (actual rows /
predicted columns). Outputs use your chosen prefix:

| Suffix | Contents |
|---|---|
| _summary.csv | Overall metrics |
| _per_class_metrics.csv | Per-class precision/recall/F1/support |
| _confusion_matrix.csv | Named class matrix, including INVALID predictions |
| _sample_counts.csv | Requested/available/used/shortage by class |
| _predictions.csv / _errors.csv | All selected predictions / misclassifications |
| _metrics.json | Full metrics, settings and provenance |
| _evidence.jsonl | Retrieved examples actually used |

## Data, definitions and verification

Preparation quarantines conflicting exact-description labels and deduplicates.
Normalized text, EWOC_ID and supplied group connections are grouped transitively
across train/validation/test. Ten folds require ten independent groups/class;
five folds give approximately 60/20/20. Template grouping is heuristic, not
semantic deduplication. Inspect split_audit.json and label_conflicts.csv. There
is no temporal split. Synthetic tickets are rejected because source lineage is
unverified. DESCRIPTION is input; TYPE_ID is target. Evaluation rejects overlap
with retrieval references or adapter training rows.

Class IDs: 1,2,3,5,6,9,11,13,16,20,26,28,40. make_messages/fit_prompt insert the
catalog and definitions, retrieved examples, ticket data and ID-only rule with
context budgeting. Current definitions are drafts inferred from names and marked
in prompts; review against your internal taxonomy before training. Editing them
after training changes the adapter signature. Old adapters need their original
class/definition configuration.

35 CPU checks pass, including config-only prepare/index/RAG evaluation, sampling,
metric exports, leakage checks, encoder settings, class-ID constraints and mocked
Transformers generation. Actual neural generation/training/merge, real embeddings
and live GPU/API still need verification with your weights, CUDA stack and CSV.
No real-data accuracy gain, GPU memory usage or runtime is claimed.
