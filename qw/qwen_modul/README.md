# EWOC Qwen3-8B: configuration-first, local models

Edit the CONFIG section immediately after the imports in `ewoc.py`. Set ACTION,
then run `python ewoc.py`. Training automatically prepares its data when needed;
you do not run prepare first. No command-line settings are required. Optional flags
remain available for compatibility. All previous workflows remain. Modules are grouped by responsibility:

| File | Purpose |
|---|---|
| ewoc.py | Top CONFIG and workflow entry points; prediction, checkpoint selection, merge and API |
| data_preparation.py | DataPreparer: validate, clean, group/split, cache and verify prepared data |
| training.py | TrainingPipeline: automatic preparation/indexing, LoRA/QLoRA training and checkpoint lifecycle |
| evaluation.py | Holdout sampling, inference, accuracy/precision/recall/F1 and confusion matrix exports |
| test_workflows.py | CPU regression tests for preparation, training dependencies, caching, leakage and evaluation |
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
| TRAIN_CSV / VALIDATION_CSV | None=automatic; set both for your own grouped CSVs |
| TEST_CSV | None=automatic held-out test split; or an independent labeled CSV |
| TEST/EVALUATION['max_per_class'] | None=all; positive count=sample up to that many/class |
| TEST/EVALUATION['samples_by_class'] | Optional dictionary, e.g. {1:100, 40:80} |

Paths default relative to the script, even if launched elsewhere. GPU_ID applies
unless your scheduler already set CUDA_VISIBLE_DEVICES. cuda:0 is the first visible
GPU. Stop competing GPU model servers. Transformers processes one prompt at a
time to limit memory; optional vLLM retains batching. Both constrain output to
configured integer class IDs. Retrieval scores are not calibrated probabilities.

## RAG without fine-tuning

Set PREPARE['csv'] to your original CSV and MODEL_PATH to your local Qwen folder.
Keep TRAIN_CSV=VALIDATION_CSV=None. Set ACTION='evaluate', ADAPTER_PATH=None and
USE_RAG=True, then run `python ewoc.py`. It prepares/reuses the grouped splits,
builds/reuses the retrieval index from training tickets, and evaluates validation.
No separate prepare or index action is required. For one ticket, use ACTION='predict'.
Required source columns: DESCRIPTION and integer TYPE_ID; EWOC_ID is recommended.

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

For BERT, select your local BERT directory and embedding_kind='bert'. The next
RAG action automatically builds/reuses a version matching the selected encoder
and training data. You do not need to run index first. Existing versions are
preserved; cache content hashes detect partial or externally edited files. Saved pooling
and native input limits are respected. embedding_max_length can lower the native
limit. Only Qwen embedding models receive the Qwen query instruction.
RAG['reranker'] optionally enables a local BGE reranker; None disables it.

## Fine-tuning

Set PREPARE['csv'] and MODEL_PATH, leave TRAIN_CSV and VALIDATION_CSV as None,
and set ACTION='train'. Edit the existing FINETUNE entries:

```python
# 'method': 'lora',              # BF16 LoRA starting configuration for H100
# 'output': ROOT / 'runs/experiment_01',
# 'epochs': 3,
# 'batch_size': 2,
# 'gradient_accumulation': 16,
# 'learning_rate': 1e-4,
```

Run `python ewoc.py`. Training calls DataPreparer.ensure() itself, then continues
into fine-tuning. If a verified split already matches the source data/settings, it
is reused. There is no manual preparation step. For QLoRA, set method='qlora', use a new output folder,
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
The training pipeline creates/reuses the matching index automatically when rag_index
is enabled; no separate index action is needed.

All epoch checkpoints retain optimizer/scheduler/RNG state. Early stopping uses
validation completion loss; best loss adapter is saved under output/adapter.
To resume the SAME run/config, set FINETUNE['resume']='latest'. A changed source,
split configuration or training-file content stops resume before GPU training.
Different settings need a new output folder. To select by classification macro F1, set SELECTION['run'],
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

Leave TEST['eval_csv']=None for the automatically prepared test split, or set it
to your independent labeled CSV. Set ACTION='test'; run
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

## Preparation, caching and custom data

Preparation is an internal dependency, not a required manual step. DataPreparer
cleans/deduplicates, quarantines conflicting exact-description labels, groups
connected descriptions/EWOC_IDs/supplied GROUP_IDs, and creates disjoint real
train/validation/test partitions. All configured classes must exist in each split.
Ten folds require ten independent groups/class (approximately 80/10/10); five
folds give approximately 60/20/20. PREPARE['seed'] controls splitting independently
of the training/sampling RANDOM_SEED. Changing the training method, learning rate,
batch size or training seed alone does not re-split your data.

Cache identity uses the raw CSV's SHA-256, labels, split options and preparation/
sklearn version. Outputs are staged, verified, then published under
PREPARE['out']/prepared/<signature>/. Source/settings changes select a new version;
previous splits are preserved. Reuse verifies file hashes, class coverage, grouping
and pairwise leakage. Corrupt/partial caches are quarantined and rebuilt from
stable source data. The manifest, split audit and conflict CSV are beside the splits.
The resolved paths/signature and training/validation content hashes are recorded
in each training run. Resume checks the original signature before generating new data.

RAG indexes are similarly versioned under INDEX_PATH/versions/<signature>/ using
training content, taxonomy, encoder configuration and local encoder file stamp.
Only training rows are indexed. Configured index paths are roots; the resolved
version is recorded in evaluation/training metadata. For deployment without the
raw CSV, set INDEX_PATH to that resolved index folder. Cache folders are runtime
outputs, not additional source modules.

For externally prepared data, set BOTH TRAIN_CSV and VALIDATION_CSV to your grouped,
real CSVs. They are validated directly without reading the raw source or splitting
again. Set TEST['eval_csv'] separately for your test file. A custom test CSV with
USE_RAG=False requires no preparation at all. Evaluation rejects overlap with
retrieval references or adapter training rows. Only DESCRIPTION is model input;
TYPE_ID is target. Synthetic tickets are rejected because source lineage is
unverified. Template grouping is heuristic, not semantic deduplication; there is
no temporal split. Standalone prepare/index actions remain available for inspecting
or prewarming dependencies, but are optional.

Class IDs: 1,2,3,5,6,9,11,13,16,20,26,28,40. make_messages/fit_prompt insert the
catalog and definitions, retrieved examples, ticket data and ID-only rule with
context budgeting. Current definitions are drafts inferred from names and marked
in prompts; review against your internal taxonomy before training. Editing them
after training changes the adapter signature. Old adapters need their original
class/definition configuration. Use a new run for this revised pipeline rather
than assuming an older script's checkpoint configuration can be resumed unchanged.

45 CPU checks pass, including first training from raw data with the neural fit
mocked, subsequent LoRA/QLoRA configuration reuse, automatic RAG indexing, changed
source/settings, corrupt caches, failed staging, resume data-change rejection,
custom grouped data, sampling/metric exports and mocked Transformers generation.
A mocked Trainer integration also exercises both methods, epoch metadata/adapter
exports and resume through the actual training function.
Run `python ewoc.py check` or `python -m unittest test_workflows -v`.
Actual neural training/generation/merge, embeddings and live GPU/API still need
verification with your weights, CUDA stack and CSV. No real-data accuracy gain,
GPU memory usage or runtime is claimed.
