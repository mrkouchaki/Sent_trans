# EWOC Qwen3-8B: hybrid RAG and H100 fine-tuning

This is a complete replacement package reconstructed from the supplied screenshots.
It does not preserve the old modules' Python calling signatures. Use the included
scripts together in a new folder; your original scripts need not be overwritten.

## What your old RAG does

Your old `ExampleRetriever` tokenizes ticket descriptions, weights words with a
TF-IDF variant, ranks historical tickets by normalized lexical similarity, and
adds labeled matches to the Qwen prompt. This is **retrieved few-shot
classification** (also called example-based RAG). It does not train Qwen and does
not use an embedding model, vector database, or neural reranker.

The new pipeline is:

1. BM25 over word unigrams/bigrams, retaining negation and action words.
2. Character TF-IDF, helpful for abbreviations and spelling variations.
3. Optional dense semantic retrieval using a local `Qwen3-Embedding-0.6B` model.
4. Reciprocal rank fusion (RRF) to combine rankings without assuming scores share a scale.
5. Optional cross-encoder reranking using local `BAAI/bge-reranker-v2-m3`.
6. Up to 12 diverse labeled examples (at most 4 per type), shortened to fit context.
7. Qwen3-8B predicts one allowed integer ID using constrained decoding.
8. Python maps that ID to the exact type name. Evidence is saved for diagnosis.

When `--embedding-model` is omitted at index creation, only the first two retrieval
channels operate; this is a deliberate lexical ablation, not full semantic RAG.
RRF and reranker scores are **not calibrated probabilities**. There is no invented
confidence score and no automatic claim that retrieved labels are correct.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Deduplicate, quarantine contradictory labels, group tickets, create fixed splits |
| `build_index.py` | Build lexical/dense references from training rows only |
| `classify_description.py` | One ticket or multiline text-file classification |
| `evaluate_classifier.py` | Accuracy, macro/weighted F1, confusion matrix, errors and evidence |
| `finetune_qwen.py` | H100 BF16 LoRA or 4-bit QLoRA, completion-only loss, epoch checkpoints |
| `select_checkpoint.py` | Choose an epoch by generated validation macro F1 |
| `merge_adapter.py` | Optional adapter merge into a BF16 model |
| `serve_api.py` | Optional local `/health` and `/classify` API |
| `ewoc_common.py`, `ewoc_retrieval.py`, `ewoc_classifier.py`, `ewoc_sft_data.py` | Shared implementation; keep beside the scripts |
| `labels.json`, `label_definitions.json` | One taxonomy shared by every stage; optional verified definitions |
| `requirements-*.txt` | Separate core, inference, and training dependencies |
| `selfcheck.py` | Meaningful CPU integrity checks; no GPU/models needed |
| `VALIDATION.txt` | Checks actually run while preparing this package |

## 1. Layout and environments

Run all commands below from the package folder. Expected layout:

```text
ewoc_qwen_advanced/
  whole_ewoc_tickets.csv
  models/
    Qwen3-8B/
    Qwen3-Embedding-0.6B/       # optional but needed for full hybrid retrieval
    bge-reranker-v2-m3/        # optional
  [the supplied Python and configuration files]
```

Copy your CSV here and point `models/` at your existing local model folders. A
Linux symlink to your existing model directory also works. Every model flag takes
a different local path if you prefer; no model weights are included in this ZIP.
The 8B generator cannot substitute for the embedding model. Obtain optional model
folders through your approved download process. No runtime model download is attempted.

Use **Linux, Python 3.11, and a CUDA-enabled environment on your H100 server**.
Use your company's configured approved Python package index (for example JFrog).
The dependency files declare API ranges; they are not a GPU-tested lockfile.
Do not force an incompatible torch version over the one required by vLLM.

Create separate environments to avoid training/vLLM CUDA dependency conflicts:

```bash
python3.11 -m venv .venv-infer
source .venv-infer/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip check
python selfcheck.py
```

For fine-tuning, create another environment:

```bash
deactivate
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-training.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If your approved index lacks these versions, keep the existing working environment
and resolve versions with your platform team. The supplied inference code targets
vLLM 0.12's `StructuredOutputsParams`, not older `guided_choice` APIs.

## 2. Prepare the real data once

Required CSV columns: `DESCRIPTION` and integer `TYPE_ID`. `EWOC_ID` is strongly
recommended. Column names are case insensitive. Optional `IS_SYNTHETIC` values
`true`, `yes`, or `1` are detected; this workflow rejects synthetic tickets because
their source lineage is not known. Split real data first and handle any augmentation
as a separate, source-traced training-only task.

`TYPE_WO`, `WorkOrder_Info`, and type-name fields are not used as features, avoiding
accidental target leakage. This package classifies from `DESCRIPTION` alone.

```bash
source .venv-infer/bin/activate
python prepare_data.py --csv whole_ewoc_tickets.csv --out data
```

Default split is approximately **80% train / 10% validation / 10% test** via ten
stratified group folds. It requires at least ten independent groups per class.
If a rare class lacks ten groups, use `--folds 5` (approximately 60/20/20), or add
more real data. The script fails if a partition lacks a configured class.

Outputs include `data/train.csv`, `validation.csv`, `test.csv`, `label_conflicts.csv`
and `split_audit.json`. Inspect the conflict file before training. Contradictory
labels for the same normalized description are excluded from all splits.

Identical normalized text and shared `EWOC_ID` values are grouped transitively.
Duplicates are removed only after preserving all ticket-ID connections. Optional
`--group-templates` also groups some descriptions differing only in long identifiers.
This heuristic may over-group; inspect the audit and rare-class counts. General
semantic near-duplicate detection and time-based holdouts are not implemented.

The default taxonomy matches your **13-class evaluation**. `31: DAS Design Review`
was in your old classifier but not that evaluation. To include it, add
`"31": "DAS Design Review"` to `labels.json`, add its optional definition, and rebuild
the splits/index and retrain. Every stage must use the same taxonomy.

`label_definitions.json` intentionally starts empty. Fill it with verified internal
definitions when available. No unverified business boundaries were fabricated.
If definitions change after fine-tuning, retrain or use that adapter's saved
definition file. Inference rejects a mismatched taxonomy/definition signature.

## 3. Build the advanced retrieval index

Use a free GPU for indexing. `CUDA_VISIBLE_DEVICES=0` is an example; choose your
allocated GPU. Stop your old vLLM service on that GPU before these offline runs.

```bash
CUDA_VISIBLE_DEVICES=0 python build_index.py \
  --train-csv data/train.csv \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --out indexes/hybrid
```

This command ends after indexing and releases the GPU. The embedding model path
and file stamp are recorded; rebuild the index if its files change. Index creation
refuses to overwrite an existing index. Use a fresh `--out` for a new experiment.
Indexes contain local joblib objects; load only indexes you built and trust.

Without an embedding model, you can still run the lexical baseline:

```bash
python build_index.py --train-csv data/train.csv --out indexes/lexical
```

## 4. Evaluate base Qwen and inspect failures

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B --no-rag \
  --eval-csv data/validation.csv --out results/base_no_rag_val

CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B --index indexes/hybrid \
  --eval-csv data/validation.csv --out results/base_hybrid_val
```

To test reranking, add `--reranker models/bge-reranker-v2-m3` to the second
command and use a different `--out`. It runs on CPU by default so it does not
compete with vLLM memory. For batch evaluation you may set
`--retrieval-device cuda:0`; retrieval is completed and its neural models released
before vLLM starts. This option is intentionally disallowed by the persistent API.

Each run writes `_metrics.json`, `_predictions.csv`, `_errors.csv`, and
`_evidence.jsonl`. Check the confusion pairs, per-class precision/recall, and
`retrieval_true_class_coverage`: whether the correct class was present among the
examples actually retained in the prompt. Use it as a diagnostic, not a label
prediction metric. Labels are never used to select a query's retrieved examples.

Try `--examples 8 --max-per-type 2` versus `--examples 12 --max-per-type 4` on the
same validation rows. `--max-per-class 200` enables a seeded validation subset for
faster experiments; use the entire holdout for the final report. Results from the
new split should not be directly compared as percentage gains over the old 70.58%.

## 5. Classify a ticket now

```bash
CUDA_VISIBLE_DEVICES=0 python classify_description.py \
  --model models/Qwen3-8B --index indexes/hybrid \
  --description "Please validate the completed E911 test calls for these sectors." \
  --out results/one_ticket.json
```

For multiline descriptions use `--text-file ticket.txt` instead of `--description`.
Do not put shell-special characters from a ticket into a shell command; the
text-file input avoids shell expansion. Without either flag the script prompts once.
Add `--no-rag` to run without an index, or `--adapter PATH` for a trained adapter.

## 6. Fine-tune on the H100

Start with **BF16 LoRA** on the H100. This updates rank-32 adapters on linear layers;
the base model weights stay frozen. It is not full-parameter fine-tuning.
The same script supports **4-bit NF4 QLoRA** using `--quantization nf4`.

```bash
deactivate
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES=0 python finetune_qwen.py \
  --model models/Qwen3-8B \
  --train-csv data/train.csv --validation-csv data/validation.csv \
  --output runs/ewoc_lora \
  --quantization none --epochs 3 \
  --batch-size 2 --gradient-accumulation 16 \
  --max-length 2048 --learning-rate 0.0001
```

Effective batch size is 32 on one GPU. SDPA, BF16, gradient checkpointing and
dynamic padding are enabled. Only the assistant's class ID and EOS tokens are
supervised; prompt tokens and padding have label `-100`. No extra tokenizer tokens
are added. The script does not require TRL or FlashAttention installation.

The run saves:

- `runs/ewoc_lora/checkpoints/checkpoint-N/` after **every epoch**, with adapter,
  optimizer/scheduler/RNG state and metadata. All epoch checkpoints are retained.
- `runs/ewoc_lora/adapter/`: best adapter by **validation completion loss**.
- `runs/ewoc_lora/training_history.json`: training/evaluation history.

Validation loss is not classification accuracy. Early stopping patience is two
evaluations by default. A checkpoint completed before interruption can be resumed:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_qwen.py \
  --model models/Qwen3-8B --output runs/ewoc_lora --resume latest
```

Keep all original non-default arguments when resuming. The script rejects changed
data, taxonomy, model files or run configuration. `--resume` restores an epoch
checkpoint, not the exported `adapter/` directory. Checkpoints are local trusted
files and may contain optimizer serialization; do not resume untrusted runs.

If GPU memory is insufficient, reduce `--batch-size` to 1, or choose
`--quantization nf4` with a new output directory such as `runs/ewoc_qlora`.
Increasing `--max-length` can preserve more long-ticket content but increases
memory use. H100 model, memory sharing and sequence lengths affect capacity;
there is no measured speed or memory guarantee for your server.

For a model explicitly trained to use retrieved examples, start a separate mixed
run (half zero-shot, half training-only retrieved prompts):

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_qwen.py \
  --model models/Qwen3-8B --output runs/ewoc_lora_rag \
  --rag-index indexes/hybrid --rag-probability 0.5 --rag-examples 4
```

The query's own group is excluded from training demonstrations. Validation
demonstrations come only from training rows. No validation/test ticket enters the
reference index. For this run, start inference comparisons with `--examples 4`;
context budgeting may further reduce the number of examples.

## 7. Choose the checkpoint by the metric you care about

Switch back to inference environment. For the zero-shot-trained run:

```bash
deactivate
source .venv-infer/bin/activate
CUDA_VISIBLE_DEVICES=0 python select_checkpoint.py \
  --model models/Qwen3-8B --run runs/ewoc_lora \
  --validation-csv data/validation.csv --no-rag \
  --out results/select_lora_no_rag
```

This generates predictions from each epoch checkpoint, selects by **validation
macro F1**, and exports `runs/ewoc_lora/adapter_best_f1/`. Each epoch runs in a fresh
subprocess to free vLLM GPU allocations. Selection rejects a validation file that
does not match the original training validation partition. It never reads test.csv.

For RAG selection, omit `--no-rag` and supply `--index indexes/hybrid`, the desired
`--examples`, and optional `--reranker`. Use the same retrieval configuration for
the final test. Do not select on test data or repeatedly tune against test results.

Compare the trained models on validation before deciding:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 --no-rag \
  --eval-csv data/validation.csv --out results/lora_no_rag_val

CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 \
  --index indexes/hybrid \
  --eval-csv data/validation.csv --out results/lora_hybrid_val
```

If another inference configuration wins, select the best epoch for that
configuration separately rather than assuming the same checkpoint is optimal.
The selector refuses to overwrite `adapter_best_f1`; preserve the earlier result
under a different name before a deliberate second selection.

Once the configuration is fixed, run final evaluation **once**:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_classifier.py \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 --no-rag \
  --eval-csv data/test.csv --out results/final_test
```

That example assumes no-RAG won. Include the validated index/reranker/example
settings instead if the RAG configuration won. The evaluator rejects overlap with
training row keys, ticket IDs, normalized text and group IDs, including for adapters
without RAG. The index is also checked against every evaluation row.

## 8. Optional merge and API

Merging is optional: the inference scripts can load the base plus adapter directly.
Use the training environment for merging into full BF16 weights:

```bash
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES=0 python merge_adapter.py \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 \
  --out models/Qwen3-8B-EWOC
```

The merged model needs additional disk space. Run it as
`--model models/Qwen3-8B-EWOC` without `--adapter`. Saved metadata retains leakage
checks and taxonomy validation. The original base files are checked before applying
an adapter; if you replace/move model files in a way that changes their recorded
file stamps, rebuild/retrain rather than silently mixing bases.

Optional local API, in the inference environment:

```bash
source .venv-infer/bin/activate
CUDA_VISIBLE_DEVICES=0 python serve_api.py \
  --model models/Qwen3-8B --adapter runs/ewoc_lora/adapter_best_f1 \
  --no-rag --port 8000
```

`GET http://127.0.0.1:8000/health` reports readiness.
`POST http://127.0.0.1:8000/classify` accepts:

```json
{"description":"Please validate the completed E911 test calls.","include_evidence":false}
```

For RAG, replace `--no-rag` with the validated retrieval settings. The API loads
models once, uses CPU retrieval and serializes synchronous generation. It binds to
localhost and runs one worker; external access/authentication belongs to your
existing deployment infrastructure. Do not run this and another vLLM server on
the same allocated GPU at the same time.

## Practical limits and verification

The local CPU integrity checks were executed. The package has **not** been run
against your CSV, Qwen weights, installed CUDA stack, or H100. There is no measured
accuracy improvement to report yet. Run `python selfcheck.py`, validate package
versions, then use a small validation subset before a full GPU run.

An ambiguous description such as "Intrado cleanup project" may omit the action
needed to distinguish Sector Deletion from another label. Inspect these cases
with domain reviewers. More examples or a larger model cannot reliably recover
information missing from the input. Keep the error/evidence outputs for review.

Primary API references used for implementation:

- Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B
- Qwen embeddings and query instructions: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- BGE cross-encoder reranking: https://huggingface.co/BAAI/bge-reranker-v2-m3
- vLLM constrained offline generation: https://docs.vllm.ai/en/v0.12.0/features/structured_outputs/
- PEFT quantized training: https://huggingface.co/docs/peft/developer_guides/quantization
- Transformers Trainer/checkpoints: https://huggingface.co/docs/transformers/main_classes/trainer
