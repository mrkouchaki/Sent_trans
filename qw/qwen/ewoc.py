"""EWOC commands: prepare, index, predict, evaluate, test, train, select, merge, serve, check."""
from ewoc_core import (
    DEFAULT_DEFINITIONS,
    DEFAULT_LABELS,
    HybridRetriever,
    KINDS,
    QUERY_INSTRUCTION,
    QwenClassifier,
    add_inference_args,
    allowed_label_tokens,
    assert_adapter_holdout,
    assert_disjoint,
    audit,
    digest,
    encode_completion,
    encode_texts,
    fit_prompt,
    format_embedding_text,
    head_tail,
    inspect_embedding_profile,
    label_signature,
    load_definitions,
    load_embedding_encoder,
    load_labels,
    local_embedding_model,
    local_model,
    make_messages,
    make_retriever,
    model_stamp,
    normalize,
    read_rows,
    retrieve_many,
    template_key,
    write_json,
    write_rows,
)

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import json
import csv
import platform
from importlib.metadata import PackageNotFoundError, version
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import random
import shutil
import subprocess
import sys
from threading import Lock
import tempfile
import unittest
import types
from unittest.mock import patch

# ====================== CONFIG: EDIT HERE ==========================
# Run `python ewoc.py`. Change ACTION for each step. Flags remain optional.
ROOT = Path(__file__).resolve().parent
ACTION = 'test'                      # prepare/index/predict/evaluate/test/train/select/merge/serve/check
GPU_ID = '0'                         # Allocated physical GPU; existing CUDA_VISIBLE_DEVICES takes priority
RANDOM_SEED = 2026                   # Reproducible splitting, sampling and training
MODEL_PATH = ROOT / 'models/Qwen3-8B' # LOCAL folder with weights, config and tokenizer
ADAPTER_PATH = None                  # None=base; or ROOT/'runs/ewoc_lora/adapter_best_f1'
LABEL_CONFIG = ROOT / 'labels.json'  # Names + business definitions; review marked draft definitions
DEFINITIONS_PATH = None              # None=use labels.json; optional legacy definitions file

PREPARE = {
    'csv': ROOT / 'whole_ewoc_tickets.csv', # Original labeled CSV
    'out': ROOT / 'data',             # Generated train/validation/test CSVs
    'folds': 10,                      # 10≈80/10/10; 5≈60/20/20; need >=folds groups/class
    'group_templates': False,         # True=also group numeric template variants (heuristic)
}
TRAIN_CSV = ROOT / 'data/train.csv'
VALIDATION_CSV = ROOT / 'data/validation.csv'
TEST_CSV = ROOT / 'data/test.csv'     # Or your independent labeled test CSV

EMBEDDING = {
    'embedding_model': None,          # None=off; local models/bert, models/all-MiniLM-L6-v2, etc.
    'embedding_kind': 'auto',         # auto/bert/minilm/sentence-transformer/qwen (embedding checkpoint)
    'embedding_max_length': None,     # None=native cap; integer can lower it
    'device': 'cpu',                  # cpu/cuda:0; building dense index may use GPU
    'batch_size': 32,                 # Encoder batch size; lower if out of memory
}
# Automatically use a separate index folder when selecting an embedding model.
INDEX_PATH = ROOT / 'indexes' / ('lexical' if EMBEDDING['embedding_model'] is None
                                else Path(EMBEDDING['embedding_model']).name)
USE_RAG = True                       # False=classify directly with Qwen, without retrieval
RAG = {
    'reranker': None,                # None=off; or local BGE reranker folder
    'retrieval_device': 'cpu',       # CPU keeps inference GPU memory free; API requires cpu
    'examples': 12,                  # Maximum retrieved examples in inference prompts
    'max_per_type': 4,               # Maximum examples from each class
    'pool': 80,                      # Retrieval candidate pool
    'rerank_top': 48,                # Candidates scored by optional reranker
}
INFERENCE = {
    'backend': 'transformers',       # transformers=PyTorch locally; vllm=optional faster engine
    'inference_device': 'cuda:0',    # cuda:0/cpu (CPU is slower)
    'inference_dtype': 'bfloat16',   # bfloat16=H100; float16/float32 also supported
    'max_model_len': 8192,           # Prompt+completion budget; lower if memory is limited
    'gpu_memory_utilization': 0.80,  # vLLM only
    'batch_size': 4,                 # Evaluation chunk size; Transformers processes one prompt at a time
}
EVALUATION = {
    'eval_csv': VALIDATION_CSV,      # Validation is for choosing settings/checkpoints
    'out': ROOT / 'results/validation',
    'max_per_class': 200,            # None=all rows; otherwise sample up to this count per class
    'samples_by_class': {},          # Optional overrides: {1: 100, 2: 200, 40: 80}
    'strict_samples': False,         # True=fail if requested counts are unavailable
}
TEST = {
    'eval_csv': TEST_CSV,            # Final or independent test set; never starts training
    'out': ROOT / 'results/test',
    'max_per_class': 200,            # None=all; no sampling with replacement
    'samples_by_class': {},          # Optional per-class overrides, as above
    'strict_samples': False,         # False=use available rows and report shortages
}
FINETUNE = {
    'method': 'lora',                # lora=BF16 LoRA; qlora=4-bit NF4 LoRA (not full-weight training)
    'output': ROOT / 'runs/ewoc_lora',# New run folder; change for each experiment
    'epochs': 3,
    'batch_size': 2,                 # H100 starting value; lower if memory is limited
    'gradient_accumulation': 16,     # Effective batch=2*16 on one GPU
    'max_length': 2048,              # Training token budget; >=512
    'learning_rate': 1e-4,
    'rank': 32,                      # 8/16/32/64
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    'target_modules': 'all-linear', # Or a list of module names, e.g. ['q_proj','v_proj']
    'max_per_class': None,           # Optional TRAIN-only class cap; None=all real rows
    'patience': 2,                   # Early stopping by validation completion loss
    'resume': None,                  # None=new; 'latest'=resume this same run/config
    'rag_index': None,               # None=ordinary SFT; INDEX_PATH=mix in retrieved training examples
    'rag_probability': 0.5,          # Fraction using RAG when rag_index is set
    'rag_examples': 4,
    'retrieval_device': 'cpu',
    'lr_scheduler_type': 'cosine',   # cosine/linear/constant
    'warmup_ratio': 0.05,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'logging_steps': 10,
    'optim': 'adamw_torch',          # Transformers optimizer name; torch AdamW by default
    'gradient_checkpointing': True, # Saves activation memory
    'tf32': True,                   # H100 TF32 matmul support
}
SELECTION = {
    'run': FINETUNE['output'],       # Select saved epochs by full validation macro F1
    'validation_csv': VALIDATION_CSV,
    'out': ROOT / 'results/checkpoint_selection', # Fresh output directory required
}
MERGE = {
    'adapter': ROOT / 'runs/ewoc_lora/adapter', # Or adapter_best_f1
    'out': ROOT / 'models/Qwen3-8B-EWOC',      # New merged BF16 model folder
}
PREDICT = {
    'description': None,            # None=interactive input; or a ticket string
    'text_file': None,              # Optional local UTF-8 file instead of description
    'out': ROOT / 'results/prediction.json',
}
SERVER = {'host': '127.0.0.1', 'port': 8000} # Local endpoint; one worker
# Offline loading is enforced in ewoc_core.py; no login, Hub ID or download.
# ==================================================================


def configured_defaults(command):
    """One source of defaults for scripts and optional CLI, without model imports."""
    shared = {'model': MODEL_PATH, 'labels': LABEL_CONFIG, 'definitions': DEFINITIONS_PATH,
              'seed': RANDOM_SEED}
    inference = dict(shared, adapter=ADAPTER_PATH, index=INDEX_PATH, no_rag=not USE_RAG,
                     **RAG, **INFERENCE)
    sections = {
        'prepare': dict(labels=LABEL_CONFIG, seed=RANDOM_SEED, **PREPARE),
        'index': dict(train_csv=TRAIN_CSV, labels=LABEL_CONFIG, out=INDEX_PATH, **EMBEDDING),
        'predict': dict(inference, **PREDICT),
        'evaluate': dict(inference, **EVALUATION),
        'test': dict(inference, **TEST),
        'train': dict(shared, train_csv=TRAIN_CSV, validation_csv=VALIDATION_CSV, **FINETUNE),
        'select': dict(inference, **SELECTION),
        'merge': dict(model=MODEL_PATH, **MERGE),
        'serve': dict(inference, **SERVER),
        'check': {},
    }
    # argparse paths remain strings, making provenance JSON and subprocesses portable.
    return {k: str(v) if isinstance(v, Path) else v for k, v in sections[command].items()}


def parse_settings(parser, command, argv=None, overrides=None):
    defaults = configured_defaults(command)
    defaults.update(overrides or {})
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for option in parser._actions:
        value = getattr(args, option.dest, None)
        if option.choices and value is not None and value not in option.choices:
            raise ValueError(f'Invalid CONFIG value for {option.dest}: {value}; options: {option.choices}')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
    return args


def prepare(rows, folds=10, seed=2026, group_templates=False):
    if folds < 3:
        raise ValueError('--folds must be >=3')
    original_audit = audit(rows)
    text_labels = defaultdict(set)
    for row in rows:
        text_labels[normalize(row['description'])].add(row['type_id'])
    conflicts = [r for r in rows if len(text_labels[normalize(r['description'])]) > 1]
    rows = [dict(r) for r in rows if len(text_labels[normalize(r['description'])]) == 1]
    unique = {}
    for row in rows:
        unique.setdefault((normalize(row['description']), row['type_id']), row)
    deduped = list(unique.values())
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [('text', normalize(row['description']))]
        if row['ewoc_id']:
            keys.append(('id', row['ewoc_id']))
        if group_templates:
            keys.append(('template', template_key(row['description'])))
        if row.get('group_id'):
            keys.append(('supplied', row['group_id']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    text_group = {normalize(r['description']): find(i) for i, r in enumerate(rows)}
    members = defaultdict(list)
    for r in deduped:
        members[text_group[normalize(r['description'])]].append(r['row_key'])
    group_ids = {k: digest(sorted(v))[:24] for k, v in members.items()}
    for r in deduped:
        r['group_id'] = group_ids[text_group[normalize(r['description'])]]
    groups_by_class = defaultdict(set)
    for r in deduped:
        groups_by_class[r['type_id']].add(r['group_id'])
    insufficient = {k: len(v) for k, v in groups_by_class.items() if len(v) < folds}
    if insufficient:
        raise ValueError(f'Need >= {folds} groups per class; found {insufficient}. Reduce --folds (minimum 3), add real examples, or remove unsupported classes.')
    if not deduped:
        raise ValueError('No unambiguous descriptions remain')
    y = np.array([r['type_id'] for r in deduped])
    g = np.array([r['group_id'] for r in deduped])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.full(len(deduped), -1)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(y)), y, g)):
        assignment[indices] = fold
    parts = {'test': [r for i, r in enumerate(deduped) if assignment[i] == 0], 'validation': [r for i, r in enumerate(deduped) if assignment[i] == 1], 'train': [r for i, r in enumerate(deduped) if assignment[i] >= 2]}
    all_classes = set(y.tolist())
    for name, part in parts.items():
        missing = all_classes - {r['type_id'] for r in part}
        if missing:
            raise ValueError(f'{name} lacks classes {missing}; use fewer folds or more independent groups.')
    assert_disjoint(parts['train'], parts['validation'])
    assert_disjoint(parts['train'], parts['test'])
    assert_disjoint(parts['validation'], parts['test'])
    report = {'source': original_audit, 'exact_duplicate_rows_removed': len(rows) - len(deduped), 'conflicting_rows_quarantined': len(conflicts), 'folds': folds, 'seed': seed, 'group_templates': group_templates, 'split_counts': {k: audit(v) for k, v in parts.items()}, 'largest_groups': Counter((r['group_id'] for r in deduped)).most_common(10), 'limitations': 'Exact text and EWOC_ID isolation guaranteed. Optional numeric-template grouping is heuristic, not general semantic deduplication. No time split: temporal drift unmeasured.'}
    return (parts, conflicts, report)

def prepare_main(argv=None):
    p = argparse.ArgumentParser(description='Audit CSV and create reproducible grouped train/validation/test splits.', prog='ewoc.py prepare')
    p.add_argument('--csv', default='whole_ewoc_tickets.csv')
    p.add_argument('--out', default='data')
    p.add_argument('--labels', default=str(DEFAULT_LABELS))
    p.add_argument('--folds', type=int, default=10, help='Two folds held out; default approximately 80/10/10')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--group-templates', action='store_true')
    args = parse_settings(p, 'prepare', argv)
    labels = load_labels(args.labels)
    rows = read_rows(args.csv, labels)
    if any((r['is_synthetic'] for r in rows)):
        raise ValueError('Use real tickets for this split. Add synthetic rows to training separately only after tracing their source groups; never put synthetic tickets in validation/test.')
    missing = set(labels) - {r['type_id'] for r in rows}
    if missing:
        raise ValueError(f'Configured labels with no real rows: {missing}. Fix labels.json or source data.')
    parts, conflicts, report = prepare(rows, args.folds, args.seed, args.group_templates)
    out = Path(args.out)
    for name, part in parts.items():
        write_rows(out / f'{name}.csv', part)
    write_rows(out / 'label_conflicts.csv', conflicts)
    write_json(out / 'labels.json', labels)
    report['source_csv'] = str(Path(args.csv).resolve())
    report['labels'] = labels
    write_json(out / 'split_audit.json', report)
    print({k: len(v) for k, v in parts.items()})
    print(f'Audit and conflict review: {out.resolve()}')

def index_main(argv=None):
    p = argparse.ArgumentParser(description='Build retrieval index from TRAINING tickets only; run once per split/model.', prog='ewoc.py index')
    p.add_argument('--train-csv', default='data/train.csv')
    p.add_argument('--labels', default=str(DEFAULT_LABELS))
    p.add_argument('--out', default='indexes/lexical')
    p.add_argument('--embedding-model', default=None, help='Optional local MiniLM, BERT, or Qwen embedding model. Omit to disable dense retrieval.')
    p.add_argument('--embedding-kind', choices=KINDS, default='auto')
    p.add_argument('--embedding-max-length', type=int, default=None, help="Optional lower token cap; never exceeds the model's saved/native limit")
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32)
    args = parse_settings(p, 'index', argv)
    rows = read_rows(args.train_csv, load_labels(args.labels))
    if not all((r.get('group_id') for r in rows)):
        raise ValueError('Run ewoc.py prepare first; index requires grouped training CSV.')
    if audit(rows)['conflicting_descriptions']:
        raise ValueError('Conflicting labels remain in training data.')
    path = HybridRetriever.build(rows, load_labels(args.labels), args.out, args.embedding_model, args.device, args.batch_size, args.embedding_kind, args.embedding_max_length)
    print(f'Built {path.resolve()} from {len(rows)} training rows')
    print('Dense channel:', 'enabled' if args.embedding_model else 'disabled (BM25 + character TF-IDF retrieval)')

def predict_main(argv=None):
    p = argparse.ArgumentParser(description='Classify one description or a UTF-8 text file; emit JSON plus retrieval evidence.', prog='ewoc.py predict')
    add_inference_args(p)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--description')
    source.add_argument('--text-file', help='Supports multiline ticket descriptions')
    p.add_argument('--out', default=None, help='Optional complete JSON result including evidence')
    args = parse_settings(p, 'predict', argv)
    description = Path(args.text_file).read_text(encoding='utf-8') if args.text_file else args.description
    if args.text_file and args.description is not None:
        raise ValueError('Set only one of PREDICT description or text_file')
    if description is None:
        description = input('EWOC ticket description: ').strip()
    if not description.strip():
        raise ValueError('Description must not be blank')
    labels = load_labels(args.labels)
    retriever = make_retriever(args, labels)
    evidence = retrieve_many(retriever, [{'description': description}], args)
    classifier = QwenClassifier(args, labels)
    result = classifier.classify_batch([description], evidence)[0]
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}, indent=2))
    if args.out:
        write_json(args.out, result)
        print(f'Evidence saved: {args.out}')

def metrics(actual, predicted, labels):
    all_ids = list(labels)
    report = classification_report(actual, predicted, labels=all_ids, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=all_ids + [-1])
    confusions = [{'actual_type_id': a, 'predicted_type_id': b, 'count': int(matrix[i, j])} for i, a in enumerate(all_ids) for j, b in enumerate(all_ids + [-1]) if a != b and matrix[i, j]]
    confusions.sort(key=lambda r: -r['count'])
    return {'rows': len(actual), 'accuracy': float(accuracy_score(actual, predicted)),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'weighted_precision': float(report['weighted avg']['precision']),
            'weighted_recall': float(report['weighted avg']['recall']),
            'weighted_f1': float(report['weighted avg']['f1-score']),
            'invalid_predictions': int(sum((p == -1 for p in predicted))),
            'classification_report': report, 'confusion_labels': all_ids + [-1],
            'confusion_matrix': matrix.tolist(), 'top_confusions': confusions[:30]}


def select_evaluation_rows(rows, labels, per_class=None, overrides=None, seed=2026, strict=False):
    """Sample without replacement; return requested/available/used counts."""
    if per_class is not None and (type(per_class) is not int or per_class < 1):
        raise ValueError('Samples per class must be a positive integer')
    overrides = {int(k): v for k, v in (overrides or {}).items()}
    unknown = set(overrides) - set(labels)
    if unknown:
        raise ValueError(f'Unknown class IDs in per-class counts: {unknown}')
    if any(type(v) is not int or v < 1 for v in overrides.values()):
        raise ValueError('Per-class overrides must be positive integers')
    rng = np.random.default_rng(seed)
    selected, counts = [], []
    for type_id, name in labels.items():
        candidates = [r for r in rows if r['type_id'] == type_id]
        available = len(candidates)
        requested = overrides.get(type_id, per_class if per_class is not None else available)
        if strict and available < requested:
            raise ValueError(f'Class {type_id} ({name}): requested {requested}, only {available} available in the holdout')
        used = min(requested, available)
        indices = rng.permutation(available)[:used]
        selected.extend(candidates[i] for i in indices)
        counts.append({'type_id': type_id, 'type_name': name, 'requested': requested,
                       'available': available, 'used': used, 'shortage': available < requested})
    if not selected:
        raise ValueError('No evaluation rows selected')
    return selected, counts


def export_metric_tables(prefix, report, labels):
    """Make all requested metrics easy to inspect without opening nested JSON."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ['type_id', 'type_name', 'precision', 'recall', 'f1', 'support']
    with Path(str(prefix)+'_per_class_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for type_id, name in labels.items():
            values = report['classification_report'][str(type_id)]
            writer.writerow({'type_id': type_id, 'type_name': name, 'precision': values['precision'],
                             'recall': values['recall'], 'f1': values['f1-score'], 'support': int(values['support'])})
    scalar_fields = ['rows', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                     'weighted_precision', 'weighted_recall', 'weighted_f1', 'invalid_predictions']
    with Path(str(prefix)+'_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerow({k: report[k] for k in scalar_fields})
    ids = report['confusion_labels']
    names = [f'{i}: {labels.get(i, "INVALID")}' for i in ids]
    with Path(str(prefix)+'_confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['actual / predicted', *names])
        for name, values in zip(names, report['confusion_matrix']):
            writer.writerow([name, *values])
    if report.get('sample_counts'):
        with Path(str(prefix)+'_sample_counts.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(report['sample_counts'][0]))
            writer.writeheader()
            writer.writerows(report['sample_counts'])

def evaluate_main(argv=None, _command='evaluate', settings=None):
    p = argparse.ArgumentParser(description='Evaluate base/LoRA Qwen with or without retrieval on a disjoint held-out CSV.', prog='ewoc.py evaluate')
    add_inference_args(p)
    p.add_argument('--eval-csv', default='data/validation.csv')
    p.add_argument('--out', default='results/base_rag_validation')
    p.add_argument('--max-per-class', '--samples-per-class', dest='max_per_class', type=int, default=None,
                   help='Up to this many real holdout rows per class; sampling is without replacement')
    p.add_argument('--samples-by-class', default=None, help='Optional JSON overrides, e.g. {"1":100,"2":200}')
    p.add_argument('--strict-samples', action=argparse.BooleanOptionalAction, help='Fail if any requested class count is unavailable')
    p.add_argument('--all-samples', dest='max_per_class', action='store_const', const=None, help='Use all rows per class')
    args = parse_settings(p, _command, argv, settings)
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    labels = load_labels(args.labels)
    rows = read_rows(args.eval_csv, labels)
    if any((r['is_synthetic'] for r in rows)):
        raise ValueError('Validation/test must contain real tickets only')
    retriever = make_retriever(args, labels)
    if retriever:
        assert_disjoint(retriever.rows, rows, 'retrieval references versus evaluation')
    metadata_path = Path(args.adapter or args.model) / 'ewoc_training_metadata.json'
    if args.adapter or metadata_path.exists():
        assert_adapter_holdout(json.loads(metadata_path.read_text()), rows)
    overrides = (json.loads(args.samples_by_class) if isinstance(args.samples_by_class, str)
                 else args.samples_by_class) or {}
    if not isinstance(overrides, dict):
        raise ValueError('--samples-by-class must be a JSON object mapping class IDs to counts')
    rows, sample_counts = select_evaluation_rows(rows, labels, args.max_per_class,
                                                overrides, args.seed, args.strict_samples)
    print('Evaluation samples (TYPE_ID: requested / available / used):')
    for count in sample_counts:
        suffix = ' [fewer available]' if count['shortage'] else ''
        print(f"  {count['type_id']}: {count['requested']} / {count['available']} / {count['used']}{suffix}")
    evidence = retrieve_many(retriever, rows, args)
    classifier = QwenClassifier(args, labels)
    results = []
    for start in range(0, len(rows), args.batch_size):
        part = rows[start:start + args.batch_size]
        predictions = classifier.classify_batch([r['description'] for r in part], evidence[start:start + args.batch_size])
        for row, prediction in zip(part, predictions):
            results.append({'ewoc_id': row['ewoc_id'], 'row_key': row['row_key'], 'description': row['description'], 'actual_type_id': row['type_id'], 'actual_type_name': labels[row['type_id']], 'predicted_type_id': prediction['type_id'] if prediction['type_id'] is not None else -1, 'predicted_type_name': prediction['type_name'], 'correct': prediction['type_id'] == row['type_id'], 'raw_response': prediction['raw'], 'finish_reason': prediction['finish_reason'], 'examples_used': prediction['prompt_stats']['examples_used'], 'evidence': prediction['evidence']})
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in results[0] if k != 'evidence']
    for suffix, subset in [('predictions', results), ('errors', [r for r in results if not r['correct']])]:
        with Path(str(prefix) + f'_{suffix}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(subset)
    with Path(str(prefix) + '_evidence.jsonl').open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps({'row_key': r['row_key'], 'evidence': r['evidence']}, ensure_ascii=False) + '\n')
    report = metrics([r['actual_type_id'] for r in results], [r['predicted_type_id'] for r in results], labels)
    packages = {}
    for name in ['vllm', 'torch', 'transformers', 'sentence-transformers', 'peft', 'scikit-learn']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report.update({'configuration': vars(args), 'labels': labels, 'python': platform.python_version(), 'packages': packages, 'evaluated_row_fingerprint': digest(sorted((r['row_key'] for r in rows))), 'index_fingerprint': retriever.data['fingerprint'] if retriever else None, 'retrieval_true_class_coverage': float(np.mean([any((ex['type_id'] == r['actual_type_id'] for ex in r['evidence'])) for r in results])) if retriever else None, 'note': 'Retrieval scores and example agreement are not calibrated class probabilities.'})
    report['sample_counts'] = sample_counts
    report['sampling_seed'] = args.seed
    report['macro_average_scope'] = 'All configured classes; inspect support for classes with zero selected rows'
    write_json(str(prefix) + '_metrics.json', report)
    export_metric_tables(prefix, report, labels)
    print(json.dumps({k: report[k] for k in ['rows', 'accuracy', 'macro_precision', 'macro_recall',
                                           'macro_f1', 'weighted_precision', 'weighted_recall',
                                           'weighted_f1', 'invalid_predictions']}, indent=2))
    print(classification_report([r['actual_type_id'] for r in results], [r['predicted_type_id'] for r in results],
                                labels=list(labels), target_names=[f'{k}: {v}' for k,v in labels.items()],
                                digits=4, zero_division=0))
    print('Confusion matrix: actual rows / predicted columns; labels', report['confusion_labels'])
    print(np.asarray(report['confusion_matrix']))
    print(f'Saved predictions, errors, evidence, per-class metrics, sample counts and confusion matrix: {prefix}')


def test_main(argv=None):
    """A separate model-test run; this never starts fine-tuning."""
    return evaluate_main(argv, _command='test')

def train_main(argv=None):
    p = argparse.ArgumentParser(description='Single-H100 BF16 LoRA or NF4 QLoRA for Qwen3-8B.', prog='ewoc.py train')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--train-csv', default='data/train.csv')
    p.add_argument('--validation-csv', default='data/validation.csv')
    p.add_argument('--labels', default=str(DEFAULT_LABELS))
    p.add_argument('--definitions', default=None, help='Optional legacy definition file; otherwise use --labels')
    p.add_argument('--output', default='runs/ewoc_lora')
    p.add_argument('--method', choices=['lora', 'qlora'], default='lora')
    p.add_argument('--quantization', choices=['none', 'nf4'], default=None, help='none=BF16 LoRA (H100 default); nf4=4-bit QLoRA with BF16 compute')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--gradient-accumulation', type=int, default=16)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--learning-rate', type=float, default=0.0001)
    p.add_argument('--rank', type=int, choices=[8, 16, 32, 64], default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--max-per-class', type=int, default=None, help='Optional TRAIN-only cap; no oversampling')
    p.add_argument('--patience', type=int, default=2, help='Early stopping by validation loss')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--resume', default=None, help='Checkpoint directory, or latest; restores optimizer/scheduler')
    p.add_argument('--rag-index', default=None, help='Optional index built solely from this train.csv')
    p.add_argument('--rag-probability', type=float, default=0.5, help='With --rag-index, train a mix of retrieved and zero-shot prompts')
    p.add_argument('--rag-examples', type=int, default=4)
    p.add_argument('--retrieval-device', default='cpu')
    p.add_argument('--target-modules', nargs='+')
    p.add_argument('--lr-scheduler-type', choices=['cosine', 'linear', 'constant'])
    p.add_argument('--warmup-ratio', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--max-grad-norm', type=float)
    p.add_argument('--logging-steps', type=int)
    p.add_argument('--optim')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction)
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction)
    args = parse_settings(p, 'train', argv)
    if args.quantization is not None:  # Preserve previous optional CLI spelling.
        args.method = 'qlora' if args.quantization == 'nf4' else 'lora'
    if args.method not in {'lora', 'qlora'}:
        raise ValueError("FINETUNE method must be 'lora' or 'qlora'")
    args.quantization = 'nf4' if args.method == 'qlora' else 'none'
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This script targets one H100. Launch with python, not torchrun.')
    if not 0 <= args.rag_probability <= 1:
        raise ValueError('--rag-probability must be between 0 and 1')
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.patience) < 1:
        raise ValueError('Epochs, batch size, accumulation, patience must be positive')
    if args.max_length < 512 or args.learning_rate <= 0:
        raise ValueError('Use max-length >=512 and positive learning rate')
    if args.max_per_class is not None and args.max_per_class < 1:
        raise ValueError('--max-per-class must be positive')
    if not 0 <= args.warmup_ratio <= 1 or args.weight_decay < 0 or args.max_grad_norm <= 0 or args.logging_steps < 1:
        raise ValueError('Check warmup_ratio, weight_decay, max_grad_norm and logging_steps')
    if not args.target_modules or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError('Check target_modules, lora_alpha and lora_dropout')
    if args.max_length % 8:
        raise ValueError('max_length must be a multiple of 8 for padded training batches')
    labels = load_labels(args.labels)
    definitions = load_definitions(args.definitions or args.labels, labels)
    train = read_rows(args.train_csv, labels)
    validation = read_rows(args.validation_csv, labels)
    if not all((r.get('group_id') for r in train + validation)):
        raise ValueError('Run ewoc.py prepare first; grouped CSVs are required')
    if any((r['is_synthetic'] for r in train + validation)):
        raise ValueError('This workflow expects real tickets. Synthetic lineage is not validated here.')
    assert_disjoint(train, validation, 'training versus validation')
    for name, rows in [('train', train), ('validation', validation)]:
        missing = set(labels) - {r['type_id'] for r in rows}
        if missing:
            raise ValueError(f'{name} missing configured classes: {missing}')
    source_train = train[:]
    rng = random.Random(args.seed)
    if args.max_per_class:
        by_class = defaultdict(list)
        for row in train:
            by_class[row['type_id']].append(row)
        train = []
        for key in sorted(by_class):
            rng.shuffle(by_class[key])
            train.extend(by_class[key][:args.max_per_class])
        rng.shuffle(train)
    base = local_model(args.model)
    output = Path(args.output)
    checkpoint_dir = output / 'checkpoints'
    if output.exists() and any(output.iterdir()) and (not args.resume):
        raise FileExistsError(f'{output} is nonempty. Use --resume latest or a new --output.')
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'base_model_path': base, 'base_model_stamp': model_stamp(base), 'label_signature': label_signature(labels, definitions), 'labels': labels, 'definitions': definitions, 'lora_rank': args.rank, 'quantization': args.quantization, 'training_row_keys': sorted({r['row_key'] for r in source_train}), 'training_group_ids': sorted({r['group_id'] for r in source_train}), 'training_ticket_hashes': sorted({digest(r['ewoc_id']) for r in source_train if r['ewoc_id']}), 'training_text_hashes': sorted({digest(normalize(r['description'])) for r in source_train}), 'validation_row_fingerprint': digest(sorted((r['row_key'] for r in validation))), 'selection_metric': 'validation completion loss (not classification macro F1)', 'training_class_counts': dict(Counter((r['type_id'] for r in train))), 'rag_mixture': bool(args.rag_index), 'config': vars(args)}
    run_config = {k: v for k, v in vars(args).items() if k != 'resume'}
    metadata['run_signature'] = digest({'config': run_config, 'train': metadata['training_row_keys'], 'validation': metadata['validation_row_fingerprint'], 'taxonomy': metadata['label_signature'], 'base': metadata['base_model_stamp']})
    metadata_path = output / 'run_metadata.json'
    if args.resume:
        if not metadata_path.exists():
            raise ValueError('Resume requires existing run_metadata.json')
        previous = json.loads(metadata_path.read_text())
        if previous['run_signature'] != metadata['run_signature']:
            raise ValueError('Resume data/model/config changed. Use original arguments or start a new run.')
    train_examples = [[] for _ in train]
    val_examples = [[] for _ in validation]
    if args.rag_index:
        retriever = HybridRetriever(args.rag_index, device=args.retrieval_device)
        if retriever.labels != labels:
            raise ValueError('RAG index taxonomy differs from training labels')
        source_keys = {r['row_key'] for r in source_train}
        if not {r['row_key'] for r in retriever.rows}.issubset(source_keys):
            raise ValueError('Training RAG index includes rows outside train.csv')
        assert_disjoint(retriever.rows, validation, 'training RAG references versus validation')

        def evidence_for(rows):
            selected = [i for i in range(len(rows)) if rng.random() < args.rag_probability]
            result = [[] for _ in rows]
            vectors = retriever.encode_queries([rows[i]['description'] for i in selected]) if selected else []
            for i, q in zip(selected, vectors):
                r = rows[i]
                result[i] = retriever.retrieve(r['description'], count=args.rag_examples, max_per_type=2, query_embedding=q, exclude_row_key=r['row_key'], exclude_group=r['group_id'])
            return result
        train_examples = evidence_for(train)
        val_examples = evidence_for(validation)
        retriever.release_neural_models()
        metadata['training_rag_index_fingerprint'] = retriever.data['fingerprint']
        if args.resume and previous.get('training_rag_index_fingerprint') != metadata['training_rag_index_fingerprint']:
            raise ValueError('RAG index changed since checkpoint')
    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required (H100 supported).')
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    class CompletionDataset(Dataset):

        def __init__(self, rows, evidence):
            self.items = []
            shortened = 0
            for row, examples in zip(rows, evidence):
                encoded, stats = encode_completion(tokenizer, row, examples, labels, definitions, args.max_length)
                self.items.append(encoded)
                shortened += stats['description_char_budget'] < min(6000, len(row['description']))
            print(f'Tokenized {len(self.items)} tickets; further shortened {shortened} long descriptions')

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]
    train_dataset = CompletionDataset(train, train_examples)
    val_dataset = CompletionDataset(validation, val_examples)

    def collate(items):
        length = (max((len(i['input_ids']) for i in items)) + 7) // 8 * 8
        batch = {}
        for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]:
            batch[key] = torch.tensor([item[key] + [pad] * (length - len(item[key])) for item in items], dtype=torch.long)
        return batch
    model_kwargs = dict(local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    if args.quantization == 'nf4':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
    model.config.use_cache = False
    if args.quantization == 'nf4':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing, gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias='none'))
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(checkpoint_dir), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler_type, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, bf16=True, tf32=args.tf32, gradient_checkpointing=args.gradient_checkpointing, gradient_checkpointing_kwargs={'use_reentrant': False}, optim=args.optim, eval_strategy='epoch', save_strategy='epoch', save_total_limit=None, save_safetensors=True, save_only_model=False, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, prediction_loss_only=True, logging_steps=args.logging_steps, report_to='none', push_to_hub=False, seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, remove_unused_columns=False, label_names=['labels'])

    class EpochMetadataCallback(TrainerCallback):

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            write_json(checkpoint / 'ewoc_training_metadata.json', dict(metadata, checkpoint_epoch=state.epoch))
            write_json(checkpoint / 'labels.json', labels)
            write_json(checkpoint / 'label_definitions.json', definitions)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=collate, processing_class=tokenizer, callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), EpochMetadataCallback()])
    resume = None
    if args.resume:
        resume = get_last_checkpoint(str(checkpoint_dir)) if args.resume == 'latest' else args.resume
        if not resume or not Path(resume).is_dir():
            raise ValueError('No resumable checkpoint found')
        if Path(resume).resolve().parent != checkpoint_dir.resolve():
            raise ValueError("Resume checkpoint must be inside this run's checkpoints/ directory")
    for package in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes']:
        try:
            metadata.setdefault('packages', {})[package] = version(package)
        except PackageNotFoundError:
            pass
    write_json(metadata_path, metadata)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_metrics('train', result.metrics)
    trainer.save_metrics('eval', trainer.evaluate())
    adapter_dir = output / 'adapter'
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    metadata['best_checkpoint'] = trainer.state.best_model_checkpoint
    metadata['best_validation_loss'] = trainer.state.best_metric
    metadata['training_finished'] = True
    write_json(adapter_dir / 'ewoc_training_metadata.json', metadata)
    write_json(adapter_dir / 'labels.json', labels)
    write_json(adapter_dir / 'label_definitions.json', definitions)
    write_json(output / 'training_history.json', trainer.state.log_history)
    write_json(metadata_path, metadata)
    print(f'Best adapter: {adapter_dir.resolve()}')
    print(f'Every epoch checkpoint (optimizer + scheduler included): {checkpoint_dir.resolve()}')
    print('Evaluate macro F1 using ewoc.py evaluate; validation loss is not classification accuracy.')

def select_main(argv=None):
    p = argparse.ArgumentParser(description='Select an epoch adapter by generated validation macro F1, in the inference env.', prog='ewoc.py select')
    add_inference_args(p)
    p.add_argument('--run', default='runs/ewoc_lora')
    p.add_argument('--validation-csv', default='data/validation.csv')
    p.add_argument('--out', default='results/checkpoint_selection')
    args = parse_settings(p, 'select', argv)
    if args.adapter:
        raise ValueError('Use --run to select checkpoints, not --adapter')
    run = Path(args.run)
    metadata = json.loads((run / 'run_metadata.json').read_text())
    rows = read_rows(args.validation_csv, load_labels(args.labels))
    if digest(sorted((r['row_key'] for r in rows))) != metadata['validation_row_fingerprint']:
        raise ValueError('Selection must use the original complete validation partition, never the test partition')
    checkpoints = sorted((run / 'checkpoints').glob('checkpoint-*'), key=lambda d: int(d.name.rsplit('-', 1)[1]))
    checkpoints = [d for d in checkpoints if (d / 'adapter_config.json').exists() and (d / 'ewoc_training_metadata.json').exists()]
    if not checkpoints:
        raise ValueError('No EWOC epoch checkpoints found')
    selected_out = run / 'adapter_best_f1'
    if selected_out.exists():
        raise FileExistsError(f'{selected_out} exists. Move it or use a separate run; selection never overwrites it.')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh --out directory for reproducible selection')
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in checkpoints:
        prefix = out / checkpoint.name
        command = [sys.executable, str(Path(__file__).resolve()), 'evaluate', '--all-samples', '--samples-by-class', '{}', '--no-strict-samples', '--backend', args.backend, '--inference-device', args.inference_device, '--inference-dtype', args.inference_dtype, '--model', args.model, '--adapter', str(checkpoint), '--labels', args.labels, '--definitions', args.definitions or args.labels, '--eval-csv', args.validation_csv, '--out', str(prefix), '--index', args.index, '--retrieval-device', args.retrieval_device, '--examples', str(args.examples), '--max-per-type', str(args.max_per_type), '--pool', str(args.pool), '--rerank-top', str(args.rerank_top), '--max-model-len', str(args.max_model_len), '--batch-size', str(args.batch_size), '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--seed', str(args.seed)]
        command.append('--no-rag' if args.no_rag else '--rag')
        if args.reranker:
            command.extend(['--reranker', args.reranker])
        subprocess.run(command, check=True)
        report = json.loads(Path(str(prefix) + '_metrics.json').read_text())
        results.append({'checkpoint': str(checkpoint), 'macro_f1': report['macro_f1'], 'accuracy': report['accuracy'], 'weighted_f1': report['weighted_f1']})
        write_json(out / 'selection_progress.json', results)
    best = max(results, key=lambda r: r['macro_f1'])
    source = Path(best['checkpoint'])
    selected_out.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and (file.name.startswith('adapter_') or file.name.startswith('tokenizer') or file.name in {'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja', 'labels.json', 'label_definitions.json', 'ewoc_training_metadata.json'}):
            shutil.copy2(file, selected_out / file.name)
    selected_meta = json.loads((selected_out / 'ewoc_training_metadata.json').read_text())
    selected_meta.update({'selection_metric': 'generated validation macro F1', 'selected_checkpoint': str(source), 'selection_scores': best, 'selection_inference_configuration': vars(args)})
    write_json(selected_out / 'ewoc_training_metadata.json', selected_meta)
    write_json(out / 'selection_result.json', {'best': best, 'all_checkpoints': results, 'selected_adapter': str(selected_out), 'configuration': vars(args)})
    print(f"Selected adapter: {selected_out.resolve()} (validation macro F1 {best['macro_f1']:.4f})")
    print('Keep this inference configuration for final test evaluation.')

def merge_main(argv=None):
    p = argparse.ArgumentParser(description='Optional: merge LoRA into a BF16 base for ordinary vLLM serving (no adapter flag).', prog='ewoc.py merge')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--adapter', default='runs/ewoc_lora/adapter')
    p.add_argument('--out', default='models/Qwen3-8B-EWOC')
    args = parse_settings(p, 'merge', argv)
    base, adapter = (local_model(args.model), local_model(args.adapter))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty/new output directory')
    metadata = json.loads((Path(adapter) / 'ewoc_training_metadata.json').read_text())
    if model_stamp(base) != metadata['base_model_stamp']:
        raise ValueError('Base files differ from the training base. Use the exact original local model.')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(adapter, local_files_only=True).save_pretrained(out)
    write_json(out / 'ewoc_training_metadata.json', metadata)
    write_json(out / 'labels.json', metadata['labels'])
    write_json(out / 'label_definitions.json', metadata['definitions'])
    print(f'Merged BF16 model: {out.resolve()}')

def serve_main(argv=None):
    p = argparse.ArgumentParser(description='Optional local FastAPI endpoint loading Qwen and the index once per process.', prog='ewoc.py serve')
    add_inference_args(p)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    args = parse_settings(p, 'serve', argv)
    if args.retrieval_device != 'cpu':
        raise ValueError('API retrieval uses CPU; reserve the GPU for vLLM. Use --retrieval-device cpu.')
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
    labels = load_labels(args.labels)
    retriever = make_retriever(args, labels)
    classifier = QwenClassifier(args, labels)
    lock = Lock()
    app = FastAPI(title='EWOC Qwen Classifier', version='1.0')

    class Request(BaseModel):
        description: str = Field(min_length=1, max_length=50000)
        include_evidence: bool = False

    @app.get('/health')
    def health():
        return {'ready': True, 'rag': retriever is not None, 'adapter': bool(args.adapter), 'classes': len(labels)}

    @app.post('/classify')
    def classify(request: Request):
        if not request.description.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail='Description cannot be blank')
        with lock:
            examples = retriever.retrieve(request.description, count=args.examples, max_per_type=args.max_per_type, pool=args.pool, rerank_top=args.rerank_top) if retriever else []
            result = classifier.classify_batch([request.description], [examples])[0]
        if not request.include_evidence:
            result.pop('evidence', None)
        return result
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
LABELS = {1: 'Call Test Validation', 2: 'E911 Provisioning', 40: 'BMC/WEA'}

def row(text, label=1, ticket='', group=''):
    return {'description': text, 'type_id': label, 'ewoc_id': ticket, 'group_id': group, 'row_key': digest([text, label, ticket])[:24], 'is_synthetic': False}

def corpus():
    return [row(f'{task} requested for uniquely named node {i:03d}', label, f'{label}-{i}') for label, task in [(1, 'Validate completed emergency calls'), (2, 'Provision emergency routing entries'), (40, 'BMC WEA broadcast configuration')] for i in range(80)]

class TinyTokenizer:
    eos_token = '\x03'

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def apply_chat_template(self, messages, **kwargs):
        return ''.join((m['role'] + ':' + m['content'] + '\n' for m in messages)) + 'assistant:<think></think>\n'

class Checks(unittest.TestCase):

    def test_plain_language_model_rejected_as_dense_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'config.json').write_text('{"architectures":["Qwen3ForCausalLM"]}')
            with self.assertRaisesRegex(ValueError, 'generation models'):
                local_embedding_model(d)

    def test_embedding_pipeline_can_share_causal_architecture(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'config.json').write_text('{"architectures":["Qwen3ForCausalLM"]}')
            (root / 'modules.json').write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
            (root / '1_Pooling').mkdir()
            (root / '1_Pooling' / 'config.json').write_text('{"pooling_mode_lasttoken":true}')
            self.assertEqual(local_embedding_model(d), str(root.resolve()))

    def test_grouped_partitions_are_disjoint_and_reproducible(self):
        a, _, _ = prepare(corpus(), folds=5)
        b, _, _ = prepare(corpus(), folds=5)
        self.assertEqual(a, b)
        for key in a:
            self.assertEqual({r['type_id'] for r in a[key]}, set(LABELS))
        assert_disjoint(a['train'], a['test'])
        assert_disjoint(a['train'], a['validation'])
        assert_disjoint(a['test'], a['validation'])

    def test_conflicting_input_is_quarantined(self):
        rows = corpus() + [row('same ambiguous input', 1), row('same ambiguous input', 2)]
        parts, conflicts, report = prepare(rows, folds=5)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(report['conflicting_rows_quarantined'], 2)
        self.assertFalse(any((r['description'] == 'same ambiguous input' for part in parts.values() for r in part)))

    def test_duplicate_normalization_and_id_bridges(self):
        rows = corpus() + [row('One common description', 1, 'BRIDGE-A'), row(' one  COMMON description ', 1, 'BRIDGE-B'), row('Another description joined by ticket ID', 1, 'BRIDGE-B')]
        parts, _, report = prepare(rows, folds=5)
        self.assertEqual(report['exact_duplicate_rows_removed'], 1)
        selected = [r for part in parts.values() for r in part if r['ewoc_id'].startswith('BRIDGE')]
        self.assertEqual(len({r['group_id'] for r in selected}), 1)

    def test_template_grouping_links_identifier_variations(self):
        rows = corpus() + [row('Unique special task for NODE12345678', 1, 'x'), row('Unique special task for NODE98765432', 1, 'y')]
        parts, _, _ = prepare(rows, folds=5, group_templates=True)
        selected = [r for part in parts.values() for r in part if r['ewoc_id'] in {'x', 'y'}]
        self.assertEqual(len({r['group_id'] for r in selected}), 1)

    def test_overlap_rejected(self):
        a = row('same input', ticket='A', group='G')
        with self.assertRaises(ValueError):
            assert_disjoint([a], [dict(a, row_key='other')])
        with self.assertRaises(ValueError):
            assert_adapter_holdout({'training_text_hashes': [digest(normalize(a['description']))]}, [a])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rows.csv'
            rows = corpus()[:5]
            write_rows(path, rows)
            self.assertEqual(read_rows(path, LABELS), rows)

    def test_lexical_retrieval_and_group_exclusion(self):
        rows = [row('Validate completed emergency calls', 1, 'A', 'G1'), row('Verify completed emergency test calls', 1, 'B', 'G2'), row('Provision PSAP routing entries', 2, 'C', 'G3'), row('BMC WEA broadcast configuration update', 40, 'D', 'G4')]
        with tempfile.TemporaryDirectory() as d:
            HybridRetriever.build(rows, LABELS, d)
            retriever = HybridRetriever(d)
            examples = retriever.retrieve('BMC WEA broadcast configuration', count=1, max_per_type=1)
            self.assertEqual(examples[0]['type_id'], 40)
            excluded = retriever.retrieve('Verify completed emergency calls', count=2, max_per_type=1, exclude_group='G1')
            self.assertTrue(all((r['group_id'] != 'G1' for r in excluded)))
            self.assertEqual(len({r['type_id'] for r in excluded}), len(excluded))

    def test_index_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            HybridRetriever.build(corpus(), LABELS, d)
            with self.assertRaises(FileExistsError):
                HybridRetriever.build(corpus(), LABELS, d)

    def test_prompt_only_loss_is_masked(self):
        tokenizer = TinyTokenizer()
        encoded, _ = encode_completion(tokenizer, row('Broadcast configuration', 40), [], LABELS, {}, 4096)
        supervised = [v for v in encoded['labels'] if v != -100]
        self.assertEqual(supervised, list(b'40\x03'))
        self.assertTrue(all((x == -100 for x in encoded['labels'][:-3])))
        self.assertEqual(len(encoded['input_ids']), len(encoded['labels']))

    def test_context_budget_removes_examples_before_ticket(self):
        tokenizer = TinyTokenizer()
        examples = [row('Example ' * 100, 1) for _ in range(12)]
        _, ids, stats = fit_prompt(tokenizer, 'Important action at the END', examples, LABELS, {}, 2200)
        self.assertLess(stats['examples_used'], 12)
        self.assertLessEqual(len(ids), 2200)
        self.assertIn(b'Important action', bytes(ids))

    def test_long_query_keeps_start_and_end(self):
        tokenizer = TinyTokenizer()
        text = 'START action ' + 'x ' * 5000 + ' END action'
        _, ids, _ = fit_prompt(tokenizer, text, [], LABELS, {}, 2000)
        self.assertLessEqual(len(ids), 2000)
        self.assertIn(b'START action', bytes(ids))
        self.assertIn(b'END action', bytes(ids))

    def test_invalid_predictions_count_as_errors(self):
        report = metrics([1, 2, 40], [1, -1, 2], LABELS)
        self.assertAlmostEqual(report['accuracy'], 1 / 3)
        self.assertEqual(report['invalid_predictions'], 1)
        self.assertEqual(sum(map(sum, report['confusion_matrix'])), 3)

class Tokenizer:

    def encode(self, text, add_special_tokens=False):
        return list(str(text).encode())

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode()

    def num_special_tokens_to_add(self, pair=False):
        return 2

class Encoder:

    def __init__(self, limit=256):
        self.max_seq_length = limit
        self.tokenizer = Tokenizer()
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        values = np.array([[1.0, text.count('call') + 1.0, text.count('routing') + 1.0] for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

def fixture(path, model_type='bert', pipeline=True):
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({'model_type': model_type, 'architectures': ['Qwen3ForCausalLM' if model_type == 'qwen3' else 'BertModel']}))
    if pipeline:
        (path / 'modules.json').write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
        (path / '1_Pooling').mkdir()
        (path / '1_Pooling' / 'config.json').write_text('{"pooling_mode_mean_tokens":true}')

class OptionalEmbeddings(unittest.TestCase):

    def test_minilm_has_plain_query_format(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'minilm'
            fixture(model)
            profile = inspect_embedding_profile(model, 'minilm')
            self.assertIsNone(profile['query_instruction'])
            self.assertEqual(format_embedding_text(Encoder(), 'call validation', profile, True), 'call validation')

    def test_raw_bert_explicitly_uses_mean_pooling(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'bert'
            fixture(model, pipeline=False)
            profile = inspect_embedding_profile(model)
            self.assertEqual(profile['pipeline'], 'bert-mean-pooling')
            self.assertIsNone(profile['query_instruction'])

    def test_qwen_generation_rejected_but_embedding_pipeline_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'qwen'
            fixture(model, 'qwen3', False)
            with self.assertRaises(ValueError):
                inspect_embedding_profile(model)
            (model / 'modules.json').write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
            (model / '1_Pooling').mkdir()
            (model / '1_Pooling' / 'config.json').write_text('{}')
            profile = inspect_embedding_profile(model)
            self.assertEqual(profile['kind'], 'qwen')
            self.assertTrue(format_embedding_text(Encoder(512), 'call validation', profile, True).startswith('Instruct: '))
            self.assertEqual(format_embedding_text(Encoder(), 'call validation', profile, False), 'call validation')

    def test_native_limit_is_never_increased(self):
        fake = types.SimpleNamespace(SentenceTransformer=lambda *a, **kw: Encoder(256), models=object())
        with patch.dict(sys.modules, {'sentence_transformers': fake}):
            encoder = load_embedding_encoder('unused', {'pipeline': 'saved', 'kind': 'minilm', 'requested_max_length': 1024})
            self.assertEqual(encoder.max_seq_length, 256)
            encoder = load_embedding_encoder('unused', {'pipeline': 'saved', 'kind': 'minilm', 'requested_max_length': 128})
            self.assertEqual(encoder.max_seq_length, 128)

    def test_long_minilm_input_keeps_both_ends(self):
        formatted = format_embedding_text(Encoder(128), 'START ' + 'x' * 5000 + ' END', {'query_instruction': None}, True)
        self.assertTrue(formatted.startswith('START'))
        self.assertTrue(formatted.endswith('END'))
        self.assertLessEqual(len(formatted.encode()) + 2, 128)

    def test_dense_index_persists_adapter_and_uses_same_query_format(self):
        with tempfile.TemporaryDirectory() as d:
            model, index = (Path(d) / 'minilm', Path(d) / 'index')
            fixture(model)
            encoder = Encoder(256)
            rows = [row('validate emergency call', 1, 'A', 'G1'), row('provision routing entries', 2, 'B', 'G2')]
            with patch('ewoc_core.load_embedding_encoder', return_value=encoder):
                HybridRetriever.build(rows, {1: 'Calls', 2: 'Provisioning'}, index, model, embedding_kind='minilm')
                retrieved = HybridRetriever(index)
                query = retrieved.encode_queries(['validate emergency call'])
                hits = retrieved.retrieve('validate emergency call', count=1, query_embedding=query[0])
                self.assertEqual(hits[0]['type_id'], 1)
                self.assertEqual(retrieved.data['embedding_profile']['effective_max_length'], 256)
                self.assertEqual(retrieved.embeddings.shape, (2, 3))
                self.assertFalse(any((text.startswith('Instruct:') for text in encoder.seen)))
                self.assertIn('no vector database', json.loads((index / 'index_manifest.json').read_text())['vector_backend'])


class ConsolidatedChecks(unittest.TestCase):
    def test_unified_names_and_definitions_config(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d)/'labels.json'
            write_json(config, {'1': {'name': 'Calls', 'definition': 'Validated call results'}})
            self.assertEqual(load_labels(config), {1: 'Calls'})
            self.assertEqual(load_definitions(config), {1: 'Validated call results'})
            write_json(config, {'1': 'Calls'})
            self.assertEqual(load_labels(config), {1: 'Calls'})

    def test_selection_invokes_unified_evaluate_command(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root/'run'
            run.mkdir()
            data = [row('Held-out call verification', 1, 'VAL', 'V')]
            write_rows(root/'validation.csv', data)
            write_json(root/'labels.json', {'1': {'name': 'Calls', 'definition': ''}})
            write_json(run/'run_metadata.json', {'validation_row_fingerprint': digest(sorted(r['row_key'] for r in data))})
            for step in [1, 2]:
                checkpoint = run/'checkpoints'/f'checkpoint-{step}'
                checkpoint.mkdir(parents=True)
                write_json(checkpoint/'adapter_config.json', {})
                write_json(checkpoint/'ewoc_training_metadata.json', {'checkpoint': step})
                (checkpoint/'adapter_model.safetensors').write_bytes(str(step).encode())
            def fake_evaluate(command, check=True):
                self.assertEqual(command[2], 'evaluate')
                self.assertEqual(Path(command[1]).name, 'ewoc.py')
                prefix = command[command.index('--out')+1]
                step = int(Path(command[command.index('--adapter')+1]).name.rsplit('-', 1)[1])
                write_json(prefix+'_metrics.json', {'macro_f1': step*.4, 'accuracy': .5, 'weighted_f1': .5})
            with patch('subprocess.run', side_effect=fake_evaluate):
                select_main(['--run', str(run), '--validation-csv', str(root/'validation.csv'),
                             '--labels', str(root/'labels.json'), '--no-rag', '--out', str(root/'selection')])
            self.assertEqual((run/'adapter_best_f1'/'adapter_model.safetensors').read_bytes(), b'2')

    def test_api_routes_survive_consolidation(self):
        routes = {}
        class App:
            def __init__(self, **kwargs): pass
            def get(self, path):
                return lambda func: routes.setdefault(path, func)
            def post(self, path):
                def register(func):
                    self_type = func.__annotations__['request']
                    if not isinstance(self_type, type):
                        raise AssertionError('Request annotation must resolve to the local request class')
                    routes[path] = func
                    return func
                return register
        fake_classifier = types.SimpleNamespace(classify_batch=lambda *a: [{'type_id': 1, 'evidence': []}])
        fakes = {'fastapi': types.SimpleNamespace(FastAPI=App),
                 'pydantic': types.SimpleNamespace(BaseModel=object, Field=lambda **kw: None),
                 'uvicorn': types.SimpleNamespace(run=lambda *a, **kw: None)}
        with patch.dict(sys.modules, fakes), patch(__name__+'.QwenClassifier', return_value=fake_classifier):
            serve_main(['--no-rag'])
        self.assertTrue(routes['/health']()['ready'])
        result = routes['/classify'](types.SimpleNamespace(description='Verify emergency calls', include_evidence=False))
        self.assertEqual(result, {'type_id': 1})


class EvaluationChecks(unittest.TestCase):
    def test_per_class_counts_overrides_and_reproducibility(self):
        selected, counts = select_evaluation_rows(corpus(), LABELS, 20, {2: 7}, 2026)
        again, _ = select_evaluation_rows(corpus(), LABELS, 20, {2: 7}, 2026)
        self.assertEqual(selected, again)
        self.assertEqual(Counter(r['type_id'] for r in selected), {1: 20, 2: 7, 40: 20})
        self.assertEqual(len({r['row_key'] for r in selected}), len(selected))
        self.assertFalse(any(r['shortage'] for r in counts))

    def test_shortages_are_recorded_or_rejected_without_duplicates(self):
        data = [row('One holdout', 1)]
        selected, counts = select_evaluation_rows(data, {1: 'Calls'}, 200)
        self.assertEqual(len(selected), 1)
        self.assertEqual(counts[0]['used'], 1)
        self.assertTrue(counts[0]['shortage'])
        with self.assertRaises(ValueError):
            select_evaluation_rows(data, {1: 'Calls'}, 200, strict=True)
        with self.assertRaises(ValueError):
            select_evaluation_rows(data, {1: 'Calls'}, 0)
        with self.assertRaises(ValueError):
            select_evaluation_rows(data, {1: 'Calls'}, 1, {99: 2})

    def test_precision_recall_and_exported_confusion_matrix(self):
        actual, predicted = [1, 1, 2, 2, 40, 40], [1, 2, 2, -1, 40, 1]
        report = metrics(actual, predicted, LABELS)
        self.assertAlmostEqual(report['accuracy'], .5)
        self.assertAlmostEqual(report['macro_precision'], 2/3)
        self.assertAlmostEqual(report['macro_recall'], .5)
        self.assertEqual(sum(map(sum, report['confusion_matrix'])), 6)
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'run'
            export_metric_tables(prefix, report, LABELS)
            with Path(str(prefix)+'_summary.csv').open() as f:
                summary = list(csv.DictReader(f))
            self.assertAlmostEqual(float(summary[0]['accuracy']), .5)
            with Path(str(prefix)+'_per_class_metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertTrue({'precision', 'recall', 'f1', 'support'}.issubset(rows[0]))
            with Path(str(prefix)+'_confusion_matrix.csv').open() as f:
                matrix = list(csv.reader(f))
            self.assertEqual(sum(sum(map(int, row[1:])) for row in matrix[1:]), 6)

    def test_separate_model_test_accepts_custom_csv_and_exports_all_metrics(self):
        import io
        from contextlib import redirect_stdout
        data = [r for r in corpus() if r['ewoc_id'].endswith(('-0', '-1', '-2'))]
        actual_by_text = {r['description']: r['type_id'] for r in data}
        def predict(descriptions, evidence):
            output = []
            for description in descriptions:
                actual = actual_by_text[description]
                predicted = 2 if actual == 1 else actual
                output.append({'type_id': predicted, 'type_name': LABELS[predicted], 'raw': str(predicted),
                               'finish_reason': 'stop', 'prompt_stats': {'examples_used': 0}, 'evidence': []})
            return output
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_rows(root/'custom_test.csv', data)
            write_json(root/'labels.json', {str(k): {'name': v, 'definition': ''} for k,v in LABELS.items()})
            prefix = root/'results'/'test'
            fake = types.SimpleNamespace(classify_batch=predict)
            with patch(__name__+'.QwenClassifier', return_value=fake), \
                 patch(__name__+'.train_main', side_effect=AssertionError('Test must not train')), \
                 redirect_stdout(io.StringIO()) as output:
                test_main(['--eval-csv', str(root/'custom_test.csv'), '--labels', str(root/'labels.json'),
                           '--samples-per-class', '2', '--no-rag', '--out', str(prefix)])
            report = json.loads(Path(str(prefix)+'_metrics.json').read_text())
            self.assertEqual(report['rows'], 6)
            self.assertAlmostEqual(report['accuracy'], 2/3)
            self.assertEqual([r['used'] for r in report['sample_counts']], [2, 2, 2])
            for suffix in ['metrics.json', 'predictions.csv', 'errors.csv', 'evidence.jsonl',
                           'summary.csv', 'per_class_metrics.csv', 'confusion_matrix.csv', 'sample_counts.csv']:
                self.assertTrue(Path(str(prefix)+'_'+suffix).exists(), suffix)
            self.assertIn('precision', output.getvalue())
            self.assertIn('Confusion matrix', output.getvalue())

    def test_draft_definitions_are_marked_and_included_in_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d)/'labels.json'
            write_json(config, {'1': {'name': 'Calls', 'definition': 'Review call results',
                                     'definition_status': 'draft_needs_review'},
                                '2': {'name': 'Provisioning', 'definition': 'Approved fixture definition',
                                      'definition_status': 'approved'}})
            labels = load_labels(config)
            definitions = load_definitions(config, labels)
        self.assertTrue(definitions[1].startswith('[DRAFT: needs domain review]'))
        self.assertEqual(definitions[2], 'Approved fixture definition')
        messages = make_messages('Verify completed test calls', [], labels, definitions)
        for definition in definitions.values():
            self.assertIn(definition, messages[0]['content'])
        self.assertIn('provisional guidance', messages[0]['content'])

    def test_standalone_settings_control_sample_counts(self):
        import test_evaluate as separate
        with patch.object(separate, 'N_ROWS_PER_TYPE', 25), \
             patch.object(separate, 'ROWS_PER_TYPE', {1: 10}), \
             patch.object(separate, 'USE_RAG', False):
            args = separate.settings_arguments()
        self.assertEqual(args[args.index('--samples-per-class')+1], '25')
        self.assertEqual(json.loads(args[args.index('--samples-by-class')+1]), {'1': 10})
        self.assertIn('--no-rag', args)


class ConfigurationChecks(unittest.TestCase):
    def test_action_runs_without_flags_and_does_not_train_by_default(self):
        with patch(__name__+'.ACTION', 'test'), patch(__name__+'.test_main') as test, \
             patch(__name__+'.train_main', side_effect=AssertionError('Unexpected training')):
            main([])
        test.assert_called_once_with([])

    def test_config_changes_reach_parser_and_optional_flags_override(self):
        parser = argparse.ArgumentParser()
        add_inference_args(parser)
        with patch.dict(INFERENCE, {'backend': 'vllm', 'batch_size': 7}), \
             patch(__name__+'.USE_RAG', False):
            configured = parse_settings(parser, 'predict', [])
            overridden = parse_settings(parser, 'predict', ['--backend', 'transformers', '--rag'])
        self.assertEqual((configured.backend, configured.batch_size, configured.no_rag), ('vllm', 7, True))
        self.assertEqual((overridden.backend, overridden.no_rag), ('transformers', False))
        with patch.dict(INFERENCE, {'backend': 'typo'}), self.assertRaises(ValueError):
            parse_settings(parser, 'predict', [])

    def test_unlimited_sampling_and_empty_overrides_reach_standalone_runner(self):
        import test_evaluate as separate
        with patch.object(separate, 'N_ROWS_PER_TYPE', None), \
             patch.object(separate, 'ROWS_PER_TYPE', {}), \
             patch.object(separate, 'ADAPTER_PATH', None), \
             patch.object(separate, 'evaluate_main') as evaluate:
            separate.main([])
        arguments = evaluate.call_args.args[0]
        self.assertIn('--all-samples', arguments)
        self.assertEqual(arguments[arguments.index('--samples-by-class')+1], '{}')
        self.assertIsNone(evaluate.call_args.kwargs['settings']['adapter'])

    def test_constrained_labels_cover_multitoken_and_prefix_ids(self):
        sequences = [[10], [10, 20], [30, 40]]
        self.assertEqual(allowed_label_tokens(sequences, [], 99), [10, 30])
        self.assertEqual(allowed_label_tokens(sequences, [10], 99), [20, 99])
        self.assertEqual(allowed_label_tokens(sequences, [30], 99), [40])
        self.assertEqual(allowed_label_tokens(sequences, [10, 20], 99), [99])
        with self.assertRaises(ValueError):
            allowed_label_tokens(sequences, [88], 99)

    def test_transformers_backend_loads_local_files_and_generates_allowed_label(self):
        from contextlib import nullcontext
        class LocalTokenizer(TinyTokenizer):
            eos_token_id = 3
            pad_token_id = 3
            def decode(self, ids, skip_special_tokens=True):
                return bytes(int(v) for v in ids if int(v) != 3).decode()
        tokenizer = LocalTokenizer()
        def generate(**kwargs):
            values = kwargs['input_ids'][0].tolist()
            fn = kwargs['prefix_allowed_tokens_fn']
            self.assertIn(ord('4'), fn(0, np.asarray(values)))
            values.append(ord('4'))
            self.assertEqual(fn(0, np.asarray(values)), [ord('0')])
            values.append(ord('0'))
            self.assertEqual(fn(0, np.asarray(values)), [3])
            return np.asarray([values + [3]])
        model = types.SimpleNamespace(to=lambda device: None, eval=lambda: None, generate=generate)
        tokenizer_loader = unittest.mock.Mock(return_value=tokenizer)
        model_loader = unittest.mock.Mock(return_value=model)
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
            bfloat16='bfloat16', long='long', ones_like=np.ones_like,
            tensor=lambda values, **kw: np.asarray(values), inference_mode=nullcontext)
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_loader),
            GenerationConfig=lambda **kw: types.SimpleNamespace(**kw))
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
            args = argparse.Namespace(**dict(configured_defaults('predict'), model=d, adapter=None))
            classifier = QwenClassifier(args, labels=LABELS, definitions={})
            result = classifier.classify_batch(['BMC/WEA broadcast request'], [[]])[0]
        self.assertEqual(result['type_id'], 40)
        self.assertEqual(result['finish_reason'], 'constrained_label')
        for loader in [tokenizer_loader, model_loader]:
            self.assertTrue(loader.call_args.kwargs['local_files_only'])
            self.assertFalse(loader.call_args.kwargs['trust_remote_code'])
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['HF_HUB_DISABLE_TELEMETRY'], '1')

    def test_config_only_prepare_index_and_rag_evaluation(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            write_rows(root/'source.csv', corpus())
            write_json(root/'labels.json', {str(k): {'name': v, 'definition': ''} for k,v in LABELS.items()})
            with patch(__name__+'.LABEL_CONFIG', root/'labels.json'), \
                 patch.dict(PREPARE, {'csv': root/'source.csv', 'out': root/'data', 'folds': 5}), \
                 patch(__name__+'.TRAIN_CSV', root/'data/train.csv'), \
                 patch(__name__+'.INDEX_PATH', root/'index'), \
                 patch(__name__+'.USE_RAG', True), \
                 patch.dict(EMBEDDING, {'embedding_model': None}), \
                 patch.dict(TEST, {'eval_csv': root/'data/test.csv', 'out': root/'results/test',
                                   'max_per_class': 2, 'samples_by_class': {40: 1}}):
                prepare_main([])
                index_main([])
                fake = types.SimpleNamespace(classify_batch=lambda descriptions, evidence: [
                    {'type_id': 1, 'type_name': LABELS[1], 'raw': '1', 'finish_reason': 'stop',
                     'prompt_stats': {'examples_used': len(ex)}, 'evidence': ex} for ex in evidence])
                with patch(__name__+'.QwenClassifier', return_value=fake):
                    test_main([])
            report = json.loads((root/'results/test_metrics.json').read_text())
            self.assertEqual(report['rows'], 5)
            self.assertIsNotNone(report['index_fingerprint'])
            self.assertEqual([r['used'] for r in report['sample_counts']], [2, 2, 1])
            self.assertTrue((root/'results/test_confusion_matrix.csv').exists())


def check_main(argv=None):
    parser = argparse.ArgumentParser(description="Run all CPU integrity checks")
    parser.parse_args(argv)
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(Checks),
        unittest.defaultTestLoader.loadTestsFromTestCase(OptionalEmbeddings),
        unittest.defaultTestLoader.loadTestsFromTestCase(ConsolidatedChecks),
        unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationChecks),
        unittest.defaultTestLoader.loadTestsFromTestCase(ConfigurationChecks),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "prepare": prepare_main, "index": index_main,
        "predict": predict_main, "evaluate": evaluate_main,
        "test": test_main,
        "train": train_main, "select": select_main,
        "merge": merge_main, "serve": serve_main,
        "check": check_main,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(commands))
    if not argv:
        if ACTION not in commands:
            raise ValueError(f'Unknown ACTION: {ACTION}')
        print(f'ACTION={ACTION}; settings from the CONFIG section')
        return commands[ACTION]([])
    if argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    return commands[args.command](argv[1:])


if __name__ == "__main__":
    main()
