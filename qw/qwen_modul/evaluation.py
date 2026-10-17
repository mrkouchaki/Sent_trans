"""Model-quality evaluation, sampling, reports and metric exports."""
from collections import Counter, defaultdict
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import csv
import json
import platform
import random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ewoc_core import (load_labels, read_rows, make_retriever, assert_disjoint,
    assert_adapter_holdout, retrieve_many, QwenClassifier, write_json, digest)

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

def run_evaluation(args):
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
