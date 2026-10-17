"""Grouped preparation and validated, content-addressed split caching."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from ewoc_core import (audit, normalize, template_key, digest, assert_disjoint, file_hash,
                       load_labels, read_rows, write_rows, write_json)

PREPARATION_VERSION = 1


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    signature: str
    reused: bool

    @property
    def train_csv(self): return self.directory / 'train.csv'

    @property
    def validation_csv(self): return self.directory / 'validation.csv'

    @property
    def test_csv(self): return self.directory / 'test.csv'


def validate_partitions(parts, labels):
    """Check real grouped data, class coverage and pairwise holdout isolation."""
    for name, rows in parts.items():
        if not rows or not all(r.get('group_id') for r in rows):
            raise ValueError(f'{name} must contain nonempty grouped rows')
        if any(r['is_synthetic'] for r in rows) or audit(rows)['conflicting_descriptions']:
            raise ValueError(f'{name} contains synthetic tickets or conflicting labels')
        missing = set(labels) - {r['type_id'] for r in rows}
        if missing:
            raise ValueError(f'{name} lacks configured classes: {missing}')
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            assert_disjoint(parts[left], parts[right], f'{left} versus {right}')


class DataPreparer:
    """Prepare once per content/configuration, verify before every reuse.

    New data/settings get a new directory. An invalid cache is quarantined and
    rebuilt. Staging is published only after all split checks succeed.
    """
    artifacts = ('train.csv', 'validation.csv', 'test.csv', 'label_conflicts.csv',
                 'labels.json', 'split_audit.json')

    def __init__(self, source_csv, output_dir, labels_path, folds=10, seed=2026,
                 group_templates=False):
        self.source_csv = Path(source_csv).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.labels_path = Path(labels_path).expanduser().resolve()
        self.folds, self.seed, self.group_templates = folds, seed, group_templates

    def identity(self):
        if not self.source_csv.is_file():
            raise FileNotFoundError(f'Original CSV missing: {self.source_csv}. Set PREPARE["csv"] to your local labeled CSV.')
        return {'version': PREPARATION_VERSION, 'source_sha256': file_hash(self.source_csv),
                'labels': load_labels(self.labels_path), 'folds': self.folds,
                'seed': self.seed, 'group_templates': self.group_templates,
                'sklearn_version': sklearn.__version__}

    def _valid_cache(self, directory, signature):
        try:
            manifest = json.loads((directory/'preparation_manifest.json').read_text())
            if manifest['signature'] != signature:
                return False
            for name in self.artifacts:
                if file_hash(directory/name) != manifest['artifacts_sha256'][name]:
                    return False
            parts = {name: read_rows(directory/f'{name}.csv', load_labels(self.labels_path))
                     for name in ('train', 'validation', 'test')}
            validate_partitions(parts, load_labels(self.labels_path))
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def ensure(self, expected_signature=None):
        identity = self.identity()
        signature = digest(identity)
        if expected_signature is not None and signature != expected_signature:
            raise ValueError('Resume stopped: source CSV, labels or split settings changed. Restore the original inputs or start a new run.')
        cache = self.output_dir/'prepared'
        directory = cache/signature
        if self._valid_cache(directory, signature):
            print(f'Data: reusing verified splits ({signature[:12]})')
            return PreparedDataset(directory, signature, True)
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='preparing_', dir=cache))
        try:
            labels = load_labels(self.labels_path)
            rows = read_rows(self.source_csv, labels)
            if any(r['is_synthetic'] for r in rows):
                raise ValueError('Preparation requires real tickets; synthetic lineage is unverified.')
            missing = set(labels)-{r['type_id'] for r in rows}
            if missing:
                raise ValueError(f'Source CSV lacks configured classes: {missing}')
            parts, conflicts, report = prepare(rows, self.folds, self.seed, self.group_templates)
            validate_partitions(parts, labels)
            for name, part in parts.items():
                write_rows(staging/f'{name}.csv', part)
            write_rows(staging/'label_conflicts.csv', conflicts)
            write_json(staging/'labels.json', labels)
            report.update(source_csv=str(self.source_csv), labels=labels)
            write_json(staging/'split_audit.json', report)
            # Detect a source file edited while it was being read.
            if self.identity() != identity:
                raise ValueError('Source CSV or preparation settings changed during preparation; retry with stable inputs.')
            write_json(staging/'preparation_manifest.json', {
                'signature': signature, 'identity': identity,
                'artifacts_sha256': {name: file_hash(staging/name) for name in self.artifacts}})
            if not self._valid_cache(staging, signature):
                raise ValueError('Prepared split verification failed')
            if directory.exists():
                if self._valid_cache(directory, signature):
                    return PreparedDataset(directory, signature, True)
                directory.rename(cache/f'invalid_{signature[:12]}_{uuid.uuid4().hex}')
            try:
                staging.rename(directory)
            except OSError:
                if not self._valid_cache(directory, signature):
                    raise
            print(f'Data: prepared and verified splits ({signature[:12]})')
            return PreparedDataset(directory, signature, False)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

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

