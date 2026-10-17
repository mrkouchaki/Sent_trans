"""Audit CSV and create reproducible grouped train/validation/test splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ewoc_common import (DEFAULT_LABELS, assert_disjoint, audit, digest, load_labels,
                         normalize, read_rows, template_key, write_json, write_rows)


def prepare(rows, folds=10, seed=2026, group_templates=False):
    if folds < 3:
        raise ValueError("--folds must be >=3")
    original_audit = audit(rows)
    # Do not silently learn contradictory labels for the exact same input.
    text_labels = defaultdict(set)
    for row in rows:
        text_labels[normalize(row["description"])].add(row["type_id"])
    conflicts = [r for r in rows if len(text_labels[normalize(r["description"])]) > 1]
    rows = [dict(r) for r in rows if len(text_labels[normalize(r["description"])]) == 1]
    # Remove exact content duplicates within a label, retaining one representative.
    unique = {}
    for row in rows:
        unique.setdefault((normalize(row["description"]), row["type_id"]), row)
    deduped = list(unique.values())
    # Build union-find over ALL non-conflicting rows so deduplication cannot erase
    # a connecting EWOC_ID edge (same text appears under two ticket IDs).
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [("text", normalize(row["description"]))]
        if row["ewoc_id"]:
            keys.append(("id", row["ewoc_id"]))
        if group_templates:
            keys.append(("template", template_key(row["description"])))
        if row.get("group_id"):
            keys.append(("supplied", row["group_id"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    text_group = {normalize(r["description"]): find(i) for i, r in enumerate(rows)}
    members = defaultdict(list)
    for r in deduped:
        members[text_group[normalize(r["description"])]].append(r["row_key"])
    group_ids = {k: digest(sorted(v))[:24] for k, v in members.items()}
    for r in deduped:
        r["group_id"] = group_ids[text_group[normalize(r["description"])]]
    groups_by_class = defaultdict(set)
    for r in deduped:
        groups_by_class[r["type_id"]].add(r["group_id"])
    insufficient = {k: len(v) for k, v in groups_by_class.items() if len(v) < folds}
    if insufficient:
        raise ValueError(f"Need >= {folds} groups per class; found {insufficient}. "
                         "Reduce --folds (minimum 3), add real examples, or remove unsupported classes.")
    if not deduped:
        raise ValueError("No unambiguous descriptions remain")
    y = np.array([r["type_id"] for r in deduped])
    g = np.array([r["group_id"] for r in deduped])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.full(len(deduped), -1)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(y)), y, g)):
        assignment[indices] = fold
    parts = {"test": [r for i,r in enumerate(deduped) if assignment[i] == 0],
             "validation": [r for i,r in enumerate(deduped) if assignment[i] == 1],
             "train": [r for i,r in enumerate(deduped) if assignment[i] >= 2]}
    all_classes = set(y.tolist())
    for name, part in parts.items():
        missing = all_classes - {r["type_id"] for r in part}
        if missing:
            raise ValueError(f"{name} lacks classes {missing}; use fewer folds or more independent groups.")
    assert_disjoint(parts["train"], parts["validation"])
    assert_disjoint(parts["train"], parts["test"])
    assert_disjoint(parts["validation"], parts["test"])
    report = {"source": original_audit, "exact_duplicate_rows_removed": len(rows)-len(deduped),
              "conflicting_rows_quarantined": len(conflicts), "folds": folds, "seed": seed,
              "group_templates": group_templates, "split_counts": {k:audit(v) for k,v in parts.items()},
              "largest_groups": Counter(r["group_id"] for r in deduped).most_common(10),
              "limitations": "Exact text and EWOC_ID isolation guaranteed. Optional numeric-template grouping "
                             "is heuristic, not general semantic deduplication. No time split: temporal drift unmeasured."}
    return parts, conflicts, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="whole_ewoc_tickets.csv")
    p.add_argument("--out", default="data")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--folds", type=int, default=10, help="Two folds held out; default approximately 80/10/10")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--group-templates", action="store_true")
    args = p.parse_args()
    labels = load_labels(args.labels)
    rows = read_rows(args.csv, labels)
    if any(r["is_synthetic"] for r in rows):
        raise ValueError("Use real tickets for this split. Add synthetic rows to training separately only "
                         "after tracing their source groups; never put synthetic tickets in validation/test.")
    missing = set(labels) - {r["type_id"] for r in rows}
    if missing:
        raise ValueError(f"Configured labels with no real rows: {missing}. Fix labels.json or source data.")
    parts, conflicts, report = prepare(rows, args.folds, args.seed, args.group_templates)
    out = Path(args.out)
    for name, part in parts.items():
        write_rows(out/f"{name}.csv", part)
    write_rows(out/"label_conflicts.csv", conflicts)
    write_json(out/"labels.json", labels)
    report["source_csv"] = str(Path(args.csv).resolve())
    report["labels"] = labels
    write_json(out/"split_audit.json", report)
    print({k:len(v) for k,v in parts.items()})
    print(f"Audit and conflict review: {out.resolve()}")


if __name__ == "__main__":
    main()
