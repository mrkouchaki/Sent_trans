"""Evaluate base/LoRA Qwen with or without retrieval on a disjoint held-out CSV."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ewoc_common import (assert_disjoint, digest, load_labels, read_rows, write_json)
from ewoc_classifier import (QwenClassifier, add_inference_args, assert_adapter_holdout,
                             make_retriever, retrieve_many)


def metrics(actual, predicted, labels):
    all_ids = list(labels)
    report = classification_report(actual,predicted,labels=all_ids,output_dict=True,zero_division=0)
    matrix = confusion_matrix(actual,predicted,labels=all_ids+[-1])
    confusions = [{"actual_type_id":a,"predicted_type_id":b,"count":int(matrix[i,j])}
                  for i,a in enumerate(all_ids) for j,b in enumerate(all_ids+[-1])
                  if a != b and matrix[i,j]]
    confusions.sort(key=lambda r:-r["count"])
    return {"rows":len(actual),"accuracy":float(accuracy_score(actual,predicted)),
            "macro_f1":float(report["macro avg"]["f1-score"]),
            "weighted_f1":float(report["weighted avg"]["f1-score"]),
            "invalid_predictions":int(sum(p==-1 for p in predicted)),"classification_report":report,
            "confusion_labels":all_ids+[-1],"confusion_matrix":matrix.tolist(),"top_confusions":confusions[:30]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_inference_args(p)
    p.add_argument("--eval-csv", default="data/validation.csv")
    p.add_argument("--out", default="results/base_rag_validation")
    p.add_argument("--max-per-class", type=int, default=None, help="Optional fixed subset; final report should use full holdout")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    labels = load_labels(args.labels)
    rows = read_rows(args.eval_csv,labels)
    if any(r["is_synthetic"] for r in rows):
        raise ValueError("Validation/test must contain real tickets only")
    retriever = make_retriever(args,labels)
    if retriever:
        assert_disjoint(retriever.rows,rows,"retrieval references versus evaluation")
    # Check overlap before allocating a model/GPU, even for --no-rag.
    metadata_path = Path(args.adapter or args.model)/"ewoc_training_metadata.json"
    if args.adapter or metadata_path.exists():
        assert_adapter_holdout(json.loads(metadata_path.read_text()),rows)
    if args.max_per_class:
        if args.max_per_class < 1:
            raise ValueError("--max-per-class must be positive")
        rng = np.random.default_rng(args.seed)
        kept = []
        for label in labels:
            candidates = [r for r in rows if r["type_id"]==label]
            order = rng.permutation(len(candidates))[:args.max_per_class]
            kept.extend(candidates[i] for i in order)
        rows = kept
    evidence = retrieve_many(retriever,rows,args)
    classifier = QwenClassifier(args,labels)
    # Metadata verifies taxonomy; holdout has already been checked above.
    results = []
    for start in range(0,len(rows),args.batch_size):
        part = rows[start:start+args.batch_size]
        predictions = classifier.classify_batch([r["description"] for r in part],
                                                evidence[start:start+args.batch_size])
        for row,prediction in zip(part,predictions):
            results.append({"ewoc_id":row["ewoc_id"],"row_key":row["row_key"],
                            "description":row["description"],"actual_type_id":row["type_id"],
                            "actual_type_name":labels[row["type_id"]],
                            "predicted_type_id":prediction["type_id"] if prediction["type_id"] is not None else -1,
                            "predicted_type_name":prediction["type_name"],
                            "correct":prediction["type_id"]==row["type_id"],
                            "raw_response":prediction["raw"],"finish_reason":prediction["finish_reason"],
                            "examples_used":prediction["prompt_stats"]["examples_used"],
                            "evidence":prediction["evidence"]})
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    fields = [k for k in results[0] if k != "evidence"]
    for suffix,subset in [("predictions",results),("errors",[r for r in results if not r["correct"]])]:
        with Path(str(prefix)+f"_{suffix}.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
    with Path(str(prefix)+"_evidence.jsonl").open("w",encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"row_key":r["row_key"],"evidence":r["evidence"]},ensure_ascii=False)+"\n")
    report = metrics([r["actual_type_id"] for r in results], [r["predicted_type_id"] for r in results], labels)
    packages = {}
    for name in ["vllm","torch","transformers","sentence-transformers","peft","scikit-learn"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report.update({"configuration":vars(args),"labels":labels,"python":platform.python_version(),
                   "packages":packages,"evaluated_row_fingerprint":digest(sorted(r["row_key"] for r in rows)),
                   "index_fingerprint":retriever.data["fingerprint"] if retriever else None,
                   "retrieval_true_class_coverage":float(np.mean([
                       any(ex["type_id"]==r["actual_type_id"] for ex in r["evidence"]) for r in results]))
                       if retriever else None,
                   "note":"Retrieval scores and example agreement are not calibrated class probabilities."})
    write_json(str(prefix)+"_metrics.json",report)
    print(json.dumps({k:report[k] for k in ["rows","accuracy","macro_f1","weighted_f1","invalid_predictions"]},indent=2))
    print(f"Saved predictions, errors, evidence, metrics: {prefix}")


if __name__ == "__main__":
    main()
