#!/usr/bin/env python3
"""Audit the saved EWOC transformer evaluation-only run without retraining.

Save in the ewoc_ticket_type project root, then run in its existing venv:
    python ewoc_error_audit.py

Defaults match the screenshots: transformer artifacts, chronological 85/15
split, minimum training class count 75. Use artifacts from the SAME run.
Only local artifacts are read; Oracle and MLflow are not contacted.
Dependencies: the same pandas/numpy/scikit-learn/joblib/sentence-transformers
environment used for training. This is new diagnostic code, not a transcription.

Outputs are written to a new timestamped directory inside the artifact folder.
The confusion matrix has actual labels in rows and predictions in columns:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


LABEL = "WorkOrder_Info"
TEXT = "DESCRIPTION"
DATE = "CREATED_DATE"


def reconstruct_split(cleaned, summary, test_fraction, min_class_count):
    """Reproduce main_train.py's evaluation-only split and verify its sizes."""
    required = {LABEL, TEXT, DATE}
    if not required.issubset(cleaned.columns):
        raise ValueError(f"cleaned_data.csv is missing {sorted(required - set(cleaned.columns))}")
    cleaned = cleaned.copy()
    cleaned["cleaned_csv_row"] = np.arange(len(cleaned)) + 2
    cleaned[DATE] = pd.to_datetime(cleaned[DATE], errors="raise")
    if cleaned[DATE].isna().any():
        raise ValueError("Missing dates: these artifacts do not match the cleaned training data")
    ordered = cleaned.sort_values(DATE, kind="stable")
    boundary = int(len(ordered) * (1 - test_fraction))
    before_filter = ordered.iloc[:boundary].copy()
    test = ordered.iloc[boundary:].copy()
    counts = before_filter[LABEL].value_counts()
    train = before_filter[before_filter[LABEL].isin(counts[counts >= min_class_count].index)].copy()
    split = summary["split"]
    if split.get("validation_rows") != 0:
        raise ValueError("This script supports evaluation-only artifacts (validation_rows must be 0)")
    for key, actual in {
        "input_cleaned_rows": len(cleaned),
        "train_rows_before_smote": len(train),
        "test_rows": len(test),
    }.items():
        if split.get(key) != actual:
            raise ValueError(f"Split mismatch for {key}: saved={split.get(key)}, reconstructed={actual}. "
                             "Check test fraction, minimum class count, and artifact run.")
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty train/test split")
    return before_filter, train, test


def verify_predictions(test, predicted, known_labels, saved_metrics):
    correct = test[LABEL].to_numpy() == predicted
    unseen = ~test[LABEL].isin(known_labels)
    actual = {"evaluated_samples": len(test), "unseen_label_rows": int(unseen.sum())}
    for key, value in actual.items():
        if saved_metrics.get(key) != value:
            raise ValueError(f"Saved metrics mismatch: {key}={saved_metrics.get(key)}, recomputed={value}")
    accuracy = float(correct.mean())
    if not np.isclose(accuracy, saved_metrics["accuracy"], rtol=0, atol=1e-6):
        raise ValueError(f"Accuracy mismatch: saved={saved_metrics['accuracy']}, recomputed={accuracy}. "
                         "Do not interpret an audit of mixed-run artifacts.")
    return correct, unseen, accuracy


def write_reports(out, before_filter, train, test, predicted, probabilities, known_labels, saved_metrics):
    correct, unseen, accuracy = verify_predictions(test, predicted, known_labels, saved_metrics)
    confidence = probabilities.max(axis=1)
    predictions = test[["cleaned_csv_row", DATE, LABEL, TEXT]].copy().reset_index(drop=True)
    predictions = predictions.rename(columns={LABEL: "actual_label"})
    predictions["predicted_label"] = predicted
    predictions["confidence"] = confidence
    predictions["correct"] = correct
    predictions["unseen_label"] = unseen.to_numpy()
    predictions["description_seen_in_train"] = predictions[TEXT].isin(train[TEXT])

    train_label_sets = train.groupby(TEXT)[LABEL].agg(lambda values: set(values))
    predictions["labels_for_same_text_in_train"] = predictions[TEXT].map(
        train_label_sets.map(lambda labels: " | ".join(sorted(labels)))
    ).fillna("")
    predictions["same_text_has_different_train_label"] = [
        bool(train_label_sets.get(text, set()) - {actual})
        for text, actual in zip(predictions[TEXT], predictions["actual_label"])
    ]
    errors = predictions[~predictions.correct].copy()
    pairs = errors.groupby(["actual_label", "predicted_label"]).size().reset_index(name="errors")
    pairs = pairs.sort_values("errors", ascending=False, kind="stable")

    labels = sorted(set(known_labels) | set(test[LABEL]) | set(predicted))
    matrix = confusion_matrix(test[LABEL], predicted, labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.index.name = "actual_label"
    class_report = pd.DataFrame(classification_report(
        test[LABEL], predicted, labels=labels, output_dict=True, zero_division=0
    )).T

    counts = pd.DataFrame({
        "train_rows_before_class_filter": before_filter[LABEL].value_counts(),
        "train_rows_used": train[LABEL].value_counts(),
        "test_rows": test[LABEL].value_counts(),
        "predicted_test_rows": pd.Series(predicted).value_counts(),
    }).fillna(0).astype(int).reindex(labels, fill_value=0)
    counts["trained_label"] = counts.index.isin(known_labels)
    counts["test_errors"] = errors.actual_label.value_counts().reindex(counts.index, fill_value=0)
    counts["train_share"] = counts.train_rows_used / len(train)
    counts["test_share"] = counts.test_rows / len(test)
    counts.index.name = "label"

    text_stats = train.groupby(TEXT).agg(rows=(LABEL, "size"), label_count=(LABEL, "nunique"))
    conflicts = text_stats[text_stats.label_count > 1].copy()
    conflicts["label_counts"] = train.groupby(TEXT)[LABEL].agg(
        lambda values: json.dumps(values.value_counts().to_dict(), ensure_ascii=False)
    ).reindex(conflicts.index)
    conflicts = conflicts.sort_values("rows", ascending=False)

    # These are audit slices, not threshold-selection data.
    slices = []
    for name, mask in {
        "all": np.ones(len(predictions), dtype=bool),
        "known_label": ~predictions.unseen_label.to_numpy(),
        "unseen_label": predictions.unseen_label.to_numpy(),
        "description_seen_in_train": predictions.description_seen_in_train.to_numpy(),
        "description_new_to_train": ~predictions.description_seen_in_train.to_numpy(),
    }.items():
        slices.append({"slice": name, "rows": int(mask.sum()),
                       "accuracy": float(correct[mask].mean()) if mask.any() else None})

    focus = predictions.actual_label.str.contains(r"\bcran\b", case=False, regex=True) | \
        predictions.predicted_label.str.contains(r"\bcran\b", case=False, regex=True)
    focus_train = train[train[LABEL].str.contains(r"\bcran\b", case=False, regex=True)]
    # Inspect early and late examples, not a single arbitrary part of history.
    examples = pd.concat([focus_train.groupby(LABEL, group_keys=False).head(10),
                          focus_train.groupby(LABEL, group_keys=False).tail(10)])
    examples = examples.drop_duplicates("cleaned_csv_row")
    monthly = pd.concat([before_filter.assign(split="before_test"), test.assign(split="test")])
    monthly["month"] = monthly[DATE].dt.strftime("%Y-%m")
    monthly = monthly.groupby(["split", "month", LABEL]).size().reset_index(name="rows")

    out.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(out / "all_predictions.csv", index=False)
    errors.sort_values("confidence", ascending=False).to_csv(out / "misclassified_tickets.csv", index=False)
    pairs.to_csv(out / "top_confusions.csv", index=False)
    confusion.to_csv(out / "confusion_matrix.csv")
    class_report.to_csv(out / "class_metrics_all_labels.csv")
    counts.to_csv(out / "class_counts.csv")
    conflicts.to_csv(out / "conflicting_training_descriptions.csv")
    predictions[focus].to_csv(out / "cran_predictions.csv", index=False)
    examples[["cleaned_csv_row", DATE, LABEL, TEXT]].to_csv(out / "cran_training_examples.csv", index=False)
    monthly.to_csv(out / "monthly_label_counts.csv", index=False)
    pd.DataFrame(slices).to_csv(out / "evaluation_slices.csv", index=False)

    audit = {
        "accuracy": accuracy,
        "correct": int(correct.sum()),
        "errors": int((~correct).sum()),
        "unseen_label_rows": int(unseen.sum()),
        "conflicting_train_description_groups": len(conflicts),
        "test_rows_seen_in_train": int(predictions.description_seen_in_train.sum()),
        "test_rows_matching_different_train_label": int(predictions.same_text_has_different_train_label.sum()),
        "evaluation_slices": slices,
        "notes": [
            "Matches saved sample counts, label set, training row count, and accuracy; no row-level manifest was saved by the original trainer.",
            "Descriptions are the already-normalized text from cleaned_data.csv.",
            "Repeated text is measured; its presence alone does not prove label leakage.",
            "Different labels for the same normalized text can reflect ambiguity, legitimate context differences, or label errors.",
            "These test data are for diagnosis. Select future model settings on validation data and evaluate on a fresh final holdout.",
        ],
    }
    (out / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=Path("EWOCTypePredArtifacts/transformer"))
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-class-count", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or args.min_class_count < 2 or args.batch_size < 1:
        parser.error("Require 0 < test-fraction < 1, min-class-count >= 2, batch-size >= 1")
    root = args.artifacts.resolve()
    summary = json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
    if summary["metrics"].get("confidence_threshold") != 0.0:
        raise ValueError("Expected evaluation-only metrics with confidence_threshold=0")
    cleaned = pd.read_csv(root / "cleaned_data.csv", dtype={LABEL: str, TEXT: str}, keep_default_na=False)
    before_filter, train, test = reconstruct_split(cleaned, summary, args.test_fraction, args.min_class_count)
    model = joblib.load(root / "rf_model.pkl")  # Original filename also holds LogisticRegression.
    label_encoder = joblib.load(root / "label_encoder.pkl")
    known_labels = label_encoder.classes_.astype(str)
    if sorted(train[LABEL].unique()) != list(known_labels):
        raise ValueError("Saved label encoder does not match reconstructed training classes")
    if not np.array_equal(model.classes_, np.arange(len(known_labels))):
        raise ValueError("Saved classifier classes do not match the label encoder")
    encoder_path = root / "sentence_encoder"
    if not encoder_path.is_dir():
        raise FileNotFoundError(f"Missing saved transformer encoder: {encoder_path}")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(encoder_path), local_files_only=True)
    features = encoder.encode(test[TEXT].tolist(), batch_size=args.batch_size,
                              normalize_embeddings=True, show_progress_bar=True)
    predicted = label_encoder.inverse_transform(model.predict(features).astype(int))
    probabilities = model.predict_proba(features)
    out = root / ("error_audit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    audit, pairs = write_reports(out, before_filter, train, test, predicted, probabilities,
                                known_labels, summary["metrics"])
    print(f"Verified saved accuracy: {audit['accuracy']:.4%}; errors: {audit['errors']}")
    print("Most frequent errors (actual -> predicted):")
    print(pairs.head(10).to_string(index=False))
    print(f"Reports saved to: {out}")


if __name__ == "__main__":
    main()
