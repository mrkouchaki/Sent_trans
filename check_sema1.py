"""Read saved head-refit results; no training, database access or model imports.

Save in src/ewoc_ttype, then run: python src/ewoc_ttype/check_semantic_run.py
An optional argument selects a particular head_refit_* folder.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def ambiguity_rate(frame, column):
    # Minimum disagreements on these observed rows, not a future accuracy limit.
    text = frame[column].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    counts = frame.assign(text_key=text).groupby(["text_key", "label"]).size().unstack(fill_value=0)
    return float((counts.sum(axis=1) - counts.max(axis=1)).sum() / len(frame))


def inspect(run):
    source = run.parent
    data = pd.read_csv(source / "dataset.csv", keep_default_na=False)
    predictions = pd.read_csv(run / "test_predictions.csv", keep_default_na=False)
    selection = json.loads((run / "selection_before_test.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(run / "comparison.csv")
    train, valid, test = [data[data.split.eq(s)] for s in ("train", "validation", "test")]
    development = pd.concat([train, valid])
    columns = ["DESCRIPTION", "label", "CREATED_DATE"]
    if not test[columns].reset_index(drop=True).equals(predictions[columns].reset_index(drop=True)):
        raise ValueError("Test predictions do not match this run's saved test rows.")
    correct = predictions.label.eq(predictions.predicted)
    row = comparison[comparison.model.eq(selection["winner"]) & comparison.split.eq("validation")].iloc[0]
    print(f"Run: {source.name}/{run.name}")
    print(f"Selected candidate: {selection['winner']} (before final head refit)")
    print(f"Accuracy / macro-F1: train {row.train_accuracy:.1%} / {row.train_macro_f1:.1%}; "
          f"validation {row.accuracy:.1%} / {row.macro_f1:.1%}")
    print(f"Validation: {valid.CREATED_DATE.min()} to {valid.CREATED_DATE.max()}")
    print(f"Test:       {test.CREATED_DATE.min()} to {test.CREATED_DATE.max()}")
    train_counts = train.label.value_counts()
    val_counts = valid.label.value_counts()
    print(f"Validation support: {(val_counts < 10).sum()}/{len(val_counts)} observed classes have <10 tickets; "
          f"{len(set(train.label) - set(valid.label))} training classes are absent.")
    raw = ambiguity_rate(development, "raw_description")
    normalized = ambiguity_rate(development, "DESCRIPTION")
    print(f"Minimum identical-text label disagreements in development: raw {raw:.1%}; normalized {normalized:.1%}")
    print("  These are snapshot statistics; raw IDs can conceal repeated wording.")
    fitted = development if selection["head_refit_on_train_plus_validation"] else train
    seen = predictions.DESCRIPTION.isin(fitted.DESCRIPTION)
    for title, mask in (("Previously seen normalized text", seen), ("New normalized text", ~seen)):
        value = f"{correct[mask].mean():.1%}" if mask.any() else "n/a"
        print(f"{title}: test accuracy {value} ({int(mask.sum())} tickets)")
    errors = predictions[~correct]
    largest = errors.label.value_counts().head(3)
    share = largest.sum() / len(errors) if len(errors) else 0
    print(f"Largest actual-class error sources ({share:.1%} of all {len(errors)} errors):")
    for label, count in largest.items():
        support = int(predictions.label.eq(label).sum())
        wrong = errors[errors.label.eq(label)].predicted.value_counts()
        unique = fitted[fitted.label.eq(label)].DESCRIPTION.nunique()
        print(f"  {label}: {count}/{support} wrong; {unique} distinct fitted texts; "
              f"train/validation tickets {train_counts.get(label, 0)}/{val_counts.get(label, 0)}")
        print(f"    Fitted/test share {fitted.label.eq(label).mean():.1%}/{support / len(predictions):.1%}; "
              f"most confused with {wrong.index[0]} ({int(wrong.iloc[0])})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="Existing head_refit_* folder")
    args = parser.parse_args()
    if args.run is None:
        root = Path(__file__).resolve().parents[2] / "EWOCTypePredArtifacts"
        completed = list(root.glob("human_tickets_*/head_refit_*/comparison.csv"))
        if not completed:
            raise SystemExit("Supply the head_refit_* folder printed by refit_semantic_head.py.")
        args.run = max(completed, key=lambda path: path.parent.name).parent
    inspect(args.run)


if __name__ == "__main__":
    main()
