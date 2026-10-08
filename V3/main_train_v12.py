#!/usr/bin/env python3
"""EWOC comparison: three fixed models, one 12-month training window.

Run from ewoc_ticket_type: python src/ewoc_ttype/main_train.py
Input: latest version from clean_ewoc_data.py, or --data <cleaned.csv>.
The newest 56 calendar days form the test; the preceding 56 days validate.
--reuse-splits <folder> preserves an existing snapshot instead.
Each model fits once for validation, then once on the 12 months before the test.
No parameter, training-window or balancing sweep. Completed fits are checkpointed.
MiniLM and BERT jointly fine-tune all layers with capped square-root weights for three epochs.
MiniLM ranks whole clauses with training TF-IDF only when a description exceeds 256 tokens.
Transformers downloads google-bert/bert-base-uncased automatically on first use.
"""

import argparse
import hashlib
import html
import importlib.util
import json
import logging
import shutil
import sys
import warnings
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATH = ROOT / "models/sentence-transformers/all-MiniLM-L6-v2"
BERT_MODEL_ID = "google-bert/bert-base-uncased"
BERT_CACHE = ROOT / "models/huggingface_cache"
BERT_REVISION = "main"
SEED, THREADS, TARGET = 42, 4, 0.85
TRAIN_MONTHS, VALIDATION_DAYS, TEST_DAYS = 12, 56, 56
METRICS = ["accuracy", "macro_f1", "weighted_f1"]
FEATURES = {"wordchar_svm": "wordchar", "tfidf_minilm_ft": "text", "bert_ft": "text"}
PARAMETERS = {
    "wordchar_svm": {"C": 2.0},
    "tfidf_minilm_ft": {"epochs": 3, "learning_rate": 2e-5, "head_learning_rate": 1e-3},
    "bert_ft": {"epochs": 3, "learning_rate": 2e-5},
}
logger = logging.getLogger("ewoc.benchmark")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


@contextmanager
def activity(message, announce=True):
    started, stop = perf_counter(), Event()
    def heartbeat():
        while not stop.wait(30):
            logger.info("%s: running (%.0fs)", message, perf_counter() - started)
    if announce:
        logger.info("%s...", message)
    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def installed_versions():
    result = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "joblib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def load_snapshot(source):
    frame = pd.read_csv(source / "dataset.csv", keep_default_na=False, low_memory=False,
                        dtype={"label": str, "DESCRIPTION": str})
    required = {"DESCRIPTION", "label", "CREATED_DATE", "split"}
    if not required <= set(frame):
        raise ValueError(f"dataset.csv needs {sorted(required)}")
    frame["CREATED_DATE"] = pd.to_datetime(frame.CREATED_DATE, format="mixed", errors="raise")
    if frame.CREATED_DATE.isna().any() or frame.label.str.strip().eq("").any():
        raise ValueError("Every ticket needs a valid date and label. Empty descriptions remain in evaluation.")
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Snapshot must contain train, validation and test splits.")
    development, test = frame[frame.split.ne("test")].copy(), frame[frame.split.eq("test")].copy()
    if development.CREATED_DATE.max() >= test.CREATED_DATE.min():
        raise ValueError("Every development ticket must precede the existing test period.")
    return frame, development, test


def prepare_cleaned_snapshot(csv_path, artifacts):
    """Use corrected TYPE_ID as the target; never stale label names or creation metadata as features."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"TYPE_ID", "DESCRIPTION", "CREATED_DATE", "CREATED_ID", "ORIGINAL_TYPE_ID", "CLEANUP_RULE"}
    if not required <= set(frame):
        raise ValueError("Run clean_ewoc_data.py on the original whole_ewoc_tickets.csv first.")
    dates = pd.to_datetime(frame.CREATED_DATE, format="mixed", errors="coerce", utc=True).dt.tz_localize(None)
    if dates.isna().any():
        raise ValueError(f"Invalid CREATED_DATE in {int(dates.isna().sum())} rows; fix the source dates before splitting.")
    if "EWOC_ID" in frame and (frame.EWOC_ID.str.strip().eq("").any() or frame.EWOC_ID.duplicated().any()):
        raise ValueError("EWOC_ID must be nonempty and unique; check for duplicate database join rows.")
    types = pd.to_numeric(frame.TYPE_ID, errors="raise")
    if types.isna().any() or types.mod(1).ne(0).any():
        raise ValueError("TYPE_ID must be an integer.")
    frame["label"] = types.astype("int64").astype(str)
    frame["CREATED_DATE"] = dates
    test_start = dates.max().normalize() + pd.Timedelta(days=1 - TEST_DAYS)
    validation_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    frame["split"] = np.where(dates.ge(test_start), "test", np.where(dates.ge(validation_start), "validation", "train"))
    key = identifier({"source": file_hash(csv_path), "test_days": TEST_DAYS,
                      "validation_days": VALIDATION_DAYS, "schema": 2})
    source = artifacts / ("cleaned_snapshot_" + key)
    source.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source / "dataset.csv", index=False)
    write_json(source / "type_name_to_id.json", {label: int(label) for label in sorted(frame.label.unique())})
    summary = csv_path.parent / "summary.json"
    if summary.exists():
        saved = read_json(summary)
        if saved.get("cleaned_sha256") != file_hash(csv_path):
            raise ValueError("Cleaned CSV was modified after its audit was saved. Rerun cleanup from the source.")
        shutil.copy2(summary, source / "cleanup_summary.json")
    write_json(source / "source.json", {"csv": str(csv_path.resolve()), "test_start": str(test_start),
        "target": "corrected TYPE_ID", "features": "DESCRIPTION only; no cleanup flags, dates, creator or type IDs"})
    load_snapshot(source)  # Fail early for missing/overlapping periods.
    return source


def chronological_folds(development, test_start):
    start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    valid = development[development.CREATED_DATE.ge(start) & development.CREATED_DATE.lt(test_start)]
    if valid.empty:
        raise ValueError(f"No tickets in validation period {start} to {test_start}.")
    return [("validation", start, valid.copy())]


def window_rows(development, cutoff, forbidden_texts=()):
    part = development[development.CREATED_DATE.lt(cutoff) &
                       development.CREATED_DATE.ge(cutoff - pd.DateOffset(months=TRAIN_MONTHS))]
    # Purge exact validation-text duplicates from training; never discard validation rows.
    return part[~part.DESCRIPTION.isin(forbidden_texts)].copy()


def metric_values(actual, predicted):
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0))}


def score_floor(scores):
    return min(scores[key] for key in METRICS)


def meets_target(scores):
    return all(scores.get(key) is not None and scores[key] > TARGET for key in METRICS)


def fit_weights(part, balancing):
    weights = compute_sample_weight("balanced", part.label)
    if balancing == "sqrt_balanced":
        weights = np.minimum(5.0, np.sqrt(weights))
    return weights / weights.mean()


def classifier(spec):
    family, params = spec["family"], spec["params"]
    if FEATURES[family] == "text":
        if __package__ in {None, ""}:
            from bert_classifier import BertClassifier
        else:
            from .bert_classifier import BertClassifier
        minilm = family == "tfidf_minilm_ft"
        return BertClassifier(ENCODER_PATH if minilm else BERT_MODEL_ID, **params,
                              seed=SEED, threads=THREADS, clause_tfidf=minilm,
                              cache_dir=BERT_CACHE, revision="main" if minilm else BERT_REVISION)
    if family == "wordchar_svm":
        return LinearSVC(**params, dual=True, max_iter=10000, tol=1e-3, random_state=SEED)
    raise ValueError(f"Unknown model: {family}")


def encoder_fingerprint(path):
    assets = sorted(p for p in path.iterdir() if p.suffix in {".json", ".txt", ".bin", ".safetensors"})
    if not any(p.suffix in {".bin", ".safetensors"} for p in assets):
        raise FileNotFoundError(f"No local MiniLM weights in {path}")
    return identifier({asset.name: file_hash(asset) for asset in assets})


class FeatureBank:
    """Fit vocabulary/IDF only on this training window; reuse across its classifiers."""

    def __init__(self, train, valid):
        self.train, self.valid = train, valid
        self.data = {}
        self.transforms = {}

    def get(self, name):
        if name not in self.data:
            if name == "word":
                vocab = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                        sublinear_tf=True, dtype=np.float32)
                self.data[name] = (vocab.fit_transform(self.train.DESCRIPTION), vocab.transform(self.valid.DESCRIPTION))
                self.transforms[name] = vocab
            elif name == "wordchar":
                word = self.get("word")
                vocab = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                        max_features=50000, sublinear_tf=True, dtype=np.float32)
                char = (vocab.fit_transform(self.train.DESCRIPTION), vocab.transform(self.valid.DESCRIPTION))
                self.transforms["char"] = vocab
                self.data[name] = tuple(normalize(sparse.hstack([a, b], format="csr")) for a, b in zip(word, char))
        return self.data[name]

    def training(self, spec):
        name = FEATURES[spec["family"]]
        x_train, x_valid = self.get(name)
        y_train = self.train.label.to_numpy()
        weights = fit_weights(self.train, spec["balancing"])
        return x_train, y_train, weights, x_valid


def fit_one(spec, bank, checkpoint_dir=None):
    if FEATURES[spec["family"]] == "text":
        model = classifier(spec)
        weights = fit_weights(bank.train, spec["balancing"])
        model.fit(bank.train.DESCRIPTION, bank.train.label, sample_weight=weights, checkpoint_dir=checkpoint_dir)
        predicted = model.predict(bank.valid.DESCRIPTION)
        return model, predicted, {"converged": True, "warnings": [], "fit_rows": len(bank.train),
            "training_note": "Completed fixed supervised epochs; not a convergence proof.",
            "training_text_counts": model.training_text_counts,
            "unseen_label_rows": int((~bank.valid.label.isin(model.classes_)).sum())}
    x_train, y_train, weights, x_valid = bank.training(spec)
    model = classifier(spec)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=THREADS):
        warnings.simplefilter("always")
        model.fit(x_train, y_train, sample_weight=weights)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, model.predict(x_valid), {
        "converged": converged, "warnings": [str(w.message) for w in caught],
        "fit_rows": len(bank.train),
        "unseen_label_rows": int((~bank.valid.label.isin(model.classes_)).sum())}


class Search:
    def __init__(self, development, folds, output):
        self.development, self.folds, self.output = development, folds, output
        self.banks, self.records = {}, {}

    def bank(self, fold):
        key = fold[0]
        if key not in self.banks:
            _, cutoff, valid = fold
            train = window_rows(self.development, cutoff, valid.DESCRIPTION)
            if len(train) < 10 or train.label.nunique() < 2:
                raise ValueError("Insufficient training tickets/classes in the 12-month window after text deduplication.")
            self.banks[key] = FeatureBank(train, valid)
        return self.banks[key]

    def run(self, specs, stage):
        logger.info("%s: %d fixed models; one validation period; no tuning sweep", stage, len(specs))
        for number, spec in enumerate(specs, 1):
            key = identifier(spec)
            destination = self.output / "trials" / key
            destination.mkdir(parents=True, exist_ok=True)
            done = destination / "result.json"
            if done.exists():
                self.records[key] = read_json(done)
                logger.info("%s %d/%d %s: using completed fit", stage, number, len(specs), spec['family'])
                continue
            if FEATURES[spec["family"]] == "text":
                self.banks.clear()  # Release sparse feature matrices before full BERT training.
            started, parts, scores = perf_counter(), [], []
            with activity(f"{stage} {number}/{len(specs)} {spec['family']}"):
                for fold in self.folds:
                    name, cutoff, valid = fold
                    metadata, predictions = destination / f"{name}.json", destination / f"{name}.npz"
                    if not metadata.exists():
                        try:
                            bank = self.bank(fold)
                            checkpoint = destination / (name + "_training")
                            if FEATURES[spec["family"]] == "text":
                                model, predicted, info = fit_one(spec, bank, checkpoint)
                                model.close()
                            else:
                                _, predicted, info = fit_one(spec, bank)
                            np.savez_compressed(predictions, row_id=valid.index.to_numpy(), predicted=np.asarray(predicted, dtype=str))
                            write_json(metadata, {**metric_values(valid.label, predicted), **info, "status": "ok"})
                            if checkpoint.exists():
                                shutil.rmtree(checkpoint)
                        except (ValueError, RuntimeError, MemoryError, OSError, ImportError) as error:
                            write_json(metadata, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
                    info = read_json(metadata)
                    scores.append(info)
                    if info["status"] == "ok":
                        with np.load(predictions, allow_pickle=False) as saved:
                            if not np.array_equal(saved["row_id"], valid.index.to_numpy()):
                                raise ValueError("Checkpoint validation rows changed; use a new output folder.")
                            parts.append(saved["predicted"].copy())
                record = {"id": key, **spec, "folds": scores, "seconds": perf_counter() - started}
                if len(parts) == len(self.folds):
                    actual = pd.concat([fold[2] for fold in self.folds]).label
                    record.update(metric_values(actual, np.concatenate(parts)), status="ok",
                                  eligible=all(row["converged"] for row in scores))
                else:
                    record.update(status="failed", eligible=False)
                write_json(done, record)
                self.records[key] = record
            if record["status"] == "ok":
                logger.info("%s | validation accuracy %.1f%% | macro-F1 %.1f%% | weighted-F1 %.1f%% | eligible=%s",
                            spec['family'], *(100 * record[key] for key in METRICS), record["eligible"])
            else:
                logger.warning("%s failed: %s", spec['family'], scores[-1].get("error", "see trial results"))
        return self.ranked()

    def ranked(self):
        return sorted([r for r in self.records.values() if r["eligible"]],
                      key=lambda r: (score_floor(r), r["accuracy"]), reverse=True)


def candidate_specs():
    return [{"family": family, "window": f"last_{TRAIN_MONTHS}m", "params": PARAMETERS[family],
             "balancing": "sqrt_balanced" if FEATURES[family] == "text" else "balanced"}
            for family in FEATURES]


def flatten_trial(record):
    return {key: value for key, value in record.items() if key not in {"folds", "params"}} | {
        "params": json.dumps(record["params"]),
        "fold_details": json.dumps(record["folds"])}


def save_evaluation(destination, frame, predicted):
    frame.assign(predicted=predicted, correct=predicted == frame.label.to_numpy()).to_csv(
        destination / "test_predictions.csv", index=False)
    report = classification_report(frame.label, predicted, output_dict=True, zero_division=0)
    labels = sorted(set(frame.label) | set(predicted))
    pd.DataFrame({label: report[label] for label in labels}).T.to_csv(destination / "test_class_metrics.csv", index_label="label")
    groups = {"all": np.ones(len(frame), dtype=bool)}
    if "CREATED_ID" in frame:
        creator = pd.to_numeric(frame.CREATED_ID, errors="coerce")
        groups.update(human=creator.notna() & creator.ne(1), superuser=creator.eq(1), unknown_creator=creator.isna())
    if {"TYPE_ID", "ORIGINAL_TYPE_ID"} <= set(frame):
        changed = pd.to_numeric(frame.TYPE_ID).ne(pd.to_numeric(frame.ORIGINAL_TYPE_ID))
        groups.update(rule_corrected=changed, original_label=~changed)
    rows = [{"slice": name, "rows": int(np.sum(mask)), **metric_values(frame.loc[mask, "label"], np.asarray(predicted)[mask])}
            for name, mask in groups.items() if np.any(mask)]
    pd.DataFrame(rows).to_csv(destination / "test_slices.csv", index=False)


def final_evaluation(development, test, output, finalists):
    # This function is called only after the validation selection has been written.
    if not (output / "selection.json").exists():
        raise ValueError("Lock validation selection before evaluating the test.")
    results = []
    for selected in finalists:
        spec = {key: selected[key] for key in ("family", "window", "balancing", "params")}
        destination = output / "finalists" / spec["family"]
        destination.mkdir(parents=True, exist_ok=True)
        done = destination / "result.json"
        if done.exists():
            saved = read_json(done)
            if saved["id"] != selected["id"]:
                raise ValueError("Finalist selection changed inside a checkpoint folder.")
            results.append(saved)
            continue
        cutoff = test.CREATED_DATE.min()
        train = window_rows(development, cutoff)
        started = perf_counter()
        result = {"id": selected["id"], **spec, "validation": {key: selected[key] for key in METRICS},
                  "passed_validation": meets_target(selected), "expected_test_rows": len(test)}
        try:
            with activity(f"Final fit {spec['family']}/{spec['window']} ({len(train)} real tickets)"):
                bank = FeatureBank(train, test)
                checkpoint = destination / "transformer_training"
                neural = FEATURES[spec["family"]] == "text"
                model_directory = "minilm" if spec["family"] == "tfidf_minilm_ft" else "bert"
                if neural:
                    model, predicted, info = fit_one(spec, bank, checkpoint)
                    model.save(destination / model_directory)
                    model.close()
                else:
                    model, predicted, info = fit_one(spec, bank)
                    joblib.dump(model, destination / "classifier.pkl")
                for name, transform in bank.transforms.items():
                    joblib.dump(transform, destination / f"{name}_tfidf.pkl")
                write_json(destination / "model_config.json", {**spec, "feature_kind": FEATURES[spec["family"]],
                    "classes": model.classes_.tolist(), "transformer_model": model_directory if neural else None,
                    "pooling": "model_native_sequence_classification" if neural else "word_char_tfidf_l2",
                    "max_length": 256 if neural else None, "input": "DESCRIPTION only; source CSV wording unchanged",
                    "clause_tfidf": spec["family"] == "tfidf_minilm_ft",
                    "bert_model": "bert" if spec["family"] == "bert_ft" else None,
                    "bert_source": BERT_MODEL_ID if spec["family"] == "bert_ft" else None,
                    "bert_revision": BERT_REVISION if spec["family"] == "bert_ft" else None,
                    "bert_truncation": "first 127 + last 127 tokens; CLS/SEP added" if spec["family"] == "bert_ft" else None,
                    "composition": "wordchar=L2(hstack(word,char)); MiniLM=TF-IDF clause selection + supervised encoder/head"})
                save_evaluation(destination, test, predicted)
            scores = metric_values(test.label, predicted)
            result.update(scores, rows=len(test), **info, status="ok", passed_test=meets_target(scores),
                          passed_all=bool(meets_target(selected) and meets_target(scores) and info["converged"]))
        except (ValueError, RuntimeError, MemoryError, OSError, ImportError) as error:
            result.update({key: None for key in METRICS}, rows=0, unseen_label_rows=None, status="failed",
                          passed_test=False, passed_all=False, error=f"{type(error).__name__}: {error}")
            logger.warning("Final fit failed for %s; remaining finalists will continue", spec["family"])
        result["seconds"] = perf_counter() - started
        write_json(done, result)
        checkpoint = destination / "transformer_training"
        if result["status"] == "ok" and checkpoint.exists():
            shutil.rmtree(checkpoint)
        results.append(result)
    return results


def render_report(output, selection, results, ranked, skipped):
    rows = [{"model": row["family"], "window": row["window"], "balance": row["balancing"],
             **{"val_" + key: row["validation"][key] for key in METRICS},
             **{"test_" + key: row[key] for key in METRICS},
             "unseen_test": row["unseen_label_rows"], "passes_all": row["passed_all"]} for row in results]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    chosen = next(row for row in results if row["id"] == selection["winner_id"])
    passers = [row["family"] for row in results if row["passed_all"]]
    verdict = ("PASS: selected model meets observed validation and test targets" if chosen["passed_all"] else
               "SELECTED MODEL FAILS: other finalists meet observed targets" if passers else
               "NO PASS: none of the finalists meets all targets")
    status = {"verdict": verdict, "winner": chosen["family"], "winner_id": chosen["id"],
              "target": TARGET, "required_metrics": METRICS, "strictly_greater_than_target": True,
              "all_test_rows_evaluated": all(row["rows"] == read_json(output / "splits.json")["test_rows"] for row in results),
              "test_selection": False, "search_complete": not skipped,
              "observed_passers": passers,
              "skipped": skipped, "fresh_independent_test_still_required": True}
    write_json(output / "requirements.json", status)
    display = comparison.copy()
    for column in [c for c in display if c.startswith(("val_", "test_"))]:
        display[column] = display[column].map(lambda value: "not evaluated" if pd.isna(value) else f"{value:.1%}")
    short = display.rename(columns={"test_accuracy": "accuracy", "test_macro_f1": "macro_F1",
                                    "test_weighted_f1": "weighted_F1"})
    print("\n" + short[["model", "window", "balance", "accuracy", "macro_F1", "weighted_F1", "passes_all"]].to_string(index=False))
    print(f"\n{verdict}. Validation-selected winner: {chosen['family']} / {chosen['window']}.")
    if passers:
        print("Finalists meeting observed targets: " + ", ".join(passers))
    print(f"Full comparison: {output / 'comparison.html'}", flush=True)
    table = pd.DataFrame([flatten_trial(row) for row in ranked])
    table = table[["family", "window", "balancing", *METRICS]]
    for column in METRICS:
        table[column] = table[column].map("{:.1%}".format)
    skipped_text = "None" if not skipped else "; ".join(skipped)
    slices = []
    for row in results:
        path = output / "finalists" / row["family"] / "test_slices.csv"
        if path.exists():
            slices.append(pd.read_csv(path).assign(model=row["family"]))
    slice_table = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()
    slice_table.to_csv(output / "test_slices.csv", index=False)
    for column in METRICS:
        if column in slice_table:
            slice_table[column] = slice_table[column].map("{:.1%}".format)
    document = f"""<!doctype html><html><meta charset="utf-8"><title>EWOC model comparison</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:35px auto;padding:0 20px;color:#192333}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8dee8;padding:8px;text-align:left}}
th{{background:#edf2f8}}h1{{margin-bottom:8px}}.result{{padding:16px;background:#edf2f8}}p{{line-height:1.5}}</style>
<h1>EWOC: three models, one training window</h1><p class="result"><strong>{html.escape(verdict)}</strong><br>
Validation-selected winner: {html.escape(chosen['family'])} / {html.escape(chosen['window'])}.</p>
<p>PASS requires accuracy, macro-F1 and weighted-F1 each strictly above 85% on chronological validation
and the unchanged test set. All test rows count; unsupported classes count as errors. No confidence filtering,
label merging or test-based model selection. F1 and class reports use the same true/predicted-label union.</p>
<h2>Model comparison</h2>{display.to_html(index=False, escape=True)}
<p>Each model uses the {TRAIN_MONTHS} months preceding the validation cutoff. Validation covers the next
{VALIDATION_DAYS} days. Final fitting uses the {TRAIN_MONTHS} months preceding the test cutoff, including validation
tickets. The winner was locked before test evaluation. New cleaned inputs reserve the newest {TEST_DAYS} days for test;
--reuse-splits preserves the supplied test period instead.
This test has been inspected in earlier development; an observed PASS is not independent production certification.</p>
<p>For cleaned inputs, TYPE_ID is the target. Label rules use the description, so these scores measure agreement
with the corrected business rules; they do not independently prove those rules are correct. Check the human-only
slice when assessing performance on human-written requests. Combined PASS does not imply every slice passes.</p>
<details><summary>Human, superuser and corrected-label test results</summary>{slice_table.to_html(index=False, escape=True)}</details>
<h2>Validation results</h2>{table.to_html(index=False, escape=True)}
<p>Three fixed configurations, one validation fit per model, then one final fit per successful model.
No parameter search, window search or balancing sweep. Linear SVM uses C=2 and balanced sample weights.</p>
<p>TF-IDF + MiniLM fine-tunes the local MiniLM encoder and classification head for three epochs. The encoder
learning rate is 2e-5; the classifier/pooler rate is 1e-3. TF-IDF learns word/phrase statistics from training
descriptions only. Descriptions that fit 256 tokens are kept intact. Longer descriptions use ranked whole clauses,
in original order with separators; the opening context is retained when short enough. An oversized single clause
or all-unknown wording falls back to head/tail truncation. TF-IDF measures statistical distinctiveness; it does not
guarantee semantic importance, and this comparison does not isolate the benefit of clause selection.</p>
<p>BERT fine-tunes all encoder layers and its head for three epochs at 2e-5 using 256 head/tail tokens.
Both neural models use square-root class weights capped at 5 before normalization. Synthetic oversampling is disabled.
A completed epoch budget is not proof of numerical convergence or good generalization.</p>
<p>Vocabulary, IDF and class weights use training data only. Exact validation-description matches are removed from
training only; validation and test rows remain intact.</p>
<p>Unavailable or unsuccessful cases: {html.escape(skipped_text)}.</p>
<p>All trial details: <a href="validation_results.csv">validation_results.csv</a>.
Acceptance details: <a href="requirements.json">requirements.json</a>.
Per-class errors and saved models are in the finalists folders.</p></html>"""
    (output / "comparison.html").write_text(document, encoding="utf-8")
    return status


def benchmark(source, output):
    frame, development, test = load_snapshot(source)
    folds = chronological_folds(development, test.CREATED_DATE.min())
    earliest = folds[0][1] - pd.DateOffset(months=TRAIN_MONTHS)
    development = development[development.CREATED_DATE.ge(earliest)].copy()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.csv", "type_name_to_id.json"):
        shutil.copy2(source / name, output / name)
    for name in ("cleanup_summary.json", "source.json"):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    shutil.copy2(__file__, output / "main_train.py")
    shutil.copy2(Path(__file__).with_name("bert_classifier.py"), output / "bert_classifier.py")
    write_json(output / "splits.json", {"test_start": str(test.CREATED_DATE.min()), "test_rows": len(test),
        "training_months": TRAIN_MONTHS, "earliest_used_date": str(earliest),
        "validation": [{"fold": name, "start": str(start), "rows": len(valid), "row_ids": valid.index.tolist()}
                       for name, start, valid in folds]})
    pd.concat([valid.assign(benchmark_fold=name) for name, _, valid in folds]).to_csv(output / "validation_rows.csv", index=False)
    logger.info("One %d-month training lookback | validation %d days | test %d tickets | 3 development + 3 final fits",
                TRAIN_MONTHS, VALIDATION_DAYS, len(test))
    search = Search(development, folds, output)
    ranked = search.run(candidate_specs(), "Validation")
    search.banks.clear()
    ordered = ranked + [row for row in search.records.values() if not row["eligible"]]
    pd.DataFrame([{"rank": index + 1 if row["eligible"] else None, **flatten_trial(row)}
                  for index, row in enumerate(ordered)]).to_csv(output / "validation_results.csv", index=False)
    if not ranked:
        raise ValueError(f"No completed, converged candidates. Details: {output / 'validation_results.csv'}")
    finalists = ranked
    selection = {"winner_id": ranked[0]["id"], "winner_family": ranked[0]["family"], "finalists": finalists,
                 "selection": "validation_min_accuracy_macroF1_weightedF1; accuracy_tiebreak",
                 "test_used_for_selection": False, "source": str(source.resolve()), "target": TARGET}
    write_json(output / "selection.json", selection)
    results = final_evaluation(development, test, output, finalists)
    skipped = []
    failed = [r for r in search.records.values() if not r["eligible"]]
    if failed:
        skipped.append(f"{len(failed)} failed/nonconverged configurations; see validation_results.csv")
    if any(row["status"] != "ok" or not row.get("converged", False) for row in results):
        skipped.append("One or more final fits failed or did not converge; see finalists/*/result.json")
    status = render_report(output, selection, results, ranked, skipped)
    write_json(output / "summary.json", {"selection": selection, "test_results": results, "requirements": status,
                                        "versions": installed_versions(), "rows": len(frame)})
    write_json(output / "completed.json", {"verdict": status["verdict"], "winner_id": selection["winner_id"]})
    return selection, results, status


def log_mlflow(output, selection, results):
    # Reporting occurs after all local model artifacts are safely written.
    if not importlib.util.find_spec("mlflow"):
        logger.warning("MLflow unavailable; local comparison and models are complete.")
        return
    import mlflow
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ewoc_ttype.ewoc_utils import mlflow_helpers
    else:
        from .ewoc_utils import mlflow_helpers
    try:
        with activity("Uploading comparison and finalists to MLflow"):
            mlflow_helpers.setup_mlflow()
            with mlflow.start_run(run_name=output.name):
                mlflow.set_tag("validation_winner", selection["winner_family"])
                mlflow.log_params({"target": TARGET, "validation_days": VALIDATION_DAYS,
                                   "training_months": TRAIN_MONTHS, "seed": SEED})
                for row in results:
                    if row["status"] == "ok":
                        mlflow.log_metrics({f"{row['family']}_test_{key}": row[key] for key in METRICS})
                for item in output.iterdir():
                    if item.is_file():
                        mlflow.log_artifact(str(item))
                mlflow.log_artifacts(str(output / "finalists"), artifact_path="finalists")
    except mlflow.exceptions.MlflowException as error:
        logger.warning("MLflow upload failed: %s. Local outputs remain at %s", error, output)


def main():
    global BERT_REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", type=Path, help="Versioned cleaned CSV; default is data/cleaned/latest.json")
    inputs.add_argument("--reuse-splits", type=Path, help="Existing snapshot folder; preserve its test split")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
    logger.setLevel(logging.INFO)
    artifacts = ROOT / "EWOCTypePredArtifacts"
    if args.reuse_splits is not None:
        source = args.reuse_splits
    else:
        pointer = ROOT / "data/cleaned/latest.json"
        if args.data is None and not pointer.exists():
            parser.error("Run clean_ewoc_data.py first, or pass --data <whole_ewoc_tickets_cleaned.csv>.")
        csv_path = args.data if args.data is not None else Path(read_json(pointer)["cleaned_file"])
        with activity("Preparing the cleaned chronological snapshot"):
            source = prepare_cleaned_snapshot(csv_path, artifacts)
    if __package__ in {None, ""}:
        from bert_classifier import prepare_bert
    else:
        from .bert_classifier import prepare_bert
    with activity("Preparing BERT through Transformers (first use downloads weights; later runs use the cache)"):
        try:
            BERT_REVISION = prepare_bert(BERT_MODEL_ID, BERT_CACHE)
        except OSError as error:
            raise RuntimeError("Transformers could not load BERT. The first download needs access to Hugging Face; "
                               "check your network/proxy or model-download access, then rerun.") from error
    logger.info("BERT ready: %s | revision %s", BERT_MODEL_ID, BERT_REVISION)
    with activity("Checking local MiniLM weights"):
        minilm_fingerprint = encoder_fingerprint(ENCODER_PATH)
    fingerprint = {"dataset": file_hash(source / "dataset.csv"), "code": file_hash(Path(__file__)),
                   "bert_code": file_hash(Path(__file__).with_name("bert_classifier.py")),
                   "bert_model": BERT_MODEL_ID, "bert_revision": BERT_REVISION,
                   "encoder": minilm_fingerprint, "versions": installed_versions(),
                   "candidates": candidate_specs(), "validation_days": VALIDATION_DAYS}
    output = artifacts / ("limited_benchmark_" + identifier(fingerprint))
    if (output / "completed.json").exists():
        report = read_json(output / "requirements.json")
        print(f"Already completed: {report['verdict']}\n{output / 'comparison.html'}")
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", fingerprint)
    logger.info("Input: %s | models: %s | output: %s", source, ", ".join(FEATURES), output)
    selection, results, _ = benchmark(source, output)
    log_mlflow(output, selection, results)


if __name__ == "__main__":
    main()
