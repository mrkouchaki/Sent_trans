#!/usr/bin/env python3
"""EWOC benchmark: seven feature-based families plus supervised BERT.

Run from ewoc_ticket_type: python src/ewoc_ttype/main_train.py
Input: latest version from clean_ewoc_data.py, or --data <cleaned.csv>.
The newest 56 calendar days form the test; the previous two 28-day blocks validate.
--reuse-splits <folder> preserves an existing snapshot instead.
All windows are screened; the two best candidates/family receive parameter tuning.
Every completed trial is checkpointed; rerunning the same code/data resumes it.
BERT jointly fine-tunes all layers on the two best distinct windows from the faster
search, with capped square-root class weighting and three fixed epochs.

Sources: sklearn's sparse-text classification example; official MiniLM model card;
imbalanced-learn SMOTE and LightGBM LGBMClassifier documentation.
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
from collections import OrderedDict
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATH = ROOT / "models/sentence-transformers/all-MiniLM-L6-v2"
BERT_PATH = ROOT / "models/bert-base-uncased"
SEED, THREADS, TARGET, FOLD_DAYS = 42, 4, 0.85, 28
TEST_DAYS = 56
METRICS = ["accuracy", "macro_f1", "weighted_f1"]
WINDOWS = ("all", "last_12m", "last_6m", "last_3m", "since_2026", "all_decay")
FEATURES = {"word_rf": "word", "wordchar_svm": "wordchar", "wordchar_nb": "wordchar",
            "minilm_lr": "semantic", "minilm_rbf": "semantic", "minilm_lgbm": "semantic",
            "hybrid_svm": "hybrid", "bert_ft": "text"}
# First entry is the screen; remaining entries refine each family's two best screen candidates.
PARAMETERS = {
    "word_rf": [{"n_estimators": 200, "min_samples_leaf": 2},
                {"n_estimators": 400, "min_samples_leaf": 1},
                {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.1}],
    "wordchar_svm": [{"C": c} for c in (2.0, 0.5, 10.0)],
    "wordchar_nb": [{"alpha": a} for a in (0.3, 0.05, 1.0)],
    "minilm_lr": [{"C": c} for c in (10.0, 1.0, 100.0)],
    "minilm_rbf": [{"C": c, "gamma": "scale"} for c in (10.0, 1.0, 100.0)],
    "minilm_lgbm": [{"n_estimators": 150, "num_leaves": 15},
                    {"n_estimators": 300, "num_leaves": 15},
                    {"n_estimators": 300, "num_leaves": 31}],
    "hybrid_svm": [{"C": c} for c in (2.0, 0.5, 10.0)],
    "bert_ft": [{"epochs": 3, "learning_rate": 2e-5}],
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
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                    "torch", "transformers", "lightgbm", "joblib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def load_snapshot(source):
    frame = pd.read_csv(source / "dataset.csv", keep_default_na=False, dtype={"label": str, "DESCRIPTION": str})
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
    validation_start = test_start - pd.Timedelta(days=2 * FOLD_DAYS)
    frame["split"] = np.where(dates.ge(test_start), "test", np.where(dates.ge(validation_start), "validation", "train"))
    key = identifier({"source": file_hash(csv_path), "test_days": TEST_DAYS, "fold_days": FOLD_DAYS, "schema": 1})
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
    folds = []
    for number in range(2):
        end = test_start - pd.Timedelta(days=FOLD_DAYS * (1 - number))
        start = end - pd.Timedelta(days=FOLD_DAYS)
        valid = development[development.CREATED_DATE.ge(start) & development.CREATED_DATE.lt(end)]
        if valid.empty:
            raise ValueError(f"No development tickets in validation block {start} to {end}.")
        folds.append((f"fold_{number + 1}", start, valid.copy()))
    return folds


def window_rows(development, cutoff, window, forbidden_texts=()):
    part = development[development.CREATED_DATE.lt(cutoff)]
    if window.startswith("last_"):
        months = int(window.removeprefix("last_").removesuffix("m"))
        part = part[part.CREATED_DATE.ge(cutoff - pd.DateOffset(months=months))]
    elif window == "since_2026":
        part = part[part.CREATED_DATE.ge(pd.Timestamp("2026-01-01"))]
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


def capped_smote(features, labels):
    """No class grows past 2x; six distinct vectors required; sparse inputs stay sparse."""
    labels = np.asarray(labels)
    counts, strategy = pd.Series(labels).value_counts(), {}
    for label, count in counts.items():
        if count < 6 or count == counts.max():
            continue
        rows = features[labels == label]
        if sparse.issparse(rows):
            rows = rows.tocsr(copy=True)
            rows.eliminate_zeros()
            rows.sort_indices()
            unique = {tuple(zip(row.indices, row.data)) for row in rows}
        else:
            unique = {row.tobytes() for row in rows}
        if len(unique) >= 6:
            strategy[label] = min(int(counts.max()), 2 * int(count))
    if not strategy:
        return features, labels
    from imblearn.over_sampling import SMOTE
    x, y = SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=SEED).fit_resample(features, labels)
    return normalize(x, copy=False), y


def fit_weights(part, cutoff, window, balancing):
    weights = np.ones(len(part))
    if window == "all_decay":
        age_days = (cutoff - part.CREATED_DATE).dt.total_seconds().to_numpy() / 86400
        weights *= np.maximum(0.05, np.exp2(-age_days / 120))  # 120-day half-life; retain old evidence.
    if balancing == "balanced":
        weights *= compute_sample_weight("balanced", part.label)
    elif balancing == "sqrt_balanced":
        weights *= np.minimum(5.0, np.sqrt(compute_sample_weight("balanced", part.label)))
    return weights / weights.mean()


def classifier(spec):
    family, params = spec["family"], spec["params"]
    if family == "bert_ft":
        if __package__ in {None, ""}:
            from bert_classifier import BertClassifier
        else:
            from .bert_classifier import BertClassifier
        return BertClassifier(BERT_PATH, **params, seed=SEED, threads=THREADS)
    if family == "word_rf":
        return RandomForestClassifier(**params, n_jobs=THREADS, random_state=SEED)
    if family in {"wordchar_svm", "hybrid_svm"}:
        return LinearSVC(**params, dual=True, max_iter=10000, tol=1e-3, random_state=SEED)
    if family == "wordchar_nb":
        return ComplementNB(**params)
    if family == "minilm_lr":
        return LogisticRegression(**params, solver="lbfgs", max_iter=2000, tol=1e-4)
    if family == "minilm_rbf":
        return SVC(**params, cache_size=256, probability=False, random_state=SEED)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, learning_rate=0.05, min_child_samples=15, reg_lambda=3.0,
                          colsample_bytree=0.8, verbosity=-1, n_jobs=THREADS, random_state=SEED,
                          deterministic=True, force_col_wise=True)


class MiniLM:
    """Frozen masked-mean MiniLM; cache compatible with the previous training script."""

    def __init__(self, path, cache):
        self.path, self.cache, self.model = path, cache, None
        assets = sorted(p for p in path.iterdir() if p.suffix in {".json", ".txt", ".bin", ".safetensors"})
        if not any(p.suffix in {".bin", ".safetensors"} for p in assets):
            raise FileNotFoundError(f"No local MiniLM weights in {path}")
        digest = hashlib.sha256(b"masked_mean;l2;max256;v1")
        for package in ("torch", "transformers"):
            digest.update(f"{package}:{version(package)}".encode())
        for asset in assets:
            digest.update(asset.name.encode())
            with asset.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        self.fingerprint = digest.hexdigest()
        cache.mkdir(parents=True, exist_ok=True)

    def encode(self, texts, stage):
        texts, inverse = np.unique(np.asarray(texts, dtype=str), return_inverse=True)
        key = hashlib.sha256(self.fingerprint.encode())
        key.update(json.dumps(texts.tolist(), ensure_ascii=False).encode("utf-8"))
        cached = self.cache / (key.hexdigest() + ".npy")
        if cached.exists():
            logger.info("%s: cached embeddings", stage)
            return np.load(cached, allow_pickle=False)[inverse]
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        torch.set_num_threads(THREADS)
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(self.path), local_files_only=True).eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        with activity(f"{stage}: encoding {len(texts)} descriptions on {self.device}"), torch.inference_mode():
            vectors = []
            for start in range(0, len(texts), 32):
                inputs = self.tokenizer(texts[start:start + 32].tolist(), padding=True, truncation=True,
                                        max_length=min(256, self.model.config.max_position_embeddings),
                                        return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
            vectors = np.concatenate(vectors).astype(np.float32)
        temporary = cached.with_suffix(".tmp.npy")
        np.save(temporary, vectors, allow_pickle=False)
        temporary.replace(cached)
        return vectors[inverse]

    def save(self, destination):
        destination.mkdir(exist_ok=True)
        for asset in self.path.iterdir():
            if asset.suffix in {".json", ".txt", ".bin", ".safetensors"}:
                shutil.copy2(asset, destination / asset.name)


class FeatureBank:
    """Fit vocabulary/IDF only on this training window; reuse across its classifiers."""

    def __init__(self, train, valid, train_vectors, valid_vectors):
        self.train, self.valid = train, valid
        self.data = {"semantic": (train_vectors, valid_vectors)}
        self.transforms, self.balanced = {}, {}

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
            else:
                lexical = self.get("wordchar")
                self.data[name] = tuple(normalize(sparse.hstack([a, sparse.csr_matrix(b)], format="csr"))
                                        for a, b in zip(lexical, self.get("semantic")))
        return self.data[name]

    def training(self, spec, cutoff):
        name = FEATURES[spec["family"]]
        x_train, x_valid = self.get(name)
        if spec["balancing"] == "smote":
            if spec["window"] == "all_decay":
                raise ValueError("SMOTE synthetic vectors have no creation date; use real-row recency weights.")
            if name not in self.balanced:
                self.balanced[name] = capped_smote(x_train, self.train.label)
            x_train, y_train = self.balanced[name]
            weights = np.ones(len(y_train))
        else:
            y_train = self.train.label.to_numpy()
            weights = fit_weights(self.train, cutoff, spec["window"], spec["balancing"])
        return x_train, y_train, weights, x_valid


def fit_one(spec, bank, cutoff, checkpoint_dir=None):
    if spec["family"] == "bert_ft":
        if spec["balancing"] != "sqrt_balanced":
            raise ValueError("BERT uses real-text weighted loss, not SMOTE token interpolation.")
        model = classifier(spec)
        weights = fit_weights(bank.train, cutoff, spec["window"], spec["balancing"])
        model.fit(bank.train.DESCRIPTION, bank.train.label, sample_weight=weights, checkpoint_dir=checkpoint_dir)
        predicted = model.predict(bank.valid.DESCRIPTION)
        return model, predicted, {"converged": True, "warnings": [], "fit_rows": len(bank.train),
            "after_smote": len(bank.train), "training_note": "Completed fixed supervised epochs; not a convergence proof.",
            "unseen_label_rows": int((~bank.valid.label.isin(model.classes_)).sum())}
    x_train, y_train, weights, x_valid = bank.training(spec, cutoff)
    model = classifier(spec)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=THREADS):
        warnings.simplefilter("always")
        model.fit(x_train, y_train, sample_weight=weights)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, model.predict(x_valid), {
        "converged": converged, "warnings": [str(w.message) for w in caught],
        "fit_rows": len(bank.train), "after_smote": len(y_train),
        "unseen_label_rows": int((~bank.valid.label.isin(model.classes_)).sum())}


class Search:
    def __init__(self, development, folds, vectors, output):
        self.development, self.folds, self.output = development, folds, output
        self.vectors, self.positions = vectors, pd.Series(np.arange(len(development)), index=development.index)
        self.banks, self.records = OrderedDict(), {}

    def bank(self, fold, window):
        key = (fold[0], window)
        if key not in self.banks:
            _, cutoff, valid = fold
            train = window_rows(self.development, cutoff, window, valid.DESCRIPTION)
            if len(train) < 10 or train.label.nunique() < 2:
                raise ValueError(f"{fold[0]}/{window}: insufficient independent training tickets/classes.")
            if len(self.banks) >= 2:
                self.banks.popitem(last=False)
            self.banks[key] = FeatureBank(train, valid, self.vectors[self.positions.loc[train.index]],
                                         self.vectors[self.positions.loc[valid.index]])
        self.banks.move_to_end(key)
        return self.banks[key]

    def run(self, specs, stage):
        best_seen = -1.0
        logger.info("%s: %d configurations; progress every 10 completions or a new best", stage, len(specs))
        for number, spec in enumerate(specs, 1):
            key = identifier(spec)
            destination = self.output / "trials" / key
            destination.mkdir(parents=True, exist_ok=True)
            done = destination / "result.json"
            if done.exists():
                self.records[key] = read_json(done)
                if self.records[key]["eligible"]:
                    best_seen = max(best_seen, score_floor(self.records[key]))
                continue
            started, parts, scores = perf_counter(), [], []
            with activity(f"{stage} {number}/{len(specs)} {spec['family']}/{spec['window']}/{spec['balancing']}", announce=False):
                for fold in self.folds:
                    name, cutoff, valid = fold
                    metadata, predictions = destination / f"{name}.json", destination / f"{name}.npz"
                    if not metadata.exists():
                        try:
                            bank = self.bank(fold, spec["window"])
                            checkpoint = destination / (name + "_training")
                            if spec["family"] == "bert_ft":
                                model, predicted, info = fit_one(spec, bank, cutoff, checkpoint)
                                model.close()
                            else:
                                _, predicted, info = fit_one(spec, bank, cutoff)
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
                                  eligible=all(row["converged"] for row in scores),
                                  worst_fold_floor=min(score_floor(row) for row in scores))
                else:
                    record.update(status="failed", eligible=False)
                write_json(done, record)
                self.records[key] = record
            improved = record["eligible"] and score_floor(record) > best_seen
            if improved:
                best_seen = score_floor(record)
            if improved or number % 10 == 0 or number == len(specs):
                logger.info("%s %d/%d complete | best validation minimum metric: %.1f%%",
                            stage, number, len(specs), 100 * max(best_seen, 0))
        return self.ranked()

    def ranked(self):
        return sorted([r for r in self.records.values() if r["eligible"]],
                      key=lambda r: (score_floor(r), r["worst_fold_floor"], r["accuracy"]), reverse=True)


def candidate_specs(families, windows, smote_available):
    return [{"family": family, "window": window, "balancing": method, "params": PARAMETERS[family][0]}
            for window in windows for family in families if family != "bert_ft"
            for method in (("none", "balanced", "smote") if smote_available and window != "all_decay"
                           else ("none", "balanced"))]


def bert_specs(ranked):
    windows = list(dict.fromkeys(row["window"] for row in ranked))[:2]
    return [{"family": "bert_ft", "window": window, "balancing": "sqrt_balanced",
             "params": PARAMETERS["bert_ft"][0]} for window in windows]


def refinements(ranked, families):
    candidates = []
    for family in families:
        best = [row for row in ranked if row["family"] == family][:2]
        for row in best:
            for params in PARAMETERS[family][1:]:
                candidates.append({key: row[key] for key in ("family", "window", "balancing")} | {"params": params})
    return sorted(candidates, key=lambda row: (row["window"], row["family"]))


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


def final_evaluation(development, test, vectors, encoder, output, finalists):
    # This function is called only after the validation selection has been written.
    if not (output / "selection.json").exists():
        raise ValueError("Lock validation selection before evaluating the test.")
    test_vectors = encoder.encode(test.DESCRIPTION, "Test")
    positions = pd.Series(np.arange(len(development)), index=development.index)
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
        train = window_rows(development, cutoff, spec["window"])
        started = perf_counter()
        result = {"id": selected["id"], **spec, "validation": {key: selected[key] for key in METRICS},
                  "passed_validation": meets_target(selected), "expected_test_rows": len(test)}
        try:
            with activity(f"Final fit {spec['family']}/{spec['window']} ({len(train)} real tickets)"):
                bank = FeatureBank(train, test, vectors[positions.loc[train.index]], test_vectors)
                checkpoint = destination / "bert_training"
                if spec["family"] == "bert_ft":
                    model, predicted, info = fit_one(spec, bank, cutoff, checkpoint)
                    model.save(destination / "bert")
                    model.close()
                else:
                    model, predicted, info = fit_one(spec, bank, cutoff)
                    joblib.dump(model, destination / "classifier.pkl")
                for name, transform in bank.transforms.items():
                    joblib.dump(transform, destination / f"{name}_tfidf.pkl")
                write_json(destination / "model_config.json", {**spec, "feature_kind": FEATURES[spec["family"]],
                    "classes": model.classes_.tolist(), "encoder": "../../encoder", "pooling": "masked_mean_then_l2",
                    "max_length": 256, "input": "DESCRIPTION only; input CSV wording unchanged",
                    "bert_model": "bert" if spec["family"] == "bert_ft" else None,
                    "bert_truncation": "first 127 + last 127 tokens; CLS/SEP added" if spec["family"] == "bert_ft" else None,
                    "composition": "wordchar=L2(hstack(word,char)); hybrid=L2(hstack(wordchar,MiniLM))"})
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
        checkpoint = destination / "bert_training"
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
    table = pd.DataFrame([flatten_trial(row) for row in ranked[:20]])
    table = table[["family", "window", "balancing", *METRICS, "worst_fold_floor"]]
    for column in METRICS + ["worst_fold_floor"]:
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
<h1>EWOC model and training-window benchmark</h1><p class="result"><strong>{html.escape(verdict)}</strong><br>
Validation-selected winner: {html.escape(chosen['family'])} / {html.escape(chosen['window'])}.</p>
<p>PASS requires accuracy, macro-F1 and weighted-F1 each strictly above 85% on pooled chronological validation
and the unchanged test set. All test rows count; unsupported classes count as errors. No confidence filtering,
label merging or test-based model selection. F1 and class reports use the same true/predicted-label union.</p>
<h2>Finalists: best validation configuration per family</h2>{display.to_html(index=False, escape=True)}
<p>Each finalist was refitted on its selected pre-test window. The winner was locked before test evaluation.
This test has been inspected in earlier development; an observed PASS is not independent production certification.</p>
<p>For cleaned inputs, TYPE_ID is the target. Label rules use the description, so these scores measure agreement
with the corrected business rules; they do not independently prove those rules are correct. Check the human-only
slice when assessing performance on human-written requests. Combined PASS does not imply every slice passes.</p>
<details><summary>Human, superuser and corrected-label test results</summary>{slice_table.to_html(index=False, escape=True)}</details>
<h2>Top 20 validation configurations</h2>{table.to_html(index=False, escape=True)}
<p>Screen: default parameters across every window and applicable balancing method. Refine: alternative
parameters for each family's two best screen candidates. This is a bounded search, not an exhaustive optimum.</p>
<p>BERT uses supervised sequence classification: all encoder layers and the head train together, three epochs at
2e-5, square-root class weights capped at 5 before normalization, 256 head/tail tokens. It runs on the two best
distinct windows from the faster search. Its token IDs are never interpolated with SMOTE. A completed epoch
budget is not proof of numerical convergence or good generalization.</p>
<p>all_decay retains history with a 120-day weight half-life and a 5% weight floor. It uses real-row weights,
without SMOTE. SMOTE elsewhere requires six distinct vectors and adds at most one synthetic row per real row.
Every fold fits its own vocabulary and IDF. Exact validation-description matches are removed from training only.</p>
<p>Unavailable or unsuccessful cases: {html.escape(skipped_text)}.</p>
<p>All trial details: <a href="validation_results.csv">validation_results.csv</a>.
Acceptance details: <a href="requirements.json">requirements.json</a>.
Per-class errors and saved models are in the finalists folders.</p></html>"""
    (output / "comparison.html").write_text(document, encoding="utf-8")
    return status


def benchmark(source, output, encoder, families=None, windows=WINDOWS, smote_available=True):
    families = list(FEATURES) if families is None else families
    frame, development, test = load_snapshot(source)
    folds = chronological_folds(development, test.CREATED_DATE.min())
    output.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.csv", "type_name_to_id.json"):
        shutil.copy2(source / name, output / name)
    for name in ("cleanup_summary.json", "source.json"):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    shutil.copy2(__file__, output / "main_train.py")
    shutil.copy2(Path(__file__).with_name("bert_classifier.py"), output / "bert_classifier.py")
    write_json(output / "splits.json", {"test_start": str(test.CREATED_DATE.min()), "test_rows": len(test),
        "validation": [{"fold": name, "start": str(start), "rows": len(valid), "row_ids": valid.index.tolist()}
                       for name, start, valid in folds]})
    pd.concat([valid.assign(benchmark_fold=name) for name, _, valid in folds]).to_csv(output / "validation_rows.csv", index=False)
    vectors = encoder.encode(development.DESCRIPTION, "Development")
    search = Search(development, folds, vectors, output)
    specs = candidate_specs(families, windows, smote_available)
    logger.info("Screen: %d configurations x 2 time blocks; then refine each family's two best.", len(specs))
    ranked = search.run(specs, "Screen")
    ranked = search.run(refinements(ranked, families), "Refine")
    if "bert_ft" in families and ranked:
        # Free the frozen encoder and feature caches before allocating the larger training model.
        if hasattr(encoder, "model"):
            encoder.model = None
        search.banks.clear()
        ranked = search.run(bert_specs(ranked), "Supervised BERT")
    ordered = ranked + [row for row in search.records.values() if not row["eligible"]]
    pd.DataFrame([{"rank": index + 1 if row["eligible"] else None, **flatten_trial(row)}
                  for index, row in enumerate(ordered)]).to_csv(output / "validation_results.csv", index=False)
    if not ranked:
        raise ValueError(f"No completed, converged candidates. Details: {output / 'validation_results.csv'}")
    leaders = {}
    for row in ranked:
        leaders.setdefault(row["family"], row)
    finalists = list(leaders.values())
    selection = {"winner_id": ranked[0]["id"], "winner_family": ranked[0]["family"], "finalists": finalists,
                 "selection": "pooled_validation_min_accuracy_macroF1_weightedF1; worst_fold_tiebreak",
                 "test_used_for_selection": False, "source": str(source.resolve()), "target": TARGET}
    write_json(output / "selection.json", selection)
    encoder.save(output / "encoder")
    results = final_evaluation(development, test, vectors, encoder, output, finalists)
    skipped = [f"{family}: unavailable dependency or intentionally not enabled" for family in FEATURES if family not in families]
    if not smote_available:
        skipped.append("SMOTE: imbalanced-learn unavailable")
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
                mlflow.log_params({"target": TARGET, "fold_days": FOLD_DAYS, "windows": ",".join(WINDOWS), "seed": SEED})
                for row in results:
                    if row["status"] == "ok":
                        mlflow.log_metrics({f"{row['family']}_test_{key}": row[key] for key in METRICS})
                for item in output.iterdir():
                    if item.is_file():
                        mlflow.log_artifact(str(item))
                mlflow.log_artifacts(str(output / "finalists"), artifact_path="finalists")
                mlflow.log_artifacts(str(output / "encoder"), artifact_path="encoder")
    except mlflow.exceptions.MlflowException as error:
        logger.warning("MLflow upload failed: %s. Local outputs remain at %s", error, output)


def main():
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
    families = list(FEATURES)
    if not importlib.util.find_spec("lightgbm"):
        families.remove("minilm_lgbm")
        logger.warning("LightGBM unavailable; report marks it untested. Install via your JFrog if needed.")
    bert_assets = sorted(p for p in BERT_PATH.glob("*") if p.suffix in {".json", ".txt", ".bin", ".safetensors"})
    if not (BERT_PATH / "config.json").exists() or not any(p.suffix in {".bin", ".safetensors"} for p in bert_assets):
        families.remove("bert_ft")
        logger.warning("BERT untested: put the approved bert-base-uncased model/tokenizer files in %s. No downloads are attempted.", BERT_PATH)
    has_smote = importlib.util.find_spec("imblearn") is not None
    if not has_smote:
        logger.warning("SMOTE unavailable: none/balanced comparisons will run; report marks SMOTE untested.")
    with activity("Checking local MiniLM and cache"):
        encoder = MiniLM(ENCODER_PATH, artifacts / "minilm_embedding_cache")
    fingerprint = {"dataset": file_hash(source / "dataset.csv"), "code": file_hash(Path(__file__)),
                   "bert_code": file_hash(Path(__file__).with_name("bert_classifier.py")),
                   "bert_assets": {p.name: file_hash(p) for p in bert_assets},
                   "encoder": encoder.fingerprint, "versions": installed_versions(),
                   "families": families, "windows": list(WINDOWS), "smote": has_smote}
    output = artifacts / ("model_benchmark_" + identifier(fingerprint))
    if (output / "completed.json").exists():
        report = read_json(output / "requirements.json")
        print(f"Already completed: {report['verdict']}\n{output / 'comparison.html'}")
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", fingerprint)
    logger.info("Input: %s | %d model families | %d windows | checkpoint folder: %s",
                source, len(families), len(WINDOWS), output)
    selection, results, _ = benchmark(source, output, encoder, families, smote_available=has_smote)
    log_mlflow(output, selection, results)


if __name__ == "__main__":
    main()
