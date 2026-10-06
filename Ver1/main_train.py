#!/usr/bin/env python3
"""Train TF-IDF + Random Forest and contrastive MiniLM + a softmax classifier.

    python src/ewoc_ttype/main_train.py --refresh
    python src/ewoc_ttype/main_train.py --reuse-splits EWOCTypePredArtifacts/<previous-run>
    python src/ewoc_ttype/main_train.py --report EWOCTypePredArtifacts/<run-folder>

Newest 15%: chronological test. Older 85%: grouped, stratified development split.
Choose settings using validation, then refit on the older 85% before testing.
MiniLM learns same-class and difficult other-class pairs from training descriptions.
Capped SMOTE runs on training vectors only; the retained softmax head is fitted
with L-BFGS after each encoder epoch. No synthetic text or test data is trained on.
"""

import argparse
import json
import logging
import math
import shutil
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, normalize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils.logging import disable_progress_bar

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ewoc_ttype.ewoc_utils import config, data_loader, mlflow_helpers
else:
    from .ewoc_utils import config, data_loader, mlflow_helpers

ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATH = ROOT / "models/sentence-transformers/all-MiniLM-L6-v2"
WHERE = """WHERE CREATED_DATE >= ADD_MONTHS(SYSDATE, -24)
AND (CREATED_ID IS NULL OR CREATED_ID <> 1)"""
SEED, TARGET, MAX_EPOCHS = config.RANDOM_STATE, 0.85, 3
METRICS = ["accuracy", "macro_f1", "weighted_f1"]
logger = logging.getLogger(__name__)


def load_data(refresh):
    orders = data_loader.fetch_full_table(
        "e911.ENMT_E911_WORK_ORDERS", where_clause=WHERE, force_refresh=refresh)
    types = data_loader.fetch_full_table("e911.LU_EWOC_TYPE", force_refresh=refresh)
    names = types.set_index("TYPE_ID")["TYPE_WO"].dropna()
    frame = orders[["DESCRIPTION", "TYPE_ID", "CREATED_DATE"]].copy()
    frame["label"] = frame.TYPE_ID.map(names)
    frame["CREATED_DATE"] = pd.to_datetime(frame.CREATED_DATE, format="mixed", errors="coerce")
    frame = frame.dropna(subset=["DESCRIPTION", "label", "CREATED_DATE"])
    frame = frame[~frame.DESCRIPTION.map(data_loader.insufficient_information)].copy()
    frame["raw_description"] = frame.DESCRIPTION
    frame["DESCRIPTION"] = frame.DESCRIPTION.map(data_loader.normalize_description)
    frame = frame[frame.DESCRIPTION.str.strip().ne("")]
    frame = frame.sort_values("CREATED_DATE", kind="stable").reset_index(drop=True)
    return frame, {str(name): int(key) for key, name in names.items()}, len(orders)


def split_data(frame):
    """Preserve the time holdout; select development folds by coverage, never scores."""
    if len(frame) < 10:
        raise ValueError("Too few usable tickets for training, validation and test.")
    cutoff = frame.CREATED_DATE.iloc[int(len(frame) * 0.85)]
    development = frame[frame.CREATED_DATE < cutoff]
    test = frame[frame.CREATED_DATE >= cutoff].copy()
    if development.DESCRIPTION.nunique() < 5 or development.label.nunique() < 2:
        raise ValueError("Need at least five distinct development texts and two classes.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    expected = development.label.value_counts(normalize=True)
    candidates = []
    with warnings.catch_warnings():
        # Rare classes are retained in training and disclosed in the run summary.
        warnings.filterwarnings("ignore", message="The least populated class in y has only")
        folds = list(splitter.split(development, development.label, development.DESCRIPTION))
    for ti, vi in folds:
        train, valid = development.iloc[ti], development.iloc[vi]
        # Keep any class represented by just one text group in training.
        missing = set(valid.label) - set(train.label)
        move = valid.DESCRIPTION.isin(valid.loc[valid.label.isin(missing), "DESCRIPTION"])
        train, valid = pd.concat([train, valid[move]]), valid[~move]
        gap = (valid.label.value_counts(normalize=True).reindex(expected.index, fill_value=0)
               - expected).abs().sum()
        candidates.append(((len(set(expected.index) - set(valid.label)), gap), train, valid))
    _, train, valid = min(candidates, key=lambda item: item[0])
    # Repair missing validation classes when another independent text group is available.
    for label in sorted(set(development.label) - set(valid.label)):
        groups = train.groupby("label").DESCRIPTION.nunique()
        for text in train.loc[train.label.eq(label), "DESCRIPTION"].unique():
            move = train.DESCRIPTION.eq(text)
            if (groups.loc[train.loc[move, "label"].unique()] > 1).all():
                valid, train = pd.concat([valid, train[move]]), train[~move]
                break
    if valid.empty:
        raise ValueError("No independent descriptions available for validation.")
    return train.sort_index().copy(), valid.sort_index().copy(), test


def balance_training(features, labels):
    """At most double a class; k=5 only for classes with six distinct real vectors."""
    counts = pd.Series(labels).value_counts()
    targets = {}
    for label, count in counts.items():
        if count < 6 or count == counts.max():
            continue
        rows = features[np.asarray(labels) == label]
        # Sparse TF-IDF stays sparse; repeated rows do not count as new evidence.
        if hasattr(rows, "tocsr"):
            rows = rows.tocsr()
            distinct = {tuple(zip(row.indices, row.data)) for row in rows}
        else:
            distinct = {row.tobytes() for row in rows}
        if len(distinct) >= 6:
            targets[int(label)] = min(int(counts.max()), 2 * int(count))
    if not targets:
        return features, labels
    features, labels = SMOTE(sampling_strategy=targets, k_neighbors=5,
                             random_state=SEED).fit_resample(features, labels)
    return normalize(features, copy=False), labels


class TicketClassifier(nn.Module):
    """MiniLM with normalized mean pooling and a retained linear softmax head."""

    def __init__(self, path, classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.max_length = min(256, self.encoder.config.max_position_embeddings)
        self.normalize_embeddings = True
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, classes))

    def inputs(self, texts):
        tokens = self.tokenizer(list(texts), padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        return {key: value.to(next(self.parameters()).device) for key, value in tokens.items()}

    def features(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, dim=1) if self.normalize_embeddings else pooled

    def forward(self, inputs):
        return self.head(self.features(inputs))

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({key: value.detach().cpu() for key, value in self.head.state_dict().items()},
                   path / "classification_head.pt")
        (path / "classifier_config.json").write_text(json.dumps({
            "num_labels": self.head[1].out_features, "pooling": "masked_mean",
            "dropout": 0.1, "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        metadata = json.loads((path / "classifier_config.json").read_text(encoding="utf-8"))
        model = cls(path, metadata["num_labels"])
        model.max_length = metadata["max_length"]
        model.normalize_embeddings = metadata.get("normalize_embeddings", False)
        model.head.load_state_dict(torch.load(path / "classification_head.pt", map_location="cpu", weights_only=True))
        return model.eval()


@torch.inference_mode()
def encode_training(model, texts):
    model.eval()
    # Evaluation/progress iteration must not consume the encoder dropout RNG.
    batches = DataLoader(list(texts), batch_size=32, shuffle=False, num_workers=0,
                         generator=torch.Generator().manual_seed(SEED))
    vectors = [model.features(model.inputs(batch)).cpu() for batch in
               tqdm(batches, desc="Encoding training descriptions", mininterval=5, dynamic_ncols=True)]
    return torch.cat(vectors).numpy()


def training_pairs(texts, labels, features, seed):
    """At most 64 distinct positive pairs/class; negatives come from nearby other classes."""
    rng = np.random.default_rng(seed)
    examples = pd.DataFrame({"text": texts, "label": labels})
    conflicts = examples.groupby("text").label.transform("nunique").gt(1)
    eligible = examples[~conflicts].drop_duplicates(["text", "label"])
    groups = {int(label): part.index.to_numpy() for label, part in eligible.groupby("label")}
    pairs = []
    for label, indices in groups.items():
        count = min(64, len(indices) * (len(indices) - 1) // 2)
        others = eligible.index[eligible.label.ne(label)].to_numpy()
        if not count or not len(others):
            continue
        chosen = set()
        while len(chosen) < count:
            chosen.add(tuple(sorted(rng.choice(indices, 2, replace=False).tolist())))
        search = NearestNeighbors(n_neighbors=min(5, len(others)), metric="cosine", n_jobs=config.N_JOBS)
        near = others[search.fit(features[others]).kneighbors(features[indices], return_distance=False)]
        negatives = dict(zip(indices.tolist(), near))
        for a, b in sorted(chosen):
            pairs.extend([(texts[a], texts[b], 1.0), (texts[a], texts[int(rng.choice(negatives[a]))], 0.0)])
    rng.shuffle(pairs)
    return pairs


def fit_semantic_head(model, features, labels):
    """Regularized multinomial softmax, optimized on capped-SMOTE training vectors."""
    sampled, targets = balance_training(features, labels)
    x, y = torch.tensor(sampled, dtype=torch.float64), torch.tensor(targets, dtype=torch.long)
    head = nn.Linear(x.shape[1], model.head[1].out_features, dtype=torch.float64)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(head(x), y) + 0.001 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.head[1].load_state_dict({key: value.float() for key, value in head.state_dict().items()})
    model.training_rows_after_smote = len(y)
    with torch.inference_mode():
        model.eval()
        real_vectors = torch.tensor(features, device=next(model.parameters()).device)
        model.training_predictions = model.head(real_vectors).argmax(1).cpu().numpy()


def transformer_epochs(frame, labels, epochs):
    """Contrastive representation learning, followed by an optimized classification head."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info("Loading local MiniLM weights...")
    model = TicketClassifier(ENCODER_PATH, len(np.unique(labels)))
    if len(model.tokenizer) < 1000:
        raise ValueError("MiniLM tokenizer is incomplete; check the local model folder.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=2e-5, weight_decay=0.01)
    texts = frame.DESCRIPTION.tolist()
    features = encode_training(model, texts)
    logger.info("Mining training pairs on %s; CPU threads=%d", device, torch.get_num_threads())
    pairs = training_pairs(texts, labels, features, SEED)
    # Keep the schedule's epoch horizon fixed when refitting for the selected epoch count.
    steps = MAX_EPOCHS * math.ceil(len(pairs) / 16)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    logger.info("MiniLM: %d pairs/epoch; %d epochs; capped SMOTE for the classification head", len(pairs), epochs)
    for epoch in range(1, epochs + 1):
        if epoch > 1:
            pairs = training_pairs(texts, labels, features, SEED + epoch - 1)
        batches = DataLoader(pairs, batch_size=16, shuffle=False, num_workers=0,
                             generator=torch.Generator().manual_seed(SEED + epoch))
        model.train()
        total_loss, total_rows = 0.0, 0
        with tqdm(batches, desc=f"Contrastive MiniLM {epoch}/{epochs}", unit="batch",
                  dynamic_ncols=True, mininterval=5) as progress:
            for left, right, target in progress:
                optimizer.zero_grad(set_to_none=True)
                vectors = model.features(model.inputs(list(left) + list(right)))
                n = len(left)
                similarity = (vectors[:n] * vectors[n:]).sum(1)
                loss = F.mse_loss(similarity, target.to(similarity))
                loss.backward()
                nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.detach().item() * len(target)
                total_rows += len(target)
                progress.set_postfix(loss=f"{total_loss / total_rows:.4f}", refresh=False)
        features = encode_training(model, texts)
        logger.info("Fitting softmax head on training embeddings with capped SMOTE")
        fit_semantic_head(model, features, labels)
        yield model.eval(), epoch


@torch.inference_mode()
def predict_texts(model, frame, description="Evaluating MiniLM"):
    model.eval()
    batches = DataLoader(frame.DESCRIPTION.tolist(), batch_size=32, shuffle=False, num_workers=0,
                         generator=torch.Generator().manual_seed(SEED))
    predictions = []
    for texts in tqdm(batches, desc=description, unit="batch", dynamic_ncols=True, mininterval=5):
        predictions.extend(model(model.inputs(texts)).argmax(1).cpu().tolist())
    return np.asarray(predictions)


def metric_values(actual, predicted):
    # Default union of true/predicted labels: identical definition in every report.
    return {"accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
            "weighted_f1": f1_score(actual, predicted, average="weighted", zero_division=0)}


def selection_score(scores):
    return min(scores[key] for key in METRICS)


def fit_forest(features, labels):
    started = perf_counter()
    logger.info("Preparing TF-IDF training vectors with capped SMOTE")
    features, labels = balance_training(features, labels)
    logger.info("Fitting Random Forest on %d vectors", len(labels))
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=config.N_JOBS, random_state=SEED)
    model.fit(features, labels)
    logger.info("Random Forest finished in %.1f min", (perf_counter() - started) / 60)
    return model, len(labels)


def evaluate(predicted_ids, frame, labels, training_texts):
    predicted = labels.inverse_transform(np.asarray(predicted_ids, dtype=int))
    scores = metric_values(frame.label, predicted)
    unseen = ~frame.DESCRIPTION.isin(training_texts)
    scores.update(rows=len(frame), unseen_label_rows=int((~frame.label.isin(labels.classes_)).sum()),
                  new_description_rows=int(unseen.sum()))
    if unseen.any():
        scores["new_description_accuracy"] = accuracy_score(frame.label[unseen], predicted[unseen])
    return scores, predicted


def save_evaluation(destination, split, frame, predicted):
    frame.assign(predicted=predicted, correct=predicted == frame.label.to_numpy()).to_csv(
        destination / f"{split}_predictions.csv", index=False)
    report = classification_report(frame.label, predicted, output_dict=True, zero_division=0)
    observed = sorted(set(frame.label) | set(predicted))
    filename = "class_metrics.csv" if split == "test" else "validation_class_metrics.csv"
    pd.DataFrame({label: report[label] for label in observed}).T.to_csv(destination / filename, index_label="label")


def print_summary(output):
    comparison = pd.read_csv(output / "comparison.csv")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    display = comparison[["model", "split", *METRICS]].copy()
    display[METRICS] = display[METRICS].map(lambda value: f"{value:.1%}")
    print("\n" + display.to_string(index=False))
    winner = summary["validation_winner"]
    scores = comparison[comparison.model.eq(winner) & comparison.split.eq("test")].iloc[0]
    passed = selection_score(scores) > TARGET
    print(f"Validation-selected model: {winner}; all three test metrics >85%: {'YES' if passed else 'NO'}")
    missing, total = int(scores.unseen_label_rows), int(scores.rows)
    print(f"Unseen test labels: {missing}/{total}; label-coverage ceiling: {1 - missing / total:.1%}")
    counts = pd.read_csv(output / "class_counts.csv")
    print(f"Training classes absent from validation: {int(((counts.train > 0) & (counts.validation == 0)).sum())}")
    print(f"Files: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Reload Oracle data")
    parser.add_argument("--reuse-splits", type=Path, help="Reuse a previous human-ticket run's exact data and splits")
    parser.add_argument("--report", type=Path, help="Show an existing run without training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    disable_progress_bar()
    logger.setLevel(logging.INFO)
    for name in ("azure", "urllib3", "httpx", "sentence_transformers", "transformers", "mlflow"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.report is not None:
        print_summary(args.report)
        return
    config.validate_config()
    mlflow_helpers.setup_mlflow()
    run_name = "human_tickets_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "EWOCTypePredArtifacts" / run_name
    output.mkdir(parents=True)
    with mlflow.start_run(run_name=run_name):
        logger.info("Loading human ticket data...")
        if args.reuse_splits:
            frame = pd.read_csv(args.reuse_splits / "dataset.csv", keep_default_na=False)
            frame["CREATED_DATE"] = pd.to_datetime(frame.CREATED_DATE, format="mixed")
            train, valid, test = [frame[frame.split.eq(name)].drop(columns="split").copy()
                                  for name in ("train", "validation", "test")]
            type_ids = json.loads((args.reuse_splits / "type_name_to_id.json").read_text(encoding="utf-8"))
            fetched = len(frame)
            if any(part.empty for part in (train, valid, test)) or set(valid.label) - set(train.label):
                raise ValueError("Reuse a run with nonempty splits and every validation class in training.")
            if (set(train.DESCRIPTION) & set(valid.DESCRIPTION) or
                    max(train.CREATED_DATE.max(), valid.CREATED_DATE.max()) >= test.CREATED_DATE.min()):
                raise ValueError("Saved splits overlap by development text or cross the test date boundary.")
        else:
            frame, type_ids, fetched = load_data(args.refresh)
            train, valid, test = split_data(frame)
        final_train = pd.concat([train, valid]).sort_values("CREATED_DATE", kind="stable")
        labels = LabelEncoder().fit(train.label)
        y_train = labels.transform(train.label)
        joblib.dump(labels, output / "label_encoder.pkl")
        (output / "type_name_to_id.json").write_text(json.dumps(type_ids, indent=2), encoding="utf-8")
        pd.concat([part.assign(split=name) for name, part in
                   [("train", train), ("validation", valid), ("test", test)]]).to_csv(output / "dataset.csv", index=False)
        counts = pd.concat({"train": train.label.value_counts(), "validation": valid.label.value_counts(),
                            "test": test.label.value_counts()}, axis=1).fillna(0).astype(int)
        counts["final_train"] = counts.train + counts.validation
        counts.to_csv(output / "class_counts.csv", index_label="label")
        mlflow.log_params({"where": WHERE, "seed": SEED, "target": TARGET,
                           "split": "newest_15pct_test; stratified_group_validation_within_older_85pct",
                           "selection": "minimum_of_accuracy_macro_f1_weighted_f1",
                           "smote": "both_models_training_vectors_only; max_2x; minimum_6_distinct_vectors; k_5",
                           "semantic": "MiniLM; normalized_mean; softmax_LBFGS_100; head_L2_0.001",
                           "fine_tuning": "contrastive_cosine_MSE; hard_negatives_top5; max_64_positive_pairs_per_class; lr_2e-5; warmup_10pct"})
        rows, settings, history = [], {}, []
        for kind in ("tfidf", "transformer"):
            logger.info("Developing %s: train=%d validation=%d", kind, len(train), len(valid))
            destination = output / kind
            destination.mkdir()
            if kind == "transformer":
                candidates = transformer_epochs(train, y_train, MAX_EPOCHS)
            else:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                             sublinear_tf=True, dtype=np.float32)
                x_train = vectorizer.fit_transform(train.DESCRIPTION)
                model, _ = fit_forest(x_train, y_train)
                x_valid = vectorizer.transform(valid.DESCRIPTION)
                candidates = [(model, 0)]
            best = -1
            for model, epoch in candidates:
                # Measure fit on real training rows, without dropout or synthetic examples.
                train_ids = (model.training_predictions
                             if kind == "transformer" else model.predict(x_train))
                train_scores = metric_values(train.label, labels.inverse_transform(train_ids.astype(int)))
                predicted_ids = (predict_texts(model, valid, "Checking validation")
                                 if kind == "transformer" else model.predict(x_valid))
                scores, predicted = evaluate(predicted_ids, valid, labels, set(train.DESCRIPTION))
                logger.info("%s epoch=%d | train acc/macro-F1: %.1f%%/%.1f%% | validation acc/macro-F1: %.1f%%/%.1f%%",
                            kind, epoch, 100 * train_scores["accuracy"], 100 * train_scores["macro_f1"],
                            100 * scores["accuracy"], 100 * scores["macro_f1"])
                history.append({"model": kind, "epoch": epoch, **scores,
                                **{"train_" + key: value for key, value in train_scores.items()}})
                pd.DataFrame(history).to_csv(output / "validation_history.csv", index=False)
                if selection_score(scores) > best:
                    best, best_scores, best_predicted = selection_score(scores), scores, predicted
                    settings[kind] = {"epochs": epoch} if kind == "transformer" else {"estimators": 300}
                    if kind == "transformer":
                        model.save(destination / "validation_checkpoint")
                        (destination / "validation_checkpoint" / "metrics.json").write_text(
                            json.dumps({"epoch": epoch, "training": train_scores, "validation": scores}, indent=2),
                            encoding="utf-8")
                        logger.info("Saved best validation checkpoint (epoch %d)", epoch)
            rows.append({"model": kind, "split": "validation", **best_scores})
            save_evaluation(destination, "validation", valid, best_predicted)
            mlflow.log_params({kind + "_" + key: value for key, value in settings[kind].items()})
            mlflow.log_metrics({kind + "_validation_" + key: value for key, value in best_scores.items()})
            del model, candidates

        winner = max(rows, key=selection_score)["model"]  # Fixed before either test score is seen.
        y_final = labels.transform(final_train.label)
        for kind in ("tfidf", "transformer"):
            logger.info("Refitting %s on %d tickets", kind, len(final_train))
            destination = output / kind
            if kind == "transformer":
                # Fresh original weights; train the selected number of contrastive epochs.
                for model, _ in transformer_epochs(final_train, y_final, settings[kind]["epochs"]):
                    pass
                model.save(destination / "text_classifier")
                predicted_ids = predict_texts(model, test)
                sampled = model.training_rows_after_smote
            else:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                             sublinear_tf=True, dtype=np.float32)
                model, sampled = fit_forest(vectorizer.fit_transform(final_train.DESCRIPTION), y_final)
                predicted_ids = model.predict(vectorizer.transform(test.DESCRIPTION))
                joblib.dump(model, destination / "classifier.pkl")
                joblib.dump(vectorizer, destination / "tfidf_vectorizer.pkl")
            scores, predicted = evaluate(predicted_ids, test, labels, set(final_train.DESCRIPTION))
            rows.append({"model": kind, "split": "test", **scores})
            save_evaluation(destination, "test", test, predicted)
            mlflow.log_metrics({kind + "_test_" + key: value for key, value in scores.items()})
            mlflow.log_metric(kind + "_final_fit_rows", sampled)
            del model

        pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
        pd.DataFrame(history).to_csv(output / "validation_history.csv", index=False)
        summary = {"validation_winner": winner, "settings": settings, "where": WHERE, "seed": SEED,
                   "target": TARGET, "eligible_rows": len(frame),
                   "excluded_unusable_rows": None if args.reuse_splits else fetched - len(frame),
                   "reused_splits": str(args.reuse_splits) if args.reuse_splits else None,
                   "training_only_labels": sorted(set(train.label) - set(valid.label)),
                   "test_start": str(test.CREATED_DATE.min()), "final_training_rows": len(final_train),
                   "encoder_path": str(ENCODER_PATH), "final_training_splits": ["train", "validation"],
                   "versions": {"mlflow": mlflow.__version__, **{name: version(name) for name in
                                ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "transformers", "torch", "joblib"]}}}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        shutil.copy2(__file__, output / "main_train.py")
        shutil.copy2(data_loader.__file__, output / "data_loader.py")
        mlflow.set_tag("validation_winner", winner)
        logger.info("Uploading run artifacts to MLflow...")
        mlflow.log_artifacts(str(output))
    print_summary(output)


if __name__ == "__main__":
    main()
