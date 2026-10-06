#!/usr/bin/env python3
"""
EWOC TType Training Script
Trains Random Forest model for ticket type prediction with MLflow tracking.

Usage:
    python src/ewoc_ttype/main_train.py

Environment Variables:
    DB_DSN_PRD, DB_USER_PRD, DB_PASSWORD_PRD: Oracle credentials
    MLFLOW_TRACKING_URI: MLflow server URI
    DATA_CACHE_DIR: Local data cache directory
    ORACLE_CLIENT_DIR_WINDS/LINUX: Path to Oracle Instant Client
"""

import argparse
import json
import logging
import shutil
from importlib.metadata import version as package_version
from datetime import datetime
from pathlib import Path
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Add src to path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ewoc_ttype.ewoc_utils import config, mlflow_helpers, data_loader
else:
    from .ewoc_utils import config, mlflow_helpers, data_loader

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "EWOCTypePredArtifacts" / "tfidf"


def _build_training_frame(
    work_orders_table: str,
    type_table: str,
    rca_table: str,
    market_table: str,
    status_table: str,
    where_clause: str | None,
    force_refresh: bool,
):
    """Load Oracle tables and enrich work orders with lookup labels."""
    logger.info("Loading work orders and lookup tables from Oracle")
    work_orders = data_loader.fetch_full_table(work_orders_table, where_clause=where_clause, force_refresh=force_refresh)
    lu_type = data_loader.fetch_full_table(type_table, force_refresh=force_refresh)
    lu_rca = data_loader.fetch_full_table(rca_table, force_refresh=force_refresh)
    lu_market = data_loader.fetch_full_table(market_table, force_refresh=force_refresh)
    _ = data_loader.fetch_full_table(status_table, force_refresh=force_refresh)

    type_dict = lu_type.set_index("TYPE_ID")["TYPE_WO"].to_dict()
    rca_dict = lu_rca.set_index("RCA_ID")["RCA"].to_dict()
    market_dict = lu_market.set_index("ID")["MARKET_CLUSTER"].to_dict()

    type_name_to_id = {
        str(name): int(type_id)
        for type_id, name in type_dict.items()
        if pd.notna(name) and pd.notna(type_id)
    }

    work_orders["WorkOrder_Info"] = work_orders["TYPE_ID"].map(type_dict)
    work_orders["RCA_Info"] = work_orders["RCA_ID"].map(rca_dict)
    work_orders["Market_Info"] = work_orders["MARKET_ID"].map(market_dict)
    logger.info("Enriched dataset shape: %s", work_orders.shape)

    return work_orders, type_name_to_id


def _clean_data(
    df: pd.DataFrame,
    label_col: str = "WorkOrder_Info",
    desc_col: str = "DESCRIPTION",
    date_col: str = "CREATED_DATE",
    min_class_count: int = 75,
):
    """Drop unusable labels, descriptions and dates, then normalize request text."""
    logger.info("Cleaning and filtering training data")
    cleaned = df.dropna(subset=[label_col, desc_col]).copy()
    if date_col in cleaned.columns:
        cleaned[date_col] = pd.to_datetime(cleaned[date_col], errors="coerce")
        cleaned = cleaned.dropna(subset=[date_col])

    cleaned = cleaned[~cleaned[desc_col].map(data_loader.insufficient_information)].copy()
    cleaned[desc_col] = cleaned[desc_col].map(data_loader.normalize_description)
    cleaned = cleaned.reset_index(drop=True)

    cleaned.to_csv(ARTIFACT_DIR / "cleaned_data.csv", index=False)
    logger.info("Cleaned dataset shape: %s", cleaned.shape)
    return cleaned


def _split_frames(df, min_class_count, test_fraction=None, evaluation_only=False):
    """Split chronologically; evaluation-only mode holds out every later row."""
    ordered = df.sort_values("CREATED_DATE", kind="stable")
    test_fraction = config.TEST_SIZE if test_fraction is None else test_fraction
    test_start = int(len(ordered) * (1 - test_fraction))
    validation_start = test_start if evaluation_only else int(test_start * 0.8)
    train = ordered.iloc[:validation_start].copy()
    validation = ordered.iloc[validation_start:test_start].copy()
    test = ordered.iloc[test_start:].copy()
    if not evaluation_only:
        validation = validation[~validation.DESCRIPTION.isin(train.DESCRIPTION)].drop_duplicates("DESCRIPTION")
        test = test[~test.DESCRIPTION.isin(ordered.iloc[:test_start].DESCRIPTION)].drop_duplicates("DESCRIPTION")
    counts = train.WorkOrder_Info.value_counts()
    train = train[train.WorkOrder_Info.isin(counts[counts >= min_class_count].index)]
    if train.WorkOrder_Info.nunique() < 2 or (not evaluation_only and validation.empty) or test.empty:
        raise RuntimeError("Need at least two training classes and nonempty evaluation sets")
    logger.info("Chronological split: train=%d validation=%d test=%d", len(train), len(validation), len(test))
    return train, validation, test


# TF-IDF fits only training text; transformer weights remain frozen.
def _vectorize_and_encode(train, validation=None, test=None, model_kind="tfidf", embedding_model=None):
    """Fit training features/labels and transform untouched validation/test text."""
    frames = [train] + ([validation] if validation is not None else []) + ([test] if test is not None else [])
    tfidf_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=20,
        ngram_range=(1, 6),
        stop_words="english",
        max_features=5000,
    )
    if model_kind == "transformer":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model, local_files_only=Path(embedding_model).exists())
        # A model folder without tokenizer files silently loads a tiny vocabulary and maps all words to [UNK].
        if len(encoder.tokenizer) < 1000:
            raise RuntimeError(f"Tokenizer files missing in {embedding_model}: add vocab.txt, tokenizer.json, tokenizer_config.json")
        encoder.save(str(ARTIFACT_DIR / "sentence_encoder"))
        features = [encoder.encode(frame.DESCRIPTION.tolist(), batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=True) for frame in frames]
        tfidf_vectorizer = None
    else:
        features = [tfidf_vectorizer.fit_transform(train.DESCRIPTION)]
        features.extend(tfidf_vectorizer.transform(frame.DESCRIPTION) for frame in frames[1:])

    label_encoder = LabelEncoder()
    label_encoder.fit(train.WorkOrder_Info)
    label_ids = {label: index for index, label in enumerate(label_encoder.classes_)}
    labels = [frame.WorkOrder_Info.map(label_ids).fillna(-1).to_numpy(dtype=int) for frame in frames]
    label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    joblib.dump(tfidf_vectorizer, ARTIFACT_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, ARTIFACT_DIR / "label_encoder.pkl")
    joblib.dump(label_mapping, ARTIFACT_DIR / "label_mapping.pkl")
    return features, labels, tfidf_vectorizer, label_encoder


def _balance_training(X_train, y_train):
    """Balance training rows only, adapting SMOTE to the smallest training class."""
    neighbors = min(5, int(pd.Series(y_train).value_counts().min()) - 1)
    if neighbors < 1:
        raise ValueError("SMOTE requires at least two training examples per class")
    smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    joblib.dump(X_train_bal, ARTIFACT_DIR / "X_train.pkl")
    joblib.dump(y_train_bal, ARTIFACT_DIR / "y_train.pkl")
    return X_train_bal, y_train_bal


def _train_model(X_train, y_train, model_kind="tfidf"):
    """Train the selected classifier and persist it in the run's artifact folder."""
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=config.RANDOM_STATE) if model_kind == "transformer" else RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, ARTIFACT_DIR / "rf_model.pkl")
    return model


def _select_threshold(model, features, labels, target_accuracy=0.9, min_accepted=30):
    """Choose the lowest validation threshold meeting accuracy and sample-count targets."""
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)
    correct = model.classes_[probabilities.argmax(axis=1)] == labels
    for threshold in np.linspace(0, 1, 101):
        accepted = confidence >= threshold
        if accepted.sum() >= min_accepted and correct[accepted].mean() >= target_accuracy:
            return float(threshold)
    logger.warning("No validation threshold meets the target; this model will abstain on all predictions")
    return 1.01


def _evaluate_model(model, X_test, y_test, label_encoder, min_confidence=0.0):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    max_proba = np.max(y_proba, axis=1)
    sorted_proba = np.sort(y_proba, axis=1)
    second_best = sorted_proba[:, -2] if sorted_proba.shape[1] > 1 else np.zeros_like(max_proba)
    margin = max_proba - second_best
    entropy = -np.sum(np.clip(y_proba, 1e-12, 1.0) * np.log(np.clip(y_proba, 1e-12, 1.0)), axis=1)

    report = classification_report(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)),
                                   target_names=label_encoder.classes_, zero_division=0)
    accepted = max_proba >= min_confidence

    metrics = {
        "accuracy": accuracy,
        "precision": precision_weighted,
        "recall": recall_weighted,
        "f1": f1_weighted,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "prediction_confidence_mean": float(np.mean(max_proba)),
        "prediction_confidence_p10": float(np.percentile(max_proba, 10)),
        "prediction_confidence_p50": float(np.percentile(max_proba, 50)),
        "prediction_confidence_p90": float(np.percentile(max_proba, 90)),
        "low_confidence_rate_lt_0_6": float(np.mean(max_proba < 0.60)),
        "prediction_margin_mean": float(np.mean(margin)),
        "prediction_entropy_mean": float(np.mean(entropy)),
        "evaluated_samples": int(len(y_test)),
        "num_classes": int(len(label_encoder.classes_)),
        "confidence_threshold": min_confidence,
        "coverage": float(accepted.mean()),
        "rejection_rate": float(1 - accepted.mean()),
        "accepted_accuracy": float((y_pred[accepted] == y_test[accepted]).mean()) if accepted.any() else 0.0,
        "unseen_label_rows": int(np.sum(y_test == -1)),
    }
    with open(ARTIFACT_DIR / "classification_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report)
    joblib.dump(metrics, ARTIFACT_DIR / "test_metrics.pkl")

    logger.info("Metrics: %s", metrics)
    return metrics, report, y_pred


def _build_training_summary(cleaned_df, X, y, X_train, X_test, y_train, y_test, metrics):
    class_counts = pd.Series(y).value_counts().sort_index()
    summary = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": {
            "cleaned_rows": int(len(cleaned_df)),
            "feature_rows": int(X.shape[0]),
            "feature_columns": int(X.shape[1]),
            "sparsity": float(1.0 - ((X.nnz if hasattr(X, "nnz") else np.count_nonzero(X)) / float(X.shape[0] * X.shape[1]))),
            "description_length_mean": float(cleaned_df["DESCRIPTION"].fillna("").str.len().mean()),
            "description_length_p95": float(cleaned_df["DESCRIPTION"].fillna("").str.len().quantile(0.95)),
            "class_count": int(class_counts.shape[0]),
            "class_imbalance_ratio": float(class_counts.max() / max(class_counts.min(), 1)),
        },
        "split": {
            "train_rows_before_smote": int(X_train.shape[0]),
            "train_rows_after_smote": int(len(y_train)),
            "test_rows": int(X_test.shape[0]),
        },
        "metrics": metrics,
    }
    return summary


def _write_training_diagnostics(summary, label_encoder, y_true, y_pred):
    with open(ARTIFACT_DIR / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    class_metrics = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    with open(ARTIFACT_DIR / "class_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(class_metrics, handle, indent=2)


class TicketTypeClassifier(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper returning ticket type details and model confidence."""

    def __init__(self, tfidf_vectorizer, label_encoder, rf_model, type_name_to_id,
                 model_kind="tfidf", min_confidence=0.0, normalize_text=False):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.label_encoder = label_encoder
        self.rf_model = rf_model
        self.type_name_to_id = type_name_to_id
        self.model_kind = model_kind
        self.min_confidence = min_confidence
        self.normalize_text = normalize_text

    def load_context(self, context):
        """Load packaged transformer weights locally; TF-IDF needs no extra loading."""
        if getattr(self, "model_kind", "tfidf") == "transformer":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(context.artifacts["sentence_encoder"], local_files_only=True)

    def predict(self, context, model_input: pd.DataFrame):
        if "DESCRIPTION" not in model_input.columns:
            raise ValueError("Input must include DESCRIPTION column")

        descriptions = model_input["DESCRIPTION"].fillna("").astype(str)
        positions = [position for position, text in enumerate(descriptions)
                     if not data_loader.insufficient_information(text)]
        results = [data_loader.abstention("INSUFFICIENT_INFORMATION") for _ in descriptions]
        if not positions:
            return results
        descriptions = descriptions.iloc[positions]
        if getattr(self, "normalize_text", False):
            descriptions = descriptions.map(data_loader.normalize_description)
        if getattr(self, "model_kind", "tfidf") == "transformer":
            X = self.encoder.encode(descriptions.tolist(), batch_size=64, normalize_embeddings=True)
        else:
            X = self.tfidf_vectorizer.transform(descriptions)
        y_pred = self.rf_model.predict(X)
        y_proba = self.rf_model.predict_proba(X)
        type_names = self.label_encoder.inverse_transform(y_pred)
        class_index = {class_label: idx for idx, class_label in enumerate(self.rf_model.classes_)}

        predictions = [
            {
                "workOrderTypeId": int(self.type_name_to_id[name]) if name in self.type_name_to_id else None,
                "workOrderType": str(name),
                "confidence": float(y_proba[row_idx, class_index[predicted_label]]),
            }
            for row_idx, (name, predicted_label) in enumerate(zip(type_names, y_pred))
        ]
        predictions = data_loader.apply_prediction_policy(predictions, getattr(self, "min_confidence", 0.0))
        for position, prediction in zip(positions, predictions):
            results[position] = prediction
        return results


def _model_code_path():
    """Stage only required Python sources for MLflow, never ticket data or secrets."""
    source = Path(__file__).resolve().parent
    destination = ARTIFACT_DIR / "code" / "ewoc_ttype"
    for relative in [Path("main_train.py"), *[Path("ewoc_utils") / name for name in
                      ("__init__.py", "config.py", "data_loader.py", "mlflow_helpers.py")]]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return str(destination)


def _register_model(
    tfidf_vectorizer,
    label_encoder,
    rf_model,
    type_name_to_id,
    metrics,
    cleaned_df,
    run_name,
    target_stage,
    model_kind="tfidf",
    min_confidence=0.0,
    model_name=None,
):
    mlflow_helpers.setup_mlflow()
    sample_pool = cleaned_df[["DESCRIPTION"]].dropna()
    if sample_pool.empty:
        input_example = pd.DataFrame([{"DESCRIPTION": "sample ticket description"}])
    else:
        input_example = sample_pool.sample(n=1, random_state=config.RANDOM_STATE)

    signature = infer_signature(input_example)
    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run found during registration")

    model_name = model_name or config.MODEL_NAME
    artifacts = {"label_encoder": str(ARTIFACT_DIR / "label_encoder.pkl")}
    if model_kind == "transformer":
        artifacts["sentence_encoder"] = str(ARTIFACT_DIR / "sentence_encoder")
    else:
        artifacts["tfidf_vectorizer"] = str(ARTIFACT_DIR / "tfidf_vectorizer.pkl")
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TicketTypeClassifier(
            tfidf_vectorizer=tfidf_vectorizer,
            label_encoder=label_encoder,
            rf_model=rf_model,
            type_name_to_id=type_name_to_id,
            model_kind=model_kind,
            min_confidence=min_confidence,
            normalize_text=True,
        ),
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature,
        artifacts=artifacts,
        code_paths=[_model_code_path()],
        pip_requirements=[f"{package}=={package_version(package)}" for package in
                          ["mlflow", "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                           "joblib", "python-dotenv", "oracledb"] +
                          (["sentence-transformers", "transformers", "torch"] if model_kind == "transformer" else [])],
    )
    run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name = '{model_name}' and run_id = '{run_id}'"
    )
    version = max(int(v.version) for v in versions) if versions else None
    if version is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage,
            archive_existing_versions=(target_stage == "Production"),
        )

    mlflow_helpers.save_run_id(run_id, str(ARTIFACT_DIR / "last_run_id.txt"))
    if model_kind == "tfidf":
        mlflow_helpers.save_run_id(run_id, str(ARTIFACT_DIR.parent / "last_run_id.txt"))
    return run_id, version


def _try_log_artifact(path: Path) -> None:
    """Best-effort artifact logging that does not fail the whole run."""
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:
        logger.warning("Artifact logging skipped for %s: %s", path, exc)


def main(argv=None):
    """Main training pipeline."""
    global ARTIFACT_DIR
    parser = argparse.ArgumentParser(description="Train EWOC TType prediction model")
    parser.add_argument("--model-kind", choices=["tfidf", "transformer"], default="tfidf")
    parser.add_argument("--evaluation-only", action="store_true", help="Train/test only; never register the model")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test share for evaluation-only runs")
    parser.add_argument("--embedding-model", default=str(Path(__file__).resolve().parents[2] / "models/sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--target-precision", type=float, default=0.90, help="Target validation accuracy among accepted predictions")
    parser.add_argument("--min-accepted", type=int, default=30, help="Minimum accepted validation examples for threshold selection")
    parser.add_argument(
        "--work-orders-table",
        default="e911.ENMT_E911_WORK_ORDERS",
        help="Oracle work orders table"
    )
    parser.add_argument(
        "--type-table",
        default="e911.LU_EWOC_TYPE",
        help="Oracle lookup table for work order type"
    )
    parser.add_argument(
        "--rca-table",
        default="e911.LU_EWOC_RCA",
        help="Oracle lookup table for RCA"
    )
    parser.add_argument(
        "--market-table",
        default="e911.LU_EWOC_MARKET",
        help="Oracle lookup table for market"
    )
    parser.add_argument(
        "--status-table",
        default="e911.LU_EWOC_STATUS",
        help="Oracle lookup table for status"
    )
    parser.add_argument(
        "--where",
        help="WHERE clause filter"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh from database"
    )
    parser.add_argument(
        "--run-name",
        help="Custom MLflow run name"
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=75,
        help="Minimum examples per class"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.65,
        help="Minimum accuracy to register model"
    )
    parser.add_argument(
        "--target-stage",
        default="Staging",
        choices=["Staging", "Production", "None"],
        help="Registry stage transition after registration"
    )

    args = parser.parse_args(argv)
    if (not 0 < args.target_precision <= 1 or args.min_accepted < 1 or args.min_class_count < 2
            or not 0 < args.test_fraction < 1):
        parser.error("Require 0 < target-precision <= 1, min-accepted >= 1, min-class-count >= 2, and 0 < test-fraction < 1")
    ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "EWOCTypePredArtifacts" / args.model_kind
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_name = config.MODEL_NAME if args.model_kind == "tfidf" else config.MODEL_NAME + "_Semantic"

    logger.info("="*70)
    logger.info("EWOC TType Training Pipeline Started")
    logger.info("="*70)

    try:
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validated")

        # Start MLflow run early so failed/below-threshold training is still visible in MLflow.
        mlflow_helpers.setup_mlflow()
        run_name = args.run_name or f"{args.model_kind}_TicketType_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {
                "test_size": config.TEST_SIZE,
                "random_state": config.RANDOM_STATE,
                "n_estimators": config.N_ESTIMATORS,
                "max_depth": config.MAX_DEPTH,
                "min_samples_split": config.MIN_SAMPLES_SPLIT,
                "min_samples_leaf": config.MIN_SAMPLES_LEAF,
                "n_jobs": config.N_JOBS,
                "pipeline": "DESCRIPTION->TFIDF->SMOTE->RandomForest" if args.model_kind == "tfidf" else "DESCRIPTION->SentenceTransformer->LogisticRegression",
                "model_kind": args.model_kind,
                "embedding_model": args.embedding_model if args.model_kind == "transformer" else "none",
                "split_policy": "chronological_85_15_no_holdout_deduplication" if args.evaluation_only else "chronological_64_16_20_at_default_test_size; normalized_holdout_deduplication",
                "evaluation_only": args.evaluation_only,
                "test_fraction": args.test_fraction if args.evaluation_only else config.TEST_SIZE,
                "text_policy": "normalize_identifiers_v1",
                "where_clause": args.where or "none",
                "work_orders_table": args.work_orders_table,
                "target_precision": args.target_precision,
                "min_accepted": args.min_accepted,
                "min_class_count": args.min_class_count,
                "min_accuracy_gate": args.min_accuracy,
            }
        )

        raw_df, type_name_to_id = _build_training_frame(
            work_orders_table=args.work_orders_table,
            type_table=args.type_table,
            rca_table=args.rca_table,
            market_table=args.market_table,
            status_table=args.status_table,
            where_clause=args.where,
            force_refresh=args.refresh,
        )

        cleaned_df = _clean_data(raw_df, min_class_count=args.min_class_count)
        if cleaned_df.empty:
            raise RuntimeError("No records left after cleaning and class filtering")

        train, validation, test = _split_frames(
            cleaned_df, args.min_class_count, args.test_fraction, args.evaluation_only)
        feature_frames = (train, test) if args.evaluation_only else (train, validation, test)
        features, labels, tfidf_vectorizer, label_encoder = _vectorize_and_encode(
            *feature_frames, model_kind=args.model_kind, embedding_model=args.embedding_model)
        if args.evaluation_only:
            X_train, X_test = features
            y_train, y_test = labels
            X_validation, y_validation = None, None
        else:
            X_train, X_validation, X_test = features
            y_train, y_validation, y_test = labels
        X_fit, y_fit = _balance_training(X_train, y_train) if args.model_kind == "tfidf" else (X_train, y_train)
        rf_model = _train_model(X_fit, y_fit, args.model_kind)
        threshold = 0.0 if args.evaluation_only else _select_threshold(
            rf_model, X_validation, y_validation, args.target_precision, args.min_accepted)
        metrics, _, y_pred = _evaluate_model(rf_model, X_test, y_test, label_encoder, threshold)
        summary = _build_training_summary(train, X_train, y_train, X_train, X_test, y_fit, y_test, metrics)
        summary["split"].update({"validation_rows": len(validation), "input_cleaned_rows": len(cleaned_df)})
        mlflow.log_params({"confidence_threshold": threshold, "feature_parameters": tfidf_vectorizer.get_params() if tfidf_vectorizer is not None else {"normalize_embeddings": True, "batch_size": 64},
                           "classifier_parameters": rf_model.get_params()})
        _write_training_diagnostics(summary, label_encoder, y_test, y_pred)

        mlflow.log_metrics({
            "dataset_cleaned_rows": summary["dataset"]["cleaned_rows"],
            "dataset_feature_columns": summary["dataset"]["feature_columns"],
            "dataset_sparsity": summary["dataset"]["sparsity"],
            "dataset_description_length_mean": summary["dataset"]["description_length_mean"],
            "dataset_description_length_p95": summary["dataset"]["description_length_p95"],
            "dataset_class_count": summary["dataset"]["class_count"],
            "dataset_class_imbalance_ratio": summary["dataset"]["class_imbalance_ratio"],
            "split_train_rows_before_smote": summary["split"]["train_rows_before_smote"],
            "split_train_rows_after_smote": summary["split"]["train_rows_after_smote"],
            "split_test_rows": summary["split"]["test_rows"],
        })
        mlflow.log_metrics(metrics)

        _try_log_artifact(ARTIFACT_DIR / "classification_report.txt")
        _try_log_artifact(ARTIFACT_DIR / "training_summary.json")
        _try_log_artifact(ARTIFACT_DIR / "class_metrics.json")
        _try_log_artifact(ARTIFACT_DIR / "test_metrics.pkl")
        _try_log_artifact(ARTIFACT_DIR / "label_mapping.pkl")

        if args.evaluation_only:
            mlflow.set_tag("registration_status", "evaluation_only")
            mlflow.end_run(status="FINISHED")
            logger.info("Evaluation complete; model was not registered.")
            return 0

        if metrics["accuracy"] < args.min_accuracy:
            mlflow.set_tag("registration_status", "skipped_below_accuracy_gate")
            mlflow.set_tag("registration_reason", f"accuracy<{args.min_accuracy}")
            mlflow.end_run(status="FINISHED")
            logger.warning(
                "Accuracy %.4f is below gate %.4f. Metrics were logged to MLflow; model registration skipped.",
                metrics["accuracy"],
                args.min_accuracy,
            )
            return 0

        run_id, version = _register_model(
            tfidf_vectorizer=tfidf_vectorizer,
            label_encoder=label_encoder,
            rf_model=rf_model,
            type_name_to_id=type_name_to_id,
            metrics=metrics,
            cleaned_df=cleaned_df,
            run_name=run_name,
            target_stage=args.target_stage,
            model_kind=args.model_kind,
            min_confidence=threshold,
            model_name=model_name,
        )

        if mlflow.active_run() is not None:
            mlflow.end_run(status="FINISHED")

        logger.info("="*70)
        logger.info("Training complete! Run ID: %s", run_id)
        logger.info("Registered model version: %s", version if version is not None else "none")
        logger.info("="*70)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
