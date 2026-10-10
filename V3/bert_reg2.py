"""Recover the best committed BERT checkpoint and register it without training.

Put this script in ewoc_ticket_type and run with its .venv Python. Keep the
trainer used for training (set --trainer accordingly). For a local CSV run,
pass --data-csv with the same CSV used for training. See --help for arguments.

Reads state.json and the original split_manifest.csv. Never writes, deletes,
renames or resumes the original checkpoints. A failed .epoch-* directory is
not a committed checkpoint and is not used. Uses the encoder and head only;
both layers of a GELU head, when present, are restored. Optimizer slots and
gradient accumulators are not allocated. Verifies source,
vocabulary, data splits, labels, and restored validation predictions before
evaluating the held-out test set. Packages the original DESCRIPTION -> list
of workOrderTypeId/workOrderType/confidence API. Creates a NEW recovery MLflow
run and a new model version in Staging if the existing metric gates pass.

Exports directly to a new short output directory, avoiding the directory
rename that failed in training. If interrupted, the new partial output can
be kept for diagnosis; the original checkpoints remain unchanged.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from uuid import uuid4

LOG = logging.getLogger("bert.recovery")


def load_trainer(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trainer not found: {path}. Set --trainer to the file used for training.")
    spec = importlib.util.spec_from_file_location("_ewoc_bert_recovery_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ("BertClassifier", "_TensorFlowBackend", "_digest_rows", "_digest_model",
                "_local_bert_tokenizer", "project_modules", "load_training_frame",
                "clean_training_frame", "validate_type_mapping", "evaluate_classifier",
                "runtime_requirements", "register_package", "write_json", "build_parser")
    if missing := [name for name in required if not hasattr(module, name)]:
        raise ValueError(f"Incompatible trainer; missing {missing}. Use the same updated trainer as the run.")
    return module


def checkpoint_details(directory):
    directory = Path(directory).expanduser().resolve()
    candidates = (directory, directory / "bert_training_state",
                  directory / "checkpoints" / "bert_training_state")
    root = next((p for p in candidates if (p / "state.json").is_file()), None)
    if root is None:
        raise FileNotFoundError(
            f"No committed state.json found under {directory}. Use the current run under "
            "your user-home/bert_runs folder, not the older EWOCBertArtifacts folder. "
            "Keep all remaining checkpoint files for inspection.")
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    sig = state["signature"]
    if sig.get("format") != 2 or sig.get("backend") != "tensorflow" or not sig.get("vocabulary_sha256"):
        raise ValueError("Checkpoint is not from the corrected TensorFlow BERT trainer.")
    best = state.get("best")
    if not isinstance(best, str) or not re.fullmatch(r"epoch_[0-9]{4}", best):
        raise ValueError("state.json does not identify a committed best epoch.")
    epoch = int(best.split("_")[1])
    if epoch != state.get("best_epoch") or not 1 <= epoch <= state.get("next_epoch", 0):
        raise ValueError("Inconsistent best-epoch metadata in state.json.")
    prefix = root / best / "training"
    if not prefix.with_suffix(".index").is_file() or not list(prefix.parent.glob("training.data-*-of-*")):
        raise FileNotFoundError(f"Best checkpoint is incomplete: {prefix}. No registration was attempted.")
    records = [row for row in state.get("history", []) if row.get("epoch") == epoch]
    if len(records) != 1 or not isinstance(records[0].get("val_macro_f1"), (float, int)):
        raise ValueError("Best checkpoint lacks its recorded validation metrics.")
    if abs(records[0]["val_macro_f1"] - state["best_score"]) > 1e-9:
        raise ValueError("Best checkpoint score is inconsistent with history.")
    LOG.info("Selected saved epoch %d; validation macro-F1 %.4f; last committed epoch %d",
             epoch, state["best_score"], state["next_epoch"])
    for file in root.glob("validation_epoch_*.json"):
        if int(file.stem.rsplit("_", 1)[1]) > state["next_epoch"]:
            LOG.warning("%s contains metrics for an uncommitted epoch; its weights are not selected.", file.name)
    return root, state, prefix, records[0]


def restore_partitions(m, frame, manifest_path, sig):
    """Use the saved row assignment, never a new random or config-based split."""
    np, pd = m.np, m.pd
    saved = pd.read_csv(manifest_path, dtype={"description_sha256": str, "group_sha256": str})
    required = {"row", "partition", "description_sha256", "group_sha256", "type_id"}
    if not required.issubset(saved) or saved[sorted(required)].isna().any().any():
        raise ValueError("The original split_manifest.csv is incomplete.")
    row_numbers = pd.to_numeric(saved["row"], errors="raise")
    if row_numbers.mod(1).ne(0).any() or row_numbers.duplicated().any():
        raise ValueError("The split manifest contains invalid or repeated row indices.")
    saved["row"] = row_numbers.astype("int64")
    if len(saved) != len(frame) or set(saved["row"]) != set(frame.index):
        raise ValueError("Current cleaned data differs from the saved split. Restore the original cached CSVs.")
    partitions = set(saved["partition"])
    required_partitions = {"train", "validation", "test"}
    if not required_partitions.issubset(partitions) or partitions - (required_partitions | {"train_excluded_by_cap"}):
        raise ValueError("Expected train, validation, test, and optional train_excluded_by_cap in the manifest.")
    aligned = frame.loc[saved["row"]]
    sha = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    checks = (aligned[m.TEXT_COLUMN].map(sha).to_numpy() == saved["description_sha256"].to_numpy(),
              aligned[m.GROUP_COLUMN].map(sha).to_numpy() == saved["group_sha256"].to_numpy(),
              aligned["TYPE_ID"].to_numpy() == pd.to_numeric(saved["type_id"], errors="raise").to_numpy())
    if not all(check.all() for check in checks):
        raise ValueError("Data, type IDs, or normalization changed since training. Restore the original cached data.")
    parts = {name: frame.loc[saved.loc[saved["partition"].eq(name), "row"]].copy()
             for name in ("train", "validation", "test")}
    classes, targets = np.unique(parts["train"][m.LABEL_COLUMN].to_numpy(dtype=str), return_inverse=True)
    if classes.tolist() != sig["classes"]:
        raise ValueError("Training labels differ from the checkpoint's class order.")
    weights = np.ones(len(targets), dtype=np.float32)
    if sig["settings"]["class_weight"] == "balanced":
        counts = np.bincount(targets)
        weights = len(targets) / (len(counts) * counts[targets])
    elif sig["settings"]["class_weight"] is not None:
        raise ValueError("Unsupported checkpoint class-weight scheme.")
    weights = np.asarray(weights / weights.mean(), dtype=np.float32)
    for name in ("train", "validation"):
        part = parts[name]
        digest = m._digest_rows(part[m.TEXT_COLUMN].map(m.bert_text).tolist(),
                                part[m.LABEL_COLUMN].to_numpy(dtype=str), weights if name == "train" else None)
        if digest != sig[name]:
            raise ValueError(f"{name} text/labels/weights differ from the saved checkpoint signature.")
    for i, name in enumerate(parts):
        for other in list(parts)[i + 1:]:
            if set(parts[name][m.GROUP_COLUMN]) & set(parts[other][m.GROUP_COLUMN]):
                raise ValueError("The saved partitions contain overlapping description groups.")
    excluded = frame.loc[saved.loc[saved["partition"].eq("train_excluded_by_cap"), "row"]]
    if set(excluded[m.GROUP_COLUMN]) & (set(parts["validation"][m.GROUP_COLUMN]) | set(parts["test"][m.GROUP_COLUMN])):
        raise ValueError("Rows excluded by the training cap overlap held-out description groups.")
    LOG.info("Original split verified: train=%d validation=%d test=%d", *(len(p) for p in parts.values()))
    return parts


def restore_classifier(m, state, prefix, args):
    sig = state["signature"]
    source = Path(args.bert_dir or sig["source"]).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Original pretrained model not found: {source}; set --bert-dir if moved.")
    if m._digest_model(source) != sig["source_fingerprint"]:
        raise ValueError("Pretrained model bytes differ from the training source.")
    options = dict(sig["settings"])
    options.update(batch_size=args.batch_size, threads=args.threads)
    classifier = m.BertClassifier(model_name=str(source), **options)
    classifier._validate_settings()
    classifier.classes_ = m.np.asarray(sig["classes"], dtype=str)
    classifier.backend_ = "tensorflow"
    # This initializes the encoder and head only. Never configure optimizers or call fit.
    classifier._backend = m._TensorFlowBackend(classifier, training=True)
    classifier.model, classifier.device = classifier._backend.model, classifier._backend.device
    vocab_hash = hashlib.sha256(json.dumps(classifier.tokenizer.get_vocab(), sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
    if vocab_hash != sig["vocabulary_sha256"]:
        classifier.close()
        raise ValueError("Vocabulary IDs do not match the saved training signature.")
    tf = classifier._backend.tf
    if hasattr(classifier._backend, "model_checkpoint_items"):
        model_items = classifier._backend.model_checkpoint_items()
    elif getattr(classifier, "head_hidden_size", 0):
        raise ValueError("This trainer cannot restore all hidden-head weights. Use the trainer from this run.")
    else:
        model_items = {"encoder": classifier.model.encoder,
                       "kernel": classifier.model.kernel, "bias": classifier.model.bias}
    checkpoint = tf.train.Checkpoint(**model_items)
    status = checkpoint.read(str(prefix))
    # Extra optimizer tensors are expected; every current model variable MUST match.
    status.expect_partial()
    status.assert_nontrivial_match()
    status.assert_existing_objects_matched()
    del status, checkpoint
    classifier.source_fingerprint_ = sig["source_fingerprint"]
    classifier.history_ = state["history"]
    classifier.best_epoch_ = state["best_epoch"]
    classifier.best_validation_macro_f1_ = state["best_score"]
    classifier.stopped_early_ = state.get("stopped_early", False)
    classifier.training_text_counts = {}
    classifier.fitted_ = True
    LOG.info("Restored encoder and classification head; no optimizer state allocated and no training performed.")
    return classifier


def package_recovered(m, classifier, mapping, output):
    """Direct export to a unique directory: no checkpoint/directory rename."""
    probe = m.pd.DataFrame({m.TEXT_COLUMN: ["Please investigate this work order.", "Scheduled network maintenance."]})
    p = classifier.predict_proba(probe[m.TEXT_COLUMN].tolist())
    names = classifier.classes_[p.argmax(axis=1)].tolist()
    expected = [{"workOrderTypeId": int(mapping[name]), "workOrderType": name,
                 "confidence": float(p[i].max())} for i, name in enumerate(names)]
    export = output / "classifier"
    export.mkdir(exist_ok=False)
    classifier._backend.export(export)
    classifier.tokenizer.save_pretrained(str(export))
    m.write_json(export / "classifier_settings.json", {
        "format_version": m.FORMAT_VERSION, "backend": classifier.backend_,
        "source": classifier.model_name, "source_fingerprint": classifier.source_fingerprint_,
        "classes": classifier.classes_.tolist(), "training_text_counts": classifier.training_text_counts,
        "history": classifier.history_, "best_epoch": classifier.best_epoch_,
        "best_validation_macro_f1": classifier.best_validation_macro_f1_,
        "stopped_early": classifier.stopped_early_, **classifier._settings()})
    classifier.close()
    mapping_path = output / "type_name_to_id.json"
    m.write_json(mapping_path, mapping)
    requirements = m.runtime_requirements()
    (output / "requirements-serving.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    source_text = Path(m.__file__).read_text(encoding="utf-8").rsplit('\nif __name__ == "__main__":', 1)[0]
    inference = output / "bert_inference.py"
    inference.write_text(source_text + "\nmlflow.models.set_model(TicketTypeBertModel())\n", encoding="utf-8")
    package = output / "mlflow_model"
    m.mlflow.pyfunc.save_model(path=str(package), python_model=str(inference),
                             artifacts={"classifier": str(export), "type_name_to_id": str(mapping_path)},
                             signature=m.infer_signature(probe, m.pd.DataFrame(expected)),
                             input_example=probe, pip_requirements=requirements)
    restored = m.mlflow.pyfunc.load_model(str(package))
    try:
        observed = restored.predict(probe)
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError("Reloaded MLflow model returned an incompatible response.")
        for actual, wanted in zip(observed, expected):
            if set(actual) != set(wanted) or any(actual[key] != wanted[key] for key in ("workOrderTypeId", "workOrderType")):
                raise RuntimeError("MLflow reload changed the API output fields or type mapping.")
            m.np.testing.assert_allclose(actual["confidence"], wanted["confidence"], atol=1e-6, rtol=1e-6)
        if restored.predict(m.pd.DataFrame({m.TEXT_COLUMN: m.pd.Series(dtype=str)})) != []:
            raise RuntimeError("MLflow empty-input response is incompatible.")
    finally:
        restored.unwrap_python_model().classifier.close()
    LOG.info("API package reload passed: type IDs, names, confidence, and empty input.")
    return package


def recover(m, args):
    root, state, prefix, best_record = checkpoint_details(args.checkpoint_dir)
    if args.inspect_only:
        print(json.dumps({"selected_epoch": state["best_epoch"], "checkpoint": str(prefix),
                          "validation": best_record}, indent=2))
        return 0
    manifest = Path(args.split_manifest).resolve() if args.split_manifest else root.parent.parent / "split_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Original split manifest missing: {manifest}; supply --split-manifest.")
    config, loader, helpers = m.project_modules()
    original = m.build_parser().parse_args([])
    original.data_csv = args.data_csv
    original.refresh = False
    for name in ("work_orders_table", "type_table", "rca_table", "market_table", "status_table"):
        if getattr(args, name):
            setattr(original, name, getattr(args, name))
    raw = m.load_training_frame(original, loader)
    frame = m.clean_training_frame(raw, loader, args.min_class_count)
    del raw
    parts = restore_partitions(m, frame, manifest, state["signature"])
    mapping = m.validate_type_mapping(frame)
    if set(mapping) != set(state["signature"]["classes"]):
        raise ValueError("Type-name mapping differs from checkpoint classes.")
    base = Path(args.output_dir).expanduser() if args.output_dir else Path.home() / "bert_recover"
    output = base.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Recovery output must be separate from the source checkpoint directory.")
    output.mkdir(parents=True, exist_ok=False)
    LOG.info("Recovery output: %s", output)
    classifier = None
    try:
        classifier = restore_classifier(m, state, prefix, args)
        encoded, counts = classifier._tokenize(parts["train"][m.TEXT_COLUMN].map(m.bert_text).tolist())
        classifier.training_text_counts = counts
        del encoded
        val_metrics, _, val_by_class, val_confusion = m.evaluate_classifier(classifier, parts["validation"], mapping)
        for current, recorded in (("accuracy", "val_accuracy"), ("macro_f1", "val_macro_f1"),
                                  ("weighted_f1", "val_weighted_f1")):
            if abs(val_metrics[current] - best_record[recorded]) > 1e-4:
                raise ValueError(f"Restored validation {current} differs from the saved epoch: "
                                 f"{val_metrics[current]:.6f} vs {best_record[recorded]:.6f}. Registration stopped.")
        m.write_json(output / "validation_class_metrics.json", val_by_class)
        val_confusion.to_csv(output / "validation_confusion_matrix.csv", index_label="actual")
        metrics, report, by_class, confusion = m.evaluate_classifier(classifier, parts["test"], mapping)
        m.write_json(output / "test_metrics.json", metrics)
        m.write_json(output / "class_metrics.json", by_class)
        (output / "classification_report.txt").write_text(report, encoding="utf-8")
        confusion.to_csv(output / "confusion_matrix.csv", index_label="actual")
        summary = {"recovered_epoch": state["best_epoch"], "source_checkpoint": str(prefix),
                   "source_state": str(root / "state.json"), "source_signature": state["signature"],
                   "training_performed": False, "original_split_verified": True,
                   "validation": val_metrics, "test": metrics, "history": state["history"]}
        m.write_json(output / "recovery_summary.json", summary)
        LOG.info("Recovered epoch %d: TEST accuracy=%.4f macro-F1=%.4f weighted-F1=%.4f",
                 state["best_epoch"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
        package = package_recovered(m, classifier, mapping, output)
    finally:
        if classifier is not None:
            classifier.close()
    # Do network setup after local export: local outputs survive an MLflow outage.
    helpers.setup_mlflow()
    if args.tracking_uri:
        m.mlflow.set_tracking_uri(args.tracking_uri)
        m.mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    if m.mlflow.active_run() is not None:
        raise RuntimeError("Launch recovery in a fresh Python process without an active MLflow run.")
    passed = metrics["accuracy"] >= args.min_accuracy and metrics["macro_f1"] >= args.min_macro_f1
    with m.mlflow.start_run(run_name=args.run_name) as run:
        m.mlflow.set_tags({"model_family": "tensorflow_bert_finetuned", "recovery_only": "true",
                           "classification_target": m.LABEL_COLUMN, "registration_status": "recovering"})
        m.mlflow.log_params({**state["signature"]["settings"], "recovered_epoch": state["best_epoch"],
                             "source_checkpoint": str(prefix), "inference_batch_size": args.batch_size,
                             "target_stage": args.target_stage, "min_accuracy_gate": args.min_accuracy,
                             "min_macro_f1_gate": args.min_macro_f1})
        m.mlflow.log_metrics(metrics)
        for record in state["history"]:
            m.mlflow.log_metrics({k: float(v) for k, v in record.items() if k != "epoch"}, step=record["epoch"])
        for file in output.iterdir():
            if file.is_file():
                m.mlflow.log_artifact(str(file), artifact_path="recovery")
        result = {"model_name": args.model_name, "recovered_epoch": state["best_epoch"],
                  "run_id": run.info.run_id, "artifact_dir": str(output), "metrics": metrics}
        if passed and not args.no_register:
            version, uri = m.register_package(package, args.model_name, args.target_stage)
            result.update(status="registered", version=version, stage=args.target_stage,
                          model_uri=f"models:/{args.model_name}/{version}", run_model_uri=uri)
        else:
            m.mlflow.log_artifacts(str(package), artifact_path="model")
            result["status"] = "not_registered_by_request" if args.no_register else "skipped_below_metric_gate"
        m.write_json(output / "registration.json", result)
        m.mlflow.log_artifact(str(output / "registration.json"), artifact_path="recovery")
        m.mlflow.set_tag("registration_status", result["status"])
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "skipped_below_metric_gate" else 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainer", default="src/ewoc_ttype/bert_train4.py")
    p.add_argument("--checkpoint-dir", required=True, help="Original run, checkpoints, or bert_training_state directory")
    p.add_argument("--split-manifest", help="Original split_manifest.csv if checkpoint directory was relocated")
    p.add_argument("--bert-dir", help="Original models/bert_tf if relocated; contents must match")
    p.add_argument("--data-csv", help="Original enriched data, if used instead of project cached data")
    p.add_argument("--min-class-count", type=int, default=75, help="Must match the original run")
    p.add_argument("--batch-size", type=int, default=2, help="Inference only; small default to reduce memory")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output-dir", help="Parent directory for NEW output; default ~/bert_recover")
    p.add_argument("--model-name", default="EWOC_TicketType_BERT")
    p.add_argument("--target-stage", choices=["Staging", "None"], default="Staging")
    p.add_argument("--run-name", default="BERT_v2_recovered")
    p.add_argument("--tracking-uri")
    p.add_argument("--min-accuracy", type=float, default=.65)
    p.add_argument("--min-macro-f1", type=float, default=0.)
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--inspect-only", action="store_true", help="Print the saved epoch and exit without loading model/data or registering")
    for flag in ("work-orders-table", "type-table", "rca-table", "market-table", "status-table"):
        p.add_argument("--" + flag, help="Override only if the original run used a nondefault table")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("azure").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    if min(args.batch_size, args.threads, args.min_class_count) < 1:
        raise ValueError("batch-size, threads and min-class-count must be positive.")
    if not args.model_name.strip() or not all(0 <= x <= 1 for x in (args.min_accuracy, args.min_macro_f1)):
        raise ValueError("Provide a nonempty model name and metric gates between 0 and 1.")
    # Force local model access; this script needs no Hugging Face downloads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return recover(load_trainer(args.trainer), args)
    except Exception:
        LOG.exception("Recovery failed. Original checkpoints were not modified.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
