"""Fine-tune local TensorFlow BERT and register an API-compatible MLflow model.

Place this file at src/ewoc_ttype/bert_train.py and run from ewoc_ticket_type:
  .\\.venv\\Scripts\\python.exe .\\src\\ewoc_ttype\\bert_train.py --target-stage Staging

Defaults: models/bert_tf, five epochs, new registry name EWOC_TicketType_BERT.
The default stage is Staging. After registration succeeds, set config.MODEL_NAME to
"EWOC_TicketType_BERT" and restart an API that loads that name through MLflow
pyfunc at the same stage. A pinned version or alias must also match. No API
loader code was supplied, so its configuration lookup needs confirmation.

The existing 0.65 accuracy registration gate is retained. A below-gate run keeps
its trained model and metrics but returns exit code 2 without registering.
Artifacts go to EWOCBertArtifacts/<timestamp>_<run-id>; existing RF files remain
in their own location. Each successful registration gets a new MLflow version.
To resume an interrupted run, repeat the command and settings with
--checkpoint-dir "<the checkpoint folder printed by the interrupted run>".
Checkpoints resume at completed epoch boundaries.

Uses the project's ewoc_utils config, data_loader, and mlflow_helpers for Oracle
and MLflow. These project modules are needed for training only, not inference.
No Random Forest, TF-IDF model or MiniLM is trained. No model download is needed.
The TensorFlow encoder AND a new classification head receive gradients.
Requires MLflow >= 2.12.2 for source-based inference packaging.
Integration-tested with TensorFlow 2.21.0, Transformers 4.57.1 and MLflow 2.22.2
using a small real TensorFlow encoder and a local MLflow registry. The actual
Kaggle checkpoint, Oracle data and deployed API are not available in this test.

MLflow predict contract:
  input: pandas DataFrame containing DESCRIPTION
  output: list of dicts with workOrderTypeId, workOrderType, confidence

Run --help for options. The exported model includes its vocabulary, fine-tuned
weights, label mapping and inference code. TensorFlow must also be installed in
an API process that calls mlflow.pyfunc.load_model (MLflow does not install it).
The run writes requirements-serving.txt with the versions used for its model.
Confidence is the maximum softmax score, not a calibrated correctness rate.

The supplied RF trainer also reads config.MODEL_NAME. If you run that trainer
again after switching the shared config to BERT, give RF its own registry name
first so it does not register a Random Forest under the BERT name.

Selection uses a grouped validation set within the outer training partition;
the outer test partition is reserved for final evaluation. The outer split
matches the supplied RF code's GroupShuffleSplit when data/order/normalization,
TEST_SIZE and RANDOM_STATE are identical. Results do not guarantee superiority.
"""

import hashlib
import inspect
import json
import logging
import math
import re
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

logger = logging.getLogger("ewoc.bert_train")
FORMAT_VERSION = 2


def _texts(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("Pass a sequence of texts, not one string.")
    result = list(values)
    if any(not isinstance(x, str) for x in result):
        raise ValueError("Every text must be a string. Fill missing values before fit/predict.")
    return result


def _labels(values, count):
    if isinstance(values, (str, bytes)):
        raise ValueError("Labels must be a sequence.")
    values = list(values)
    if len(values) != count or any(x is None or isinstance(x, (float, np.floating))
                                  and not np.isfinite(x) for x in values):
        raise ValueError("Labels must be nonmissing and match the number of texts.")
    result = np.asarray(values, dtype=str)
    if result.ndim != 1 or any(not x.strip() for x in result):
        raise ValueError("Labels must be a one-dimensional sequence of nonempty values.")
    return result


def _source(model_name):
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    # Also support running src/ewoc_ttype/main_train*.py from another directory.
    if not path.is_absolute():
        for parent in Path(__file__).resolve().parents:
            candidate = parent / path
            if candidate.is_dir():
                return str(candidate.resolve())
    if path.is_absolute() or str(model_name).startswith((".", "models/", "models\\")):
        raise FileNotFoundError(f"Local model directory does not exist: {model_name}")
    return str(model_name)


def _digest_rows(texts, labels, weights=None):
    digest = hashlib.sha256()
    for i, (text, label) in enumerate(zip(texts, labels)):
        row = [text, str(label)]
        if weights is not None:
            row.append(float(weights[i]))
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _digest_model(source):
    path = Path(source)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    patterns = ("config.json", "*.safetensors", "pytorch_model*.bin", "*.index.json",
                "tokenizer*.json", "vocab.txt", "special_tokens_map.json", "saved_model.pb",
                "variables/*", "assets/*")
    files = sorted({p for pattern in patterns for p in path.glob(pattern) if p.is_file()})
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class ClauseSelector:
    @staticmethod
    def head_tail(token_ids, budget):
        if len(token_ids) <= budget:
            return list(token_ids)
        head = (budget + 1) // 2
        tail = budget - head
        return list(token_ids[:head]) + (list(token_ids[-tail:]) if tail else [])


def _lr_scale(step, total, warmup_ratio):
    warmup = int(total * warmup_ratio)
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return max(0.0, (total - step) / max(total - warmup, 1))


class _TensorFlowBackend:
    def __init__(self, owner, training=True):
        import tensorflow as tf
        from transformers import BertTokenizer
        self.owner, self.tf = owner, tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(owner.threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            logger.debug("TensorFlow already initialized; retaining its thread settings.")
        tf.random.set_seed(owner.seed)
        self.device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
        path = Path(owner.model_name)
        if not training:
            self.model = tf.saved_model.load(str(path / "tf_model"))
            owner.tokenizer = BertTokenizer.from_pretrained(str(path), local_files_only=True)
            return
        owner.tokenizer = BertTokenizer(vocab_file=str(path / "assets" / "vocab.txt"), do_lower_case=True)
        encoder = tf.saved_model.load(str(path))
        if not getattr(encoder, "trainable_variables", []):
            raise ValueError("This SavedModel exposes no trainable variables; full fine-tuning is unavailable.")
        dummy = {key: tf.zeros([1, 4], tf.int32)
                 for key in ("input_word_ids", "input_mask", "input_type_ids")}
        dummy["input_mask"] = tf.ones([1, 4], tf.int32)
        example = encoder(dummy, training=False)
        if "pooled_output" not in example:
            raise ValueError("Expected a BERT SavedModel returning pooled_output.")
        hidden = int(example["pooled_output"].shape[-1])

        class TicketModel(tf.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.kernel = tf.Variable(tf.random.truncated_normal(
                    [hidden, len(owner.classes_)], stddev=0.02, seed=owner.seed), name="classifier_kernel")
                self.bias = tf.Variable(tf.zeros([len(owner.classes_)]), name="classifier_bias")

            @tf.function(input_signature=[{
                key: tf.TensorSpec([None, None], tf.int32, name=key)
                for key in ("input_word_ids", "input_mask", "input_type_ids")
            }])
            def __call__(self, inputs):
                pooled = self.encoder(inputs, training=False)["pooled_output"]
                return {"logits": tf.matmul(pooled, self.kernel) + self.bias}

        self.model = TicketModel()

    def configure(self):
        tf, owner = self.tf, self.owner
        self.encoder_vars = list(self.model.encoder.trainable_variables)
        self.head_vars = [self.model.kernel, self.model.bias]
        self.variables = self.encoder_vars + self.head_vars
        self.encoder_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=owner.learning_rate, weight_decay=owner.weight_decay)
        self.head_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=owner.head_learning_rate, weight_decay=owner.weight_decay)
        for optimizer, variables in ((self.encoder_optimizer, self.encoder_vars),
                                     (self.head_optimizer, self.head_vars)):
            optimizer.exclude_from_weight_decay(var_list=[v for v in variables if len(v.shape) < 2])
            optimizer.build(variables)
        self.accumulators = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.variables]
        self.active_indices = None
        self.checkpoint = tf.train.Checkpoint(
            encoder=self.model.encoder, kernel=self.model.kernel, bias=self.model.bias,
            encoder_optimizer=self.encoder_optimizer, head_optimizer=self.head_optimizer)

        @tf.function(reduce_retracing=True)
        def gradients(inputs, targets, weights, divisor):
            with tf.GradientTape() as tape:
                pooled = self.model.encoder(inputs, training=True)["pooled_output"]
                pooled = tf.nn.dropout(pooled, rate=owner.dropout, seed=owner.seed)
                logits = tf.matmul(pooled, self.model.kernel) + self.model.bias
                if owner.label_smoothing:
                    labels = tf.one_hot(targets, len(owner.classes_))
                    losses = tf.keras.losses.categorical_crossentropy(
                        labels, logits, from_logits=True, label_smoothing=owner.label_smoothing)
                else:
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                weighted_sum = tf.reduce_sum(losses * weights)
                loss = weighted_sum / divisor
                tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
            grads = tape.gradient(loss, self.variables)
            active = [i for i, gradient in enumerate(grads) if gradient is not None]
            if not any(i < len(self.encoder_vars) for i in active):
                raise ValueError("No encoder gradients: this SavedModel cannot be fine-tuned through pooled_output.")
            if not all(i in active for i in range(len(self.encoder_vars), len(self.variables))):
                raise ValueError("Classification head is disconnected from the loss.")
            self.active_indices = active
            for i in active:
                self.accumulators[i].assign_add(tf.convert_to_tensor(grads[i]))
            return weighted_sum

        self.gradients = gradients

    def start_epoch(self, epoch):
        self.tf.random.set_seed(self.owner.seed + epoch)

    def batch(self, features):
        padded = self.owner.tokenizer.pad(features, padding=True, return_tensors="np")
        return {target: self.tf.constant(padded[source], dtype=self.tf.int32)
                for source, target in (("input_ids", "input_word_ids"),
                                       ("attention_mask", "input_mask"),
                                       ("token_type_ids", "input_type_ids"))}

    def accumulate(self, features, targets, weights, divisor):
        tf = self.tf
        return float(self.gradients(self.batch(features), tf.constant(targets, tf.int32),
                                   tf.constant(weights, tf.float32), tf.constant(divisor, tf.float32)).numpy())

    def update(self, scale):
        tf = self.tf
        gradients, _ = tf.clip_by_global_norm([self.accumulators[i] for i in self.active_indices], 1.0)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite training gradient")
        split = len(self.encoder_vars)
        encoder_pairs = [(g, self.variables[i]) for g, i in zip(gradients, self.active_indices) if i < split]
        head_pairs = [(g, self.variables[i]) for g, i in zip(gradients, self.active_indices) if i >= split]
        self.encoder_optimizer.learning_rate.assign(self.owner.learning_rate * scale)
        self.head_optimizer.learning_rate.assign(self.owner.head_learning_rate * scale)
        self.encoder_optimizer.apply_gradients(encoder_pairs)
        self.head_optimizer.apply_gradients(head_pairs)
        for accumulator in self.accumulators:
            accumulator.assign(tf.zeros_like(accumulator))

    def logits(self, features):
        return self.model(self.batch(features))["logits"].numpy()

    def save_state(self, directory):
        self.checkpoint.write(str(directory / "training"))

    def load_state(self, directory):
        self.checkpoint.read(str(directory / "training")).assert_consumed()

    def export(self, directory):
        self.tf.saved_model.save(self.model, str(directory / "tf_model"),
                                 signatures={"serving_default": self.model.__call__.get_concrete_function()})


class BertClassifier:
    def __init__(self, model_name="models/bert_tf", epochs=3, learning_rate=2e-5,
                 seed=42, threads=4, cache_dir=None, revision="main",
                 head_learning_rate=None, *, max_length=256, batch_size=8, accumulation=4,
                 weight_decay=0.01, warmup_ratio=0.1, dropout=0.1, patience=2,
                 min_delta=1e-4, class_weight=None, label_smoothing=0.0):
        self.model_name = str(model_name)
        self.epochs, self.learning_rate = epochs, learning_rate
        self.seed, self.threads = seed, threads
        self.max_length, self.batch_size, self.accumulation = max_length, batch_size, accumulation
        self.head_learning_rate = 1e-4 if head_learning_rate is None else head_learning_rate
        self.weight_decay, self.warmup_ratio, self.dropout = weight_decay, warmup_ratio, dropout
        self.patience, self.min_delta = patience, min_delta
        self.class_weight, self.label_smoothing = class_weight, label_smoothing
        self.cache_dir, self.revision = str(cache_dir) if cache_dir is not None else None, revision
        self.load_options = {"revision": revision}
        if cache_dir is not None:
            self.load_options["cache_dir"] = str(cache_dir)
        self.name = "BERT"
        self.selector = None
        self._backend = None
        self.fitted_ = False

    def _settings(self):
        names = ("epochs", "learning_rate", "seed", "threads", "max_length", "batch_size", "accumulation",
                 "head_learning_rate", "weight_decay", "warmup_ratio", "dropout",
                 "patience", "min_delta", "class_weight", "label_smoothing")
        return {name: (getattr(self, name).item() if isinstance(getattr(self, name), np.generic)
                       else getattr(self, name)) for name in names}

    def _validate_settings(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer between 0 and 2**31 - 1.")
        for name in ("epochs", "threads", "max_length", "batch_size", "accumulation", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 4 <= self.max_length <= 512:
            raise ValueError("BERT max_length must be between 4 and 512.")
        for name in ("learning_rate", "head_learning_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("warmup_ratio", "dropout", "label_smoothing"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative.")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")

    def _tokenize(self, texts):
        result, counts = [], {"full": 0, "clauses": 0, "head_tail": 0}
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        for offset in range(0, len(texts), 256):
            chunk = texts[offset:offset + 256]
            raw = self.tokenizer(chunk, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            for text, ids in zip(chunk, raw):
                if len(ids) > budget:
                    ids, mode = ClauseSelector.head_tail(ids, budget), "head_tail"
                else:
                    mode = "full"
                counts[mode] += 1
                item = self.tokenizer.prepare_for_model(ids, return_attention_mask=True, return_token_type_ids=True)
                if len(item["input_ids"]) > self.max_length:
                    raise ValueError("Text selection exceeded the token budget.")
                result.append(item)
        return result, counts

    def fit(self, texts, labels, sample_weight=None, checkpoint_dir=None, validation_data=None):
        self._validate_settings()
        texts = _texts(texts)
        labels = _labels(labels, len(texts))
        self.classes_, targets = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Sequence classification needs at least two training classes.")
        weights = np.ones(len(texts), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        if weights.shape != (len(texts),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Training weights must be finite, positive and match training rows.")
        if self.class_weight == "balanced":
            if sample_weight is not None:
                raise ValueError("Use either sample_weight or class_weight='balanced', to avoid double weighting.")
            counts = np.bincount(targets)
            weights = len(targets) / (len(counts) * counts[targets])
        weights = np.asarray(weights / weights.mean(), dtype=np.float32)
        val_texts, val_labels = [], []
        if validation_data is not None:
            if len(validation_data) != 2:
                raise ValueError("validation_data must be (texts, labels).")
            val_texts = _texts(validation_data[0])
            val_labels = _labels(validation_data[1], len(val_texts))
            if not val_texts or not set(val_labels).issubset(set(self.classes_)):
                raise ValueError("Validation must be nonempty, with classes present in training.")
            missing = sorted(set(self.classes_) - set(val_labels))
            if missing:
                logger.warning("Validation is missing %d training classes; macro-F1 will include them as zero.", len(missing))
            normalize = lambda text: " ".join(text.lower().split())
            overlap = {normalize(t) for t in texts}.intersection(normalize(t) for t in val_texts)
            overlap.discard("")
            if overlap:
                raise ValueError(f"{len(overlap)} normalized texts overlap training/validation. Split duplicate tickets together.")
        else:
            logger.warning("No validation_data supplied: no early stopping; retaining the final epoch.")
        if self._backend is not None:
            self.close()
        self.fitted_ = False
        self.model_name = _source(self.model_name)
        self.load_options["local_files_only"] = Path(self.model_name).is_dir()
        if not (Path(self.model_name) / "saved_model.pb").is_file():
            raise FileNotFoundError("Use the extracted local TensorFlow SavedModel directory.")
        self.backend_ = "tensorflow"
        self._backend = _TensorFlowBackend(self)
        self.model, self.device = self._backend.model, self._backend.device
        self.selector = None
        encoded, self.training_text_counts = self._tokenize(texts)
        val_encoded, _ = self._tokenize(val_texts)
        self._backend.configure()
        self.source_fingerprint_ = _digest_model(self.model_name)
        signature = {"format": FORMAT_VERSION, "backend": self.backend_, "source": self.model_name,
                     "revision": self.revision, "source_fingerprint": self.source_fingerprint_,
                     "settings": self._settings(), "classes": self.classes_.tolist(),
                     "train": _digest_rows(texts, labels, weights),
                     "validation": _digest_rows(val_texts, val_labels)}
        batches = math.ceil(len(targets) / self.batch_size)
        total_steps = self.epochs * math.ceil(batches / self.accumulation)
        temporary = tempfile.TemporaryDirectory(prefix="bert-training-") if checkpoint_dir is None else None
        root = Path(temporary.name) if temporary else Path(checkpoint_dir) / "bert_training_state"
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "state.json"
        state = {"signature": signature, "next_epoch": 0, "global_step": 0, "best_score": None,
                 "best_epoch": 0, "bad_epochs": 0, "history": [], "latest": None, "best": None,
                 "stopped_early": False}
        try:
            if manifest.exists():
                state = json.loads(manifest.read_text(encoding="utf-8"))
                if state["signature"] != signature:
                    raise ValueError("Checkpoint differs in data, weights, model or configuration. Use a new checkpoint_dir.")
                self._backend.load_state(root / state["latest"])
            logger.info("%s [%s/%s]: %d rows, %d classes, max_length=%d, effective batch=%d, selection=%s",
                        self.name, self.backend_, self.device, len(texts), len(self.classes_), self.max_length,
                        self.batch_size * self.accumulation, self.training_text_counts)
            for epoch in range(state["next_epoch"], self.epochs):
                if state["stopped_early"]:
                    break
                order = np.random.default_rng(self.seed + epoch).permutation(len(targets))
                self._backend.start_epoch(epoch)
                started, last_log, loss_sum, seen = perf_counter(), perf_counter(), 0.0, 0
                for step, offset in enumerate(range(0, len(order), self.batch_size)):
                    indices = order[offset:offset + self.batch_size]
                    group_start = (step // self.accumulation) * self.accumulation * self.batch_size
                    group_size = min(self.accumulation * self.batch_size, len(targets) - group_start)
                    loss_sum += self._backend.accumulate([encoded[i] for i in indices], targets[indices],
                                                        weights[indices], group_size)
                    seen += len(indices)
                    if (step + 1) % self.accumulation == 0 or step + 1 == batches:
                        self._backend.update(_lr_scale(state["global_step"], total_steps, self.warmup_ratio))
                        state["global_step"] += 1
                    if perf_counter() - last_log >= 30 or step + 1 == batches:
                        elapsed = perf_counter() - started
                        logger.info("%s epoch %d/%d batch %d/%d loss %.4f ETA %.1f min", self.name,
                                    epoch + 1, self.epochs, step + 1, batches, loss_sum / seen,
                                    elapsed / (step + 1) * (batches - step - 1) / 60)
                        last_log = perf_counter()
                record = {"epoch": epoch + 1, "train_loss": loss_sum / seen,
                          "seconds": perf_counter() - started}
                if val_encoded:
                    probabilities = self._probabilities(val_encoded)
                    predicted = self.classes_[probabilities.argmax(axis=1)]
                    score = float(f1_score(val_labels, predicted, labels=self.classes_, average="macro", zero_division=0))
                    record.update(val_macro_f1=score,
                                  val_weighted_f1=float(f1_score(val_labels, predicted, average="weighted", zero_division=0)),
                                  val_accuracy=float(accuracy_score(val_labels, predicted)),
                                  val_log_loss=float(log_loss(val_labels, probabilities, labels=self.classes_)))
                    improved = state["best_score"] is None or score > state["best_score"] + self.min_delta
                    logger.info("Validation: macro-F1 %.4f, weighted-F1 %.4f, accuracy %.4f", score,
                                record["val_weighted_f1"], record["val_accuracy"])
                else:
                    score, improved = None, True
                state["history"].append(record)
                folder_name = f"epoch_{epoch + 1:04d}"
                stage = Path(tempfile.mkdtemp(prefix=".epoch-", dir=root))
                try:
                    self._backend.save_state(stage)
                    destination = root / folder_name
                    if destination.exists():
                        shutil.rmtree(destination)  # Uncommitted epoch from an interrupted write.
                    stage.replace(destination)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
                if improved:
                    state.update(best=folder_name, best_epoch=epoch + 1, best_score=score, bad_epochs=0)
                else:
                    state["bad_epochs"] += 1
                state.update(next_epoch=epoch + 1, latest=folder_name,
                             stopped_early=bool(val_encoded and state["bad_epochs"] >= self.patience))
                pending = root / "state.pending.json"
                pending.write_text(json.dumps(state, indent=2), encoding="utf-8")
                pending.replace(manifest)
                for old in root.glob("epoch_[0-9][0-9][0-9][0-9]"):
                    if old.is_dir() and old.name not in (state["latest"], state["best"]):
                        shutil.rmtree(old)
            self._backend.load_state(root / state["best"])
            self.history_, self.best_epoch_ = state["history"], state["best_epoch"]
            self.best_validation_macro_f1_ = state["best_score"]
            self.stopped_early_ = state["stopped_early"]
            self.fitted_ = True
            logger.info("%s ready: retained epoch %d; validation macro-F1=%s", self.name,
                        self.best_epoch_, self.best_validation_macro_f1_)
            return self
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _probabilities(self, encoded):
        rows = []
        for offset in range(0, len(encoded), self.batch_size):
            logits = self._backend.logits(encoded[offset:offset + self.batch_size]).astype(np.float64)
            if not np.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction logits.")
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            rows.append(probabilities / probabilities.sum(axis=1, keepdims=True))
        return np.concatenate(rows) if rows else np.empty((0, len(self.classes_)))

    def predict_proba(self, texts):
        """Columns follow classes_; softmax scores are not guaranteed calibrated."""
        if not self.fitted_:
            raise ValueError("Fit or load the classifier before prediction.")
        encoded, _ = self._tokenize(_texts(texts))
        return self._probabilities(encoded)

    def predict(self, texts):
        probabilities = self.predict_proba(texts)
        if not len(probabilities):
            return np.asarray([], dtype=self.classes_.dtype)
        return self.classes_[probabilities.argmax(axis=1)]

    def save(self, directory):
        if not self.fitted_:
            raise ValueError("Fit or load the classifier before saving.")
        directory = Path(directory).resolve()
        source = Path(self.model_name).resolve()
        if directory == source or directory in source.parents:
            raise ValueError("Save to a separate classifier directory, outside the source model.")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Save destination must be empty; use a new experiment/artifact directory.")
        directory.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".bert-export-", dir=directory.parent))
        try:
            self._backend.export(stage)
            self.tokenizer.save_pretrained(str(stage))
            settings = {"format_version": FORMAT_VERSION, "backend": self.backend_, "source": self.model_name,
                        "source_fingerprint": self.source_fingerprint_, "classes": self.classes_.tolist(),
                        "training_text_counts": self.training_text_counts, "history": self.history_,
                        "best_epoch": self.best_epoch_, "best_validation_macro_f1": self.best_validation_macro_f1_,
                        "stopped_early": self.stopped_early_, **self._settings()}
            (stage / "classifier_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
            if directory.exists():
                directory.rmdir()
            stage.replace(directory)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    @classmethod
    def load(cls, directory, threads=4):
        directory = Path(directory).resolve()
        settings = json.loads((directory / "classifier_settings.json").read_text(encoding="utf-8"))
        result = cls(str(directory), threads=threads)
        for key in result._settings():
            if key in settings and key != "threads":
                setattr(result, key, settings[key])
        result._validate_settings()
        result.load_options = {"local_files_only": True}
        result.backend_ = settings.get("backend")
        if result.backend_ != "tensorflow":
            raise ValueError("Expected a TensorFlow BERT classifier artifact.")
        result._backend = _TensorFlowBackend(result, training=False)
        result.model, result.device = result._backend.model, result._backend.device
        result.classes_ = np.asarray(settings["classes"], dtype=str)
        result.training_text_counts = settings.get("training_text_counts", {})
        result.source_fingerprint_ = settings.get("source_fingerprint")
        result.history_, result.best_epoch_ = settings.get("history", []), settings.get("best_epoch")
        result.best_validation_macro_f1_ = settings.get("best_validation_macro_f1")
        result.stopped_early_ = settings.get("stopped_early", False)
        result.fitted_ = True
        return result

    def close(self):
        """Release this classifier; safe before fit and safe to call repeatedly."""
        import gc
        backend = self._backend
        self._backend, self.model, self.tokenizer, self.selector = None, None, None, None
        self.fitted_ = False
        if backend is not None and isinstance(backend, _TensorFlowBackend):
            # Break the compiled gradient closure's reference cycle.
            backend.gradients = None
        del backend
        gc.collect()


# ----------------------- Standalone training / MLflow -----------------------
import argparse
import importlib.metadata
import sys
from datetime import datetime, timezone

import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from packaging.version import Version
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

REGISTERED_MODEL_NAME = "EWOC_TicketType_BERT"
LOCAL_BERT_DIRECTORY = "models/bert_tf"
EXCLUDED_TYPE_IDS = {32, 33, 34, 35}
TEXT_COLUMN = "DESCRIPTION"
LABEL_COLUMN = "WorkOrder_Info"
GROUP_COLUMN = "NORMALIZED_DESCRIPTION"


def bert_text(value):
    """Identical, light preprocessing for training and serving; retain negation."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


class TicketTypeBertModel(mlflow.pyfunc.PythonModel):
    """No live TensorFlow model is pickled: load weights from bundled artifacts."""

    def load_context(self, context):
        self.classifier = BertClassifier.load(context.artifacts["classifier"])
        self.type_name_to_id = json.loads(
            Path(context.artifacts["type_name_to_id"]).read_text(encoding="utf-8"))
        missing = set(self.classifier.classes_) - set(self.type_name_to_id)
        if missing:
            raise ValueError(f"Model artifact has unmapped labels: {sorted(missing)}")

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or TEXT_COLUMN not in model_input:
            raise ValueError("Input must be a DataFrame with a DESCRIPTION column")
        texts = model_input[TEXT_COLUMN].map(bert_text).tolist()
        probabilities = self.classifier.predict_proba(texts)
        if not len(probabilities):
            return []
        positions = probabilities.argmax(axis=1)
        names = self.classifier.classes_[positions]
        return [{"workOrderTypeId": int(self.type_name_to_id[name]),
                 "workOrderType": str(name),
                 "confidence": float(probabilities[i, positions[i]])}
                for i, name in enumerate(names)]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def validate_type_mapping(frame):
    """Reject ambiguous lookup labels instead of returning the wrong Oracle ID."""
    pairs = frame[[LABEL_COLUMN, "TYPE_ID"]].drop_duplicates()
    name_counts = pairs.groupby(LABEL_COLUMN)["TYPE_ID"].nunique()
    id_counts = pairs.groupby("TYPE_ID")[LABEL_COLUMN].nunique()
    ambiguous_names = name_counts[name_counts > 1].index
    ambiguous_ids = id_counts[id_counts > 1].index
    if len(ambiguous_names) or len(ambiguous_ids):
        counts = (frame.groupby([LABEL_COLUMN, "TYPE_ID"])
                  .size().reset_index(name="training_rows"))
        conflicts = counts.loc[
            counts[LABEL_COLUMN].isin(ambiguous_names)
            | counts["TYPE_ID"].isin(ambiguous_ids)
        ].sort_values([LABEL_COLUMN, "TYPE_ID"])
        raise ValueError(
            "Ambiguous work-order type mapping. This trainer predicts a type name, "
            "then maps it to one TYPE_ID. Conflicting entries in the cleaned training data:\n"
            + conflicts.to_string(index=False)
            + "\nTraining has not started. Confirm the intended IDs against LU_EWOC_TYPE. "
            "If the IDs are distinct valid classes sharing a display name, use TYPE_ID "
            "as the training target and an ID-to-name mapping throughout training and serving. "
            "Do not resolve this by choosing the first or most frequent ID."
        )
    return {str(name): int(type_id) for name, type_id in pairs.itertuples(index=False, name=None)}


def clean_training_frame(frame, data_loader, min_class_count):
    required = {TEXT_COLUMN, LABEL_COLUMN, "TYPE_ID"}
    if missing := required - set(frame):
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    cleaned = frame.dropna(subset=sorted(required)).copy()
    ids = pd.to_numeric(cleaned["TYPE_ID"], errors="coerce")
    cleaned = cleaned.loc[ids.notna() & np.isfinite(ids) & ids.mod(1).eq(0)].copy()
    cleaned["TYPE_ID"] = ids.loc[cleaned.index].astype("int64")
    cleaned = cleaned.loc[~cleaned["TYPE_ID"].isin(EXCLUDED_TYPE_IDS)].copy()
    if "CREATED_DATE" in cleaned:
        cleaned["CREATED_DATE"] = pd.to_datetime(cleaned["CREATED_DATE"], errors="coerce", utc=True)
        cleaned = cleaned.dropna(subset=["CREATED_DATE"])
    cleaned[TEXT_COLUMN] = cleaned[TEXT_COLUMN].astype(str)
    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype(str)
    usable = cleaned[TEXT_COLUMN].map(bert_text).ne("") & cleaned[LABEL_COLUMN].str.strip().ne("")
    insufficient = cleaned[TEXT_COLUMN].map(data_loader.insufficient_information).astype(bool)
    cleaned = cleaned.loc[usable & ~insufficient].copy()
    cleaned[GROUP_COLUMN] = cleaned[TEXT_COLUMN].map(data_loader.normalize_description)
    if cleaned[GROUP_COLUMN].isna().any():
        raise ValueError("normalize_description returned missing grouping keys.")
    cleaned[GROUP_COLUMN] = cleaned[GROUP_COLUMN].astype(str)
    counts = cleaned[LABEL_COLUMN].value_counts()
    cleaned = cleaned.loc[cleaned[LABEL_COLUMN].isin(counts[counts >= min_class_count].index)]
    cleaned = cleaned.reset_index(drop=True)
    if cleaned[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Fewer than two classes remain after cleaning and minimum-count filtering.")
    # Audit ambiguity; do not silently change labels or remove difficult test examples.
    conflicts = cleaned.groupby(GROUP_COLUMN)[LABEL_COLUMN].nunique().gt(1)
    if conflicts.any():
        logger.warning("%d normalized descriptions have conflicting labels; kept in one partition.", conflicts.sum())
    validate_type_mapping(cleaned)
    logger.info("Cleaning: %d input rows -> %d rows / %d classes", len(frame), len(cleaned),
                cleaned[LABEL_COLUMN].nunique())
    return cleaned


def split_training_frame(frame, test_size, validation_size, seed):
    """Match the RF outer split; choose validation only from outer training rows."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pool_idx, test_idx = next(outer.split(frame, groups=frame[GROUP_COLUMN]))
    pool, test = frame.iloc[train_pool_idx].copy(), frame.iloc[test_idx].copy()
    classes = set(frame[LABEL_COLUMN])
    if missing := classes - set(pool[LABEL_COLUMN]):
        raise ValueError(f"Outer training partition lacks classes {sorted(missing)}. "
                         "Need more independent descriptions for those classes; do not train on test rows.")
    if missing := classes - set(test[LABEL_COLUMN]):
        logger.warning("Test split has no examples for %s; their recall cannot be measured.", sorted(missing))
    splitter = GroupShuffleSplit(n_splits=50, test_size=validation_size, random_state=seed + 1)
    selected = None
    best_coverage = -1
    for train_idx, val_idx in splitter.split(pool, groups=pool[GROUP_COLUMN]):
        train, validation = pool.iloc[train_idx].copy(), pool.iloc[val_idx].copy()
        if set(train[LABEL_COLUMN]) != classes:
            continue
        coverage = validation[LABEL_COLUMN].nunique()
        if coverage > best_coverage:
            selected, best_coverage = (train, validation), coverage
        if coverage == len(classes):
            break
    if selected is None:
        raise ValueError("Cannot form a grouped validation split with every class in training. "
                         "Collect more distinct descriptions or review the validation fraction.")
    train, validation = selected
    partitions = (train, validation, test)
    for i, left in enumerate(partitions):
        for right in partitions[i + 1:]:
            if set(left[GROUP_COLUMN]) & set(right[GROUP_COLUMN]):
                raise RuntimeError("Normalized descriptions overlap across partitions.")
            # Additional check against any mismatch in the external normalization helper.
            left_text = set(left[TEXT_COLUMN].map(lambda x: bert_text(x).lower()))
            right_text = set(right[TEXT_COLUMN].map(lambda x: bert_text(x).lower()))
            if left_text & right_text:
                raise ValueError("Identical BERT input text crosses partitions; review normalize_description.")
    return partitions


def evaluate_classifier(classifier, test_frame):
    truth = test_frame[LABEL_COLUMN].to_numpy(dtype=str)
    probabilities = classifier.predict_proba(test_frame[TEXT_COLUMN].map(bert_text).tolist())
    prediction = classifier.classes_[probabilities.argmax(axis=1)]
    labels = classifier.classes_.tolist()
    metrics = {"accuracy": float(accuracy_score(truth, prediction)),
               "test_log_loss": float(log_loss(truth, probabilities, labels=labels)),
               "evaluated_samples": len(truth), "num_classes": len(labels)}
    for average in ("macro", "weighted"):
        for name, fn in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            metrics[f"{average}_{name}"] = float(fn(truth, prediction, labels=labels,
                                                       average=average, zero_division=0))
    metrics.update(precision=metrics["weighted_precision"], recall=metrics["weighted_recall"],
                   f1=metrics["weighted_f1"])
    confidence = probabilities.max(axis=1)
    sorted_scores = np.sort(probabilities, axis=1)
    metrics.update(prediction_confidence_mean=float(confidence.mean()),
                   low_confidence_rate_lt_0_6=float((confidence < .6).mean()),
                   prediction_margin_mean=float((sorted_scores[:, -1] - sorted_scores[:, -2]).mean()),
                   prediction_entropy_mean=float(-(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1).mean()))
    for p in (10, 50, 90):
        metrics[f"prediction_confidence_p{p}"] = float(np.percentile(confidence, p))
    report = classification_report(truth, prediction, labels=labels, zero_division=0)
    by_class = classification_report(truth, prediction, labels=labels, zero_division=0, output_dict=True)
    confusion = pd.DataFrame(confusion_matrix(truth, prediction, labels=labels), index=labels, columns=labels)
    return metrics, report, by_class, confusion


def runtime_requirements():
    import tensorflow as tf
    requirements = [f"tensorflow=={tf.__version__}"]
    for package in ("keras", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn", "joblib", "cloudpickle", "mlflow"):
        requirements.append(f"{package}=={importlib.metadata.version(package)}")
    return requirements


def package_model(classifier, mapping, output_dir):
    """Save a portable MLflow package, then verify its exact serving contract."""
    export = output_dir / "classifier"
    classifier.save(export)
    mapping_path = output_dir / "type_name_to_id.json"
    write_json(mapping_path, mapping)
    probe = pd.DataFrame({TEXT_COLUMN: ["Please investigate this work order.", "Scheduled network maintenance."]})
    p = classifier.predict_proba(probe[TEXT_COLUMN].tolist())
    expected_names = classifier.classes_[p.argmax(axis=1)].tolist()
    expected_scores = p.max(axis=1)
    expected_ids = [mapping[name] for name in expected_names]
    expected_output = pd.DataFrame({"workOrderTypeId": expected_ids,
                                    "workOrderType": expected_names, "confidence": expected_scores})
    classifier.close()  # Release optimizer state before MLflow restores the inference model.
    requirements = runtime_requirements()
    (output_dir / "requirements-serving.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    package = output_dir / "mlflow_model"
    # Source-based packaging avoids cloudpickling TensorFlow function graphs and
    # isolates the inference code for each registered model version.
    source_text = Path(__file__).read_text(encoding="utf-8")
    source_text = source_text.rsplit('\nif __name__ == "__main__":', 1)[0]
    model_code = output_dir / "bert_inference.py"
    model_code.write_text(source_text + "\nmlflow.models.set_model(TicketTypeBertModel())\n", encoding="utf-8")
    mlflow.pyfunc.save_model(
        path=str(package), python_model=str(model_code),
        artifacts={"classifier": str(export), "type_name_to_id": str(mapping_path)},
        signature=infer_signature(probe, expected_output), input_example=probe,
        pip_requirements=requirements,
    )
    restored = mlflow.pyfunc.load_model(str(package))
    observed = restored.predict(probe)
    if not isinstance(observed, list) or len(observed) != len(probe):
        raise RuntimeError("MLflow reload changed the API return type or row count.")
    if any(set(row) != {"workOrderTypeId", "workOrderType", "confidence"} for row in observed):
        raise RuntimeError("MLflow prediction fields do not match the current API contract.")
    if [row["workOrderType"] for row in observed] != expected_names or [row["workOrderTypeId"] for row in observed] != expected_ids:
        raise RuntimeError("MLflow reload changed label/TYPE_ID mapping.")
    np.testing.assert_allclose([row["confidence"] for row in observed], expected_scores, atol=1e-6, rtol=1e-6)
    if restored.predict(pd.DataFrame({TEXT_COLUMN: pd.Series(dtype=str)})) != []:
        raise RuntimeError("Empty input must return an empty prediction list.")
    restored.unwrap_python_model().classifier.close()
    del restored
    logger.info("MLflow package reload passed: IDs, labels, confidence and empty-batch output.")
    return package


def register_package(package, model_name, target_stage, registry_alias=None):
    mlflow.log_artifacts(str(package), artifact_path="model")
    uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    # Use the returned version; never select the global latest version in a race.
    version = mlflow.register_model(uri, model_name, await_registration_for=300)
    client = MlflowClient()
    client.set_model_version_tag(model_name, version.version, "model_family", "tensorflow_bert_finetuned")
    if target_stage != "None":
        client.transition_model_version_stage(
            name=model_name, version=version.version, stage=target_stage,
            archive_existing_versions=(target_stage == "Production"))
    if registry_alias:
        client.set_registered_model_alias(model_name, registry_alias, version.version)
    return str(version.version), uri


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune one local TensorFlow BERT and register it in MLflow.")
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--bert-dir", default=LOCAL_BERT_DIRECTORY)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8, help="Effective batch = batch-size x accumulation")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-ratio", type=float, default=.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validation-size", type=float, default=.15, help="Fraction of outer training groups")
    parser.add_argument("--test-size", type=float, help="Defaults to config.TEST_SIZE")
    parser.add_argument("--seed", type=int, help="Defaults to config.RANDOM_STATE")
    parser.add_argument("--min-class-count", type=int, default=75)
    parser.add_argument("--min-accuracy", type=float, default=.65, help="Existing registration gate")
    parser.add_argument("--min-macro-f1", type=float, default=0., help="Optional additional registration gate")
    parser.add_argument("--target-stage", choices=["Staging", "Production", "None"], default="Staging")
    parser.add_argument("--registry-alias", help="Only needed if your API loads a registry alias")
    parser.add_argument("--no-register", action="store_true", help="Train/evaluate/package/log without a registry version")
    parser.add_argument("--artifact-dir", default="EWOCBertArtifacts")
    parser.add_argument("--checkpoint-dir", help="Reuse the same directory and settings to resume at an epoch boundary")
    parser.add_argument("--data-csv", help="Optional enriched CSV with DESCRIPTION, WorkOrder_Info and TYPE_ID")
    parser.add_argument("--tracking-uri", help="Override the project's MLflow tracking URI")
    parser.add_argument("--run-name")
    parser.add_argument("--work-orders-table", default="e911.ENMT_E911_WORK_ORDERS")
    parser.add_argument("--type-table", default="e911.LU_EWOC_TYPE")
    parser.add_argument("--rca-table", default="e911.LU_EWOC_RCA")
    parser.add_argument("--market-table", default="e911.LU_EWOC_MARKET")
    parser.add_argument("--status-table", default="e911.LU_EWOC_STATUS")
    parser.add_argument("--where")
    parser.add_argument("--refresh", action="store_true")
    return parser


def project_modules():
    # Support both `python src/ewoc_ttype/bert_train.py` and `python -m ...`.
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from ewoc_ttype.ewoc_utils import config, data_loader, mlflow_helpers
    return config, data_loader, mlflow_helpers


def load_training_frame(args, data_loader):
    if args.data_csv:
        return pd.read_csv(args.data_csv)
    fetch_kwargs = {"force_refresh": args.refresh}
    if args.where:
        if "where" not in inspect.signature(data_loader.fetch_full_table).parameters:
            raise ValueError(
                "--where was requested, but this project's fetch_full_table does not "
                "support a where argument. Use a loader that supports the filter or "
                "provide an already-filtered --data-csv."
            )
        fetch_kwargs["where"] = args.where
    frame = data_loader.fetch_full_table(args.work_orders_table, **fetch_kwargs).copy()
    type_lookup = data_loader.fetch_full_table(args.type_table, force_refresh=args.refresh)
    rca_lookup = data_loader.fetch_full_table(args.rca_table, force_refresh=args.refresh)
    market_lookup = data_loader.fetch_full_table(args.market_table, force_refresh=args.refresh)
    data_loader.fetch_full_table(args.status_table, force_refresh=args.refresh)
    valid = type_lookup.dropna(subset=["TYPE_ID", "TYPE_WO"])
    if valid.groupby("TYPE_ID")["TYPE_WO"].nunique().gt(1).any():
        raise ValueError("Oracle type lookup contains conflicting names for a TYPE_ID.")
    frame[LABEL_COLUMN] = frame["TYPE_ID"].map(valid.set_index("TYPE_ID")["TYPE_WO"].to_dict())
    frame["RCA_Info"] = frame["RCA_ID"].map(rca_lookup.set_index("RCA_ID")["RCA"].to_dict())
    frame["Market_Info"] = frame["MARKET_ID"].map(market_lookup.set_index("ID")["MARKET_CLUSTER"].to_dict())
    return frame


def run_training(args, config, data_loader, mlflow_helpers):
    """Injectable project adapters also permit isolated integration testing."""
    if Version(mlflow.__version__) < Version("2.12.2"):
        raise RuntimeError("Source-based BERT packaging requires mlflow>=2.12.2 in training and serving.")
    test_size = config.TEST_SIZE if args.test_size is None else args.test_size
    seed = config.RANDOM_STATE if args.seed is None else args.seed
    if not 0 < test_size < 1 or not 0 < args.validation_size < 1:
        raise ValueError("test-size and validation-size must be between 0 and 1.")
    if not args.model_name.strip() or args.min_class_count < 1:
        raise ValueError("A model name and positive min-class-count are required.")
    for gate in (args.min_accuracy, args.min_macro_f1):
        if not 0 <= gate <= 1:
            raise ValueError("Registration metric gates must lie between 0 and 1.")
    source = Path(args.bert_dir).expanduser().resolve()
    for relative in ("saved_model.pb", "assets/vocab.txt", "variables/variables.index"):
        if not (source / relative).is_file():
            raise FileNotFoundError(f"Local BERT model is missing {source / relative}")
    if not list((source / "variables").glob("variables.data-*")):
        raise FileNotFoundError("Local BERT variable data shards are missing.")
    classifier = BertClassifier(
        model_name=str(source), epochs=args.epochs, learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate, seed=seed, threads=args.threads,
        max_length=args.max_length, batch_size=args.batch_size, accumulation=args.accumulation,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio, patience=args.patience,
        class_weight=None if args.class_weight == "none" else "balanced")
    classifier._validate_settings()
    if not args.data_csv:
        config.validate_config()
    mlflow_helpers.setup_mlflow()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    if mlflow.active_run() is not None:
        raise RuntimeError("An MLflow run is already active. Launch this standalone trainer in a fresh process.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=args.run_name or f"BERT_TicketType_{stamp}") as run:
        output = Path(args.artifact_dir).resolve() / f"{stamp}_{run.info.run_id[:8]}"
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else output / "checkpoints"
        logger.info("BERT-only run. Artifacts: %s; checkpoints: %s", output, checkpoint)
        mlflow.set_tags({"model_family": "tensorflow_bert_finetuned", "registration_status": "training"})
        mlflow.log_params({**classifier._settings(), "model_name": args.model_name,
                           "bert_source": str(source), "test_size": test_size,
                           "validation_size_within_train": args.validation_size,
                           "min_class_count": args.min_class_count,
                           "excluded_type_ids": "32,33,34,35", "target_stage": args.target_stage,
                           "min_accuracy_gate": args.min_accuracy, "min_macro_f1_gate": args.min_macro_f1,
                           "pipeline": "RAW_TEXT->BERT_TOKENIZER->FULL_ENCODER_FINETUNE->CLASSIFICATION_HEAD",
                           "text_policy": "whitespace_only; uncased_wordpiece; head_tail_truncation"})
        try:
            raw = load_training_frame(args, data_loader)
            frame = clean_training_frame(raw, data_loader, args.min_class_count)
            del raw
            train, validation, test = split_training_frame(frame, test_size, args.validation_size, seed)
            mapping = validate_type_mapping(frame)
            # Preserve row/group fingerprints to identify the exact evaluation split without logging ticket text.
            manifest_rows = []
            for partition_name, partition in (("train", train), ("validation", validation), ("test", test)):
                for idx, row in partition.iterrows():
                    manifest_rows.append({"row": int(idx), "partition": partition_name,
                                          "description_sha256": hashlib.sha256(str(row[TEXT_COLUMN]).encode()).hexdigest(),
                                          "group_sha256": hashlib.sha256(str(row[GROUP_COLUMN]).encode()).hexdigest(),
                                          "type_id": int(row["TYPE_ID"])})
            pd.DataFrame(manifest_rows).to_csv(output / "split_manifest.csv", index=False)
            distribution = pd.DataFrame({name: part[LABEL_COLUMN].value_counts()
                                         for name, part in (("train", train), ("validation", validation), ("test", test))}).fillna(0).astype(int)
            distribution.to_csv(output / "class_distribution.csv", index_label="workOrderType")
            logger.info("Rows: train=%d validation=%d test=%d; classes=%d", len(train), len(validation), len(test), len(mapping))
            classifier.fit(train[TEXT_COLUMN].map(bert_text).tolist(), train[LABEL_COLUMN].tolist(),
                           validation_data=(validation[TEXT_COLUMN].map(bert_text).tolist(), validation[LABEL_COLUMN].tolist()),
                           checkpoint_dir=checkpoint)
            for record in classifier.history_:
                mlflow.log_metrics({k: float(v) for k, v in record.items() if k != "epoch"}, step=record["epoch"])
            metrics, report, by_class, confusion = evaluate_classifier(classifier, test)
            summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "registered_model_name": args.model_name, "run_id": run.info.run_id,
                       "source_fingerprint": classifier.source_fingerprint_,
                       "dataset": {"cleaned_rows": len(frame), "class_count": len(mapping)},
                       "split": {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
                       "best_epoch": classifier.best_epoch_,
                       "best_validation_macro_f1": classifier.best_validation_macro_f1_,
                       "stopped_early": classifier.stopped_early_, "training_text_counts": classifier.training_text_counts,
                       "metrics": metrics, "confidence_calibration": "uncalibrated_softmax",
                       "history": classifier.history_}
            (output / "classification_report.txt").write_text(report, encoding="utf-8")
            write_json(output / "test_metrics.json", metrics)
            write_json(output / "class_metrics.json", by_class)
            write_json(output / "training_summary.json", summary)
            confusion.to_csv(output / "confusion_matrix.csv", index_label="actual")
            mlflow.log_metrics({**metrics, "best_epoch": classifier.best_epoch_,
                                "best_validation_macro_f1": classifier.best_validation_macro_f1_,
                                "split_train_rows": len(train), "split_validation_rows": len(validation),
                                "split_test_rows": len(test), "dataset_cleaned_rows": len(frame)})
            logger.info("Test: accuracy=%.4f macro-F1=%.4f weighted-F1=%.4f",
                        metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
            package = package_model(classifier, mapping, output)
            # Reports only; model and tokenizer are uploaded once as the MLflow package.
            for artifact in sorted(output.iterdir()):
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact), artifact_path="training")
            passed = metrics["accuracy"] >= args.min_accuracy and metrics["macro_f1"] >= args.min_macro_f1
            if passed and not args.no_register:
                version, uri = register_package(package, args.model_name, args.target_stage, args.registry_alias)
                result = {"status": "registered", "model_name": args.model_name, "version": version,
                          "stage": args.target_stage, "model_uri": f"models:/{args.model_name}/{version}",
                          "run_model_uri": uri, "run_id": run.info.run_id, "artifact_dir": str(output)}
                logger.info("Registered %s version %s, stage %s", args.model_name, version, args.target_stage)
            else:
                mlflow.log_artifacts(str(package), artifact_path="model")
                result = {"status": "not_registered_by_request" if args.no_register else "skipped_below_metric_gate",
                          "model_name": args.model_name, "run_id": run.info.run_id, "artifact_dir": str(output)}
                logger.warning("%s. Trained model and diagnostics remain available.", result["status"])
            write_json(output / "registration.json", result)
            mlflow.log_artifact(str(output / "registration.json"), artifact_path="training")
            mlflow.set_tag("registration_status", result["status"])
            print(json.dumps(result, indent=2))
            return result
        except BaseException:
            mlflow.set_tag("registration_status", "failed")
            raise
        finally:
            classifier.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config, data_loader, mlflow_helpers = project_modules()
        result = run_training(args, config, data_loader, mlflow_helpers)
        return 2 if result["status"] == "skipped_below_metric_gate" else 0
    except Exception:
        logger.exception("BERT training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
