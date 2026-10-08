"""Supervised BERT/MiniLM: jointly train encoder and classification head.

Transformers downloads pretrained weights on first use and reuses its cache.
MiniLM uses the local checkpoint and a training-only TF-IDF clause selector.
Epoch checkpoints resume optimizer, scheduler and model; completed fits discard them.
"""

import json
import logging
import math
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("ewoc.benchmark")


class ClauseSelector:
    """Rank intact clauses with training TF-IDF; keep original order and short inputs."""
    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=30000,
            sublinear_tf=True, norm=None,
            token_pattern=r"(?u)\b(?=[a-z0-9_-]*[a-z])[a-z0-9_-]{2,}\b", dtype=np.float32)
        self.vectorizer.fit([self.scoring_text(text) for text in texts])
        return self

    @staticmethod
    def scoring_text(text):
        # Mask contacts, long numbers and TT references for ranking; keep terms such as E911/5G.
        return re.sub(r"https?://\S+|[\w.%+-]+@[\w.-]+\.[a-z]{2,}|\b(?:TT\d+|\d{5,})\b",
                      " ", text, flags=re.IGNORECASE)

    def select(self, text, token_ids, tokenizer, budget):
        if len(token_ids) <= budget:
            return token_ids, "full"
        clauses = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|[\r\n]+", text) if s.strip()]
        ids = tokenizer(clauses, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        vectors = self.vectorizer.transform([self.scoring_text(s) for s in clauses])
        # Mean TF-IDF over observed n-grams avoids preferring a clause just because it is long.
        scores = np.asarray(vectors.sum(axis=1)).ravel() / np.maximum(vectors.getnnz(axis=1), 1)
        if len(clauses) < 2 or not np.any(scores):
            return self.head_tail(token_ids, budget), "head_tail"
        # Preserve the opening context when it leaves room for informative clauses.
        selected = [0] if 0 < len(ids[0]) <= budget // 2 else []
        remaining = budget - sum(len(ids[i]) for i in selected)
        for index in np.argsort(-scores, kind="stable"):
            cost = len(ids[index]) + bool(selected)
            if index not in selected and 0 < len(ids[index]) and cost <= remaining:
                selected.append(int(index))
                remaining -= cost
        if not selected:
            return self.head_tail(token_ids, budget), "head_tail"
        # Retain boundary markers so separated clauses are not presented as adjacent words.
        chosen = sorted(selected)
        output = []
        for index in chosen:
            if output:
                output.append(tokenizer.sep_token_id)
            output.extend(ids[index])
        return output, "clauses"

    @staticmethod
    def head_tail(token_ids, budget):
        head = (budget + 1) // 2
        return token_ids[:head] + token_ids[-(budget - head):]


def prepare_bert(model_id, cache_dir):
    """Fetch/check weights before the long benchmark, then pin the resolved revision."""
    import gc
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    options = {"cache_dir": str(cache_dir)}
    config = AutoConfig.from_pretrained(model_id, **options)
    revision = getattr(config, "_commit_hash", None) or "main"
    options["revision"] = revision
    AutoTokenizer.from_pretrained(model_id, **options)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, **options)
    del model
    gc.collect()
    return revision


class BertClassifier:
    def __init__(self, model_name="google-bert/bert-base-uncased", epochs=3, learning_rate=2e-5,
                 seed=42, threads=4, cache_dir=None, revision="main", clause_tfidf=False,
                 head_learning_rate=None):
        self.model_name, self.epochs, self.learning_rate = str(model_name), epochs, learning_rate
        self.load_options = {"revision": revision}
        if cache_dir is not None:
            self.load_options["cache_dir"] = str(cache_dir)
        self.seed, self.threads = seed, threads
        self.max_length, self.batch_size, self.accumulation = 256, 8, 4
        self.clause_tfidf, self.head_learning_rate = clause_tfidf, head_learning_rate or learning_rate
        self.name = "TF-IDF + MiniLM" if clause_tfidf else "BERT"
        self.selector = None
        if Path(self.model_name).is_dir():
            self.load_options["local_files_only"] = True

    def _tokenize(self, texts):
        # Keep both ends of long tickets; prelaunch/context may appear after a long site list.
        texts = list(texts)
        result, counts = [], {"full": 0, "clauses": 0, "head_tail": 0}
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        for offset in range(0, len(texts), 256):
            raw = self.tokenizer(texts[offset:offset + 256], add_special_tokens=False,
                                 truncation=False, verbose=False)["input_ids"]
            for text, ids in zip(texts[offset:offset + 256], raw):
                if self.selector is not None:
                    ids, mode = self.selector.select(text, ids, self.tokenizer, budget)
                elif len(ids) > budget:
                    ids, mode = ClauseSelector.head_tail(ids, budget), "head_tail"
                else:
                    mode = "full"
                counts[mode] += 1
                result.append(self.tokenizer.prepare_for_model(ids, return_attention_mask=True,
                                                               return_token_type_ids=True))
        return result, counts

    def fit(self, texts, labels, sample_weight=None, checkpoint_dir=None):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
        from transformers.utils import logging as hf_logging

        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        texts = list(texts)
        self.classes_, targets = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Sequence classification needs at least two training classes.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.load_options)
        config = AutoConfig.from_pretrained(self.model_name, **self.load_options)
        if config.model_type != "bert":
            raise ValueError("The BERT/MiniLM checkpoint must have model_type='bert'.")
        config.num_labels = len(self.classes_)
        config.id2label = dict(enumerate(self.classes_.tolist()))
        config.label2id = {label: index for index, label in config.id2label.items()}
        config.problem_type = "single_label_classification"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config, **self.load_options).to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.selector = ClauseSelector().fit(texts) if self.clause_tfidf else None
        encoded, self.training_text_counts = self._tokenize(texts)
        weights = np.ones(len(targets)) if sample_weight is None else np.asarray(sample_weight)
        if len(weights) != len(targets) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Training weights must be finite, positive and match the real training rows.")
        weights = weights / weights.mean()
        labels_tensor = torch.tensor(targets, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        def collate(indices):
            batch = self.tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors="pt")
            return batch, labels_tensor[indices], weights_tensor[indices]

        groups = []
        for is_head in (False, True):
            for no_decay in (False, True):
                parameters = [p for name, p in self.model.named_parameters()
                    if (name.startswith("classifier.") or ".pooler." in name) == is_head
                    and (name.endswith("bias") or "LayerNorm.weight" in name) == no_decay]
                if parameters:
                    groups.append({"params": parameters, "weight_decay": 0.0 if no_decay else 0.01,
                                   "lr": self.head_learning_rate if is_head else self.learning_rate})
        optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
        batches = math.ceil(len(targets) / self.batch_size)
        total_steps = self.epochs * math.ceil(batches / self.accumulation)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
        checkpoint = Path(checkpoint_dir) / "epoch.pt" if checkpoint_dir is not None else None
        signature = {"classes": self.classes_.tolist(), "rows": len(targets), "epochs": self.epochs,
                     "learning_rate": self.learning_rate, "max_length": self.max_length,
                     "model": self.model_name, "revision": self.load_options["revision"],
                     "clause_tfidf": self.clause_tfidf, "head_learning_rate": self.head_learning_rate}
        first_epoch = 0
        if checkpoint is not None and checkpoint.exists():
            saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if saved["signature"] != signature:
                raise ValueError("Checkpoint does not match this training configuration.")
            self.model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            first_epoch = saved["epoch"]
            del saved
        logger.info("%s: %s | %d tickets | %d classes | %d/%d epochs complete | text selection %s",
                    self.name, self.device, len(targets), len(self.classes_), first_epoch, self.epochs,
                    self.training_text_counts)
        for epoch in range(first_epoch, self.epochs):
            torch.manual_seed(self.seed + epoch)
            loader = torch.utils.data.DataLoader(list(range(len(targets))), batch_size=self.batch_size,
                shuffle=True, generator=torch.Generator().manual_seed(self.seed + epoch),
                collate_fn=collate, num_workers=0)
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            started, last_log, total_loss, seen = perf_counter(), perf_counter(), 0.0, 0
            for step, (inputs, y, row_weights) in enumerate(loader):
                inputs, y, row_weights = inputs.to(self.device), y.to(self.device), row_weights.to(self.device)
                logits = self.model(**inputs).logits
                losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
                weighted = losses * row_weights
                # Exact accumulation even for the final, short batch/group.
                group_start = (step // self.accumulation) * self.accumulation * self.batch_size
                group_size = min(self.accumulation * self.batch_size, len(targets) - group_start)
                loss = weighted.sum() / group_size
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became nonfinite; this trial cannot pass.")
                loss.backward()
                total_loss += float(weighted.detach().sum())
                seen += len(y)
                if (step + 1) % self.accumulation == 0 or step + 1 == batches:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if perf_counter() - last_log >= 30 or step + 1 == batches:
                    elapsed = perf_counter() - started
                    logger.info("%s epoch %d/%d | batch %d/%d | loss %.4f | epoch ETA %.1f min",
                        self.name, epoch + 1, self.epochs, step + 1, batches, total_loss / seen,
                        elapsed / (step + 1) * (batches - step - 1) / 60)
                    last_log = perf_counter()
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"signature": signature, "epoch": epoch + 1, "model": self.model.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, temporary)
                temporary.replace(checkpoint)
        self.model.eval()
        return self

    def predict(self, texts):
        import torch
        encoded, _ = self._tokenize(texts)
        predicted = []
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(encoded), self.batch_size):
                batch = self.tokenizer.pad(encoded[offset:offset + self.batch_size], padding=True,
                                           return_tensors="pt").to(self.device)
                predicted.extend(self.model(**batch).logits.argmax(-1).cpu().tolist())
        return self.classes_[np.asarray(predicted, dtype=int)]

    def save(self, directory):
        directory = Path(directory)
        self.model.save_pretrained(str(directory), safe_serialization=True)
        self.tokenizer.save_pretrained(str(directory))
        if self.selector is not None:
            joblib.dump(self.selector.vectorizer, directory / "clause_tfidf.pkl")
        settings = {"clause_tfidf": self.clause_tfidf, "max_length": self.max_length,
                    "source": self.model_name, "training_text_counts": self.training_text_counts}
        (directory / "classifier_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory, threads=4):
        """Reload trained weights plus the same clause selector for prediction."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        directory = Path(directory)
        settings = json.loads((directory / "classifier_settings.json").read_text(encoding="utf-8"))
        result = cls(str(directory), clause_tfidf=settings["clause_tfidf"], threads=threads)
        torch.set_num_threads(threads)
        result.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        result.model = AutoModelForSequenceClassification.from_pretrained(str(directory), local_files_only=True).to(result.device).eval()
        result.max_length = settings["max_length"]
        result.training_text_counts = settings["training_text_counts"]
        result.classes_ = np.asarray([result.model.config.id2label[i] for i in range(result.model.config.num_labels)])
        if result.clause_tfidf:
            result.selector = ClauseSelector()
            result.selector.vectorizer = joblib.load(directory / "clause_tfidf.pkl")
        return result

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
