"""BERT sequence classification: jointly train encoder and classification head.

Transformers downloads pretrained weights on first use and reuses its cache.
No Trainer/accelerate dependency or SMOTE on token IDs.
Epoch checkpoints resume optimizer, scheduler and model; completed fits discard them.
"""

import logging
import math
from pathlib import Path
from time import perf_counter

import numpy as np

logger = logging.getLogger("ewoc.benchmark")


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
                 seed=42, threads=4, cache_dir=None, revision="main"):
        self.model_name, self.epochs, self.learning_rate = str(model_name), epochs, learning_rate
        self.load_options = {"revision": revision}
        if cache_dir is not None:
            self.load_options["cache_dir"] = str(cache_dir)
        self.seed, self.threads = seed, threads
        self.max_length, self.batch_size, self.accumulation = 256, 8, 4

    def _tokenize(self, texts):
        # Keep both ends of long tickets; prelaunch/context may appear after a long site list.
        texts = list(texts)
        result, truncated = [], 0
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        for offset in range(0, len(texts), 256):
            raw = self.tokenizer(texts[offset:offset + 256], add_special_tokens=False,
                                 truncation=False, verbose=False)["input_ids"]
            for ids in raw:
                if len(ids) > budget:
                    head = (budget + 1) // 2
                    ids = ids[:head] + ids[-(budget - head):]
                    truncated += 1
                result.append(self.tokenizer.prepare_for_model(ids, return_attention_mask=True,
                                                               return_token_type_ids=True))
        return result, truncated

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
        self.classes_, targets = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("BERT needs at least two training classes.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.load_options)
        config = AutoConfig.from_pretrained(self.model_name, **self.load_options)
        if config.model_type != "bert":
            raise ValueError("bert_ft requires a checkpoint with model_type='bert'.")
        config.num_labels = len(self.classes_)
        config.id2label = dict(enumerate(self.classes_.tolist()))
        config.label2id = {label: index for index, label in config.id2label.items()}
        config.problem_type = "single_label_classification"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config, **self.load_options).to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        encoded, truncated = self._tokenize(texts)
        weights = np.ones(len(targets)) if sample_weight is None else np.asarray(sample_weight)
        if len(weights) != len(targets) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Training weights must be finite, positive and match the real training rows.")
        weights = weights / weights.mean()
        labels_tensor = torch.tensor(targets, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        def collate(indices):
            batch = self.tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors="pt")
            return batch, labels_tensor[indices], weights_tensor[indices]

        groups = [{"params": [p for name, p in self.model.named_parameters()
                               if not (name.endswith("bias") or "LayerNorm.weight" in name)], "weight_decay": 0.01},
                  {"params": [p for name, p in self.model.named_parameters()
                               if name.endswith("bias") or "LayerNorm.weight" in name], "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
        batches = math.ceil(len(targets) / self.batch_size)
        total_steps = self.epochs * math.ceil(batches / self.accumulation)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
        checkpoint = Path(checkpoint_dir) / "epoch.pt" if checkpoint_dir is not None else None
        signature = {"classes": self.classes_.tolist(), "rows": len(targets), "epochs": self.epochs,
                     "learning_rate": self.learning_rate, "max_length": self.max_length,
                     "model": self.model_name, "revision": self.load_options["revision"]}
        first_epoch = 0
        if checkpoint is not None and checkpoint.exists():
            saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if saved["signature"] != signature:
                raise ValueError("BERT checkpoint does not match this training configuration.")
            self.model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            first_epoch = saved["epoch"]
            del saved
        logger.info("BERT: %s | %d tickets | %d classes | %d/%d epochs complete | %d long tickets use head+tail",
                    self.device, len(targets), len(self.classes_), first_epoch, self.epochs, truncated)
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
                    raise ValueError("BERT loss became nonfinite; this trial cannot pass.")
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
                    logger.info("BERT epoch %d/%d | batch %d/%d | loss %.4f | epoch ETA %.1f min",
                        epoch + 1, self.epochs, step + 1, batches, total_loss / seen,
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
        self.model.save_pretrained(str(directory), safe_serialization=True)
        self.tokenizer.save_pretrained(str(directory))

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
