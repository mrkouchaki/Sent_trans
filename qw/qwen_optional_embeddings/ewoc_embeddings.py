"""Optional local embedding adapters. Importing this module never loads torch.

MiniLM/SentenceTransformer pipelines retain their saved pooling and input limit.
Raw BERT-compatible encoders are explicitly wrapped with masked mean pooling.
Qwen embedding checkpoints alone receive a query instruction.
"""
from __future__ import annotations
import json
from pathlib import Path
from ewoc_common import QUERY_INSTRUCTION

KINDS = ("auto", "minilm", "bert", "sentence-transformer", "qwen")


def inspect_embedding_profile(path, kind="auto", max_length=None):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Local embedding model directory missing: {path}")
    if kind not in KINDS:
        raise ValueError(f"Unknown embedding kind: {kind}")
    if max_length is not None and max_length < 32:
        raise ValueError("Embedding max length must be >=32")
    config_path = path/"config.json"
    manifest_path = path/"modules.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_type = str(config.get("model_type", "")).lower()
    architectures = config.get("architectures", [])
    has_pipeline = manifest_path.is_file()
    if has_pipeline:
        modules = json.loads(manifest_path.read_text())
        pooling = [m for m in modules if str(m.get("type", "")).endswith(".Pooling")]
        if not pooling or not all((path/m.get("path", "")/"config.json").is_file() for m in pooling):
            raise ValueError("Incomplete SentenceTransformer pipeline: include its pooling files")
    else:
        if any("ForCausalLM" in str(a) for a in architectures) or model_type.startswith("qwen"):
            raise ValueError("Ordinary generation models cannot be silently used as embedding models. "
                             "Provide a sentence embedding checkpoint or a BERT-compatible encoder; "
                             "omit --embedding-model to disable dense retrieval.")
        if model_type not in {"bert", "distilbert", "roberta", "albert", "electra", "deberta", "deberta-v2"}:
            raise ValueError("Raw embedding encoder is not a supported BERT-family model. "
                             "Use a complete SentenceTransformer checkpoint instead.")
        if kind in {"qwen", "sentence-transformer"}:
            raise ValueError("This embedding kind requires its complete modules.json and pooling configuration")
    if kind == "auto":
        resolved_kind = "qwen" if has_pipeline and model_type.startswith("qwen") else (
            "sentence-transformer" if has_pipeline else "bert")
    else:
        resolved_kind = kind
    if resolved_kind == "qwen" and not model_type.startswith("qwen"):
        raise ValueError("--embedding-kind qwen requires a Qwen embedding checkpoint")
    if model_type.startswith("qwen") and resolved_kind != "qwen":
        raise ValueError("For Qwen embedding checkpoints use --embedding-kind qwen or auto")
    return {"kind": resolved_kind, "pipeline": "saved" if has_pipeline else "bert-mean-pooling",
            "requested_max_length": max_length,
            "query_instruction": QUERY_INSTRUCTION if resolved_kind == "qwen" else None,
            "format_version": 1}


def load_embedding_encoder(path, profile, device="cpu"):
    from sentence_transformers import SentenceTransformer, models
    if profile["pipeline"] == "saved":
        encoder = SentenceTransformer(str(path), device=device, local_files_only=True,
                                      trust_remote_code=False)
    else:
        transformer = models.Transformer(
            str(path), model_args={"local_files_only": True, "trust_remote_code": False},
            tokenizer_args={"local_files_only": True, "trust_remote_code": False},
            config_args={"local_files_only": True, "trust_remote_code": False})
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        encoder = SentenceTransformer(modules=[transformer, pooling], device=device)
    native = int(encoder.max_seq_length)
    requested = profile.get("effective_max_length") or profile.get("requested_max_length")
    default_cap = 1024 if profile["kind"] == "qwen" else native
    encoder.max_seq_length = min(native, requested or default_cap)
    return encoder


def format_embedding_text(encoder, text, profile, is_query=False):
    """Preserve both ends of long tickets within this encoder's token budget."""
    prefix = ("Instruct: " + profile["query_instruction"] + "\nQuery: "
              if is_query and profile.get("query_instruction") else "")
    tokenizer = encoder.tokenizer
    budget = (encoder.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
              - len(tokenizer.encode(prefix, add_special_tokens=False)) - 8)
    if budget < 16:
        raise ValueError("Embedding input budget too short for its instruction")
    tokens = tokenizer.encode(str(text), add_special_tokens=False)
    if len(tokens) > budget:
        head = max(1, int(budget*.65))
        text = (tokenizer.decode(tokens[:head], skip_special_tokens=True) + "\n...\n" +
                tokenizer.decode(tokens[-(budget-head):], skip_special_tokens=True))
    return prefix + str(text)


def encode_texts(encoder, texts, profile, is_query=False, batch_size=32, progress=False):
    prepared = [format_embedding_text(encoder, t, profile, is_query) for t in texts]
    return encoder.encode(prepared, batch_size=batch_size, prompt="",
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=progress)
