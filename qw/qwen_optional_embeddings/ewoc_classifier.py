"""Local Qwen3-8B inference, optional LoRA adapter and retrieval evidence.

Generation is constrained to allowed integer IDs; names are mapped in Python.
This is a replacement module, not an API-compatible patch of the photographed code.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

import json
from collections import Counter
from pathlib import Path

from ewoc_common import (DEFAULT_DEFINITIONS, DEFAULT_LABELS, digest, fit_prompt,
                         label_signature, load_definitions, load_labels, normalize)
from ewoc_retrieval import local_model, model_stamp


def add_inference_args(parser):
    parser.add_argument("--model", default="models/Qwen3-8B", help="Local base model directory")
    parser.add_argument("--adapter", default=None, help="Saved adapter/ directory from finetune_qwen.py")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--definitions", default=str(DEFAULT_DEFINITIONS))
    parser.add_argument("--index", default="indexes/lexical")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--reranker", default=None, help="Optional local BAAI/bge-reranker-v2-m3 directory")
    parser.add_argument("--retrieval-device", default="cpu", help="CPU avoids contending with vLLM memory")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--max-per-type", type=int, default=4)
    parser.add_argument("--pool", type=int, default=80)
    parser.add_argument("--rerank-top", type=int, default=48)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)


def make_retriever(args, labels):
    if args.no_rag:
        return None
    from ewoc_retrieval import HybridRetriever
    retriever = HybridRetriever(args.index, device=args.retrieval_device,
                                 reranker_path=args.reranker, rerank_device=args.retrieval_device)
    if retriever.labels != labels:
        raise ValueError("Index and inference label maps differ. Rebuild index with the same labels.json.")
    return retriever


def retrieve_many(retriever, rows, args):
    if retriever is None:
        return [[] for _ in rows]
    embeddings = retriever.encode_queries([r["description"] for r in rows])
    evidence = [retriever.retrieve(r["description"], args.examples, args.max_per_type,
                                   args.pool, args.rerank_top, query_embedding=q,
                                   exclude_row_key=r.get("row_key"), exclude_group=r.get("group_id"))
                for r,q in zip(rows,embeddings)]
    # In evaluation/one-shot inference all retrieval occurs before vLLM allocates VRAM.
    retriever.release_neural_models()
    return evidence


def assert_adapter_holdout(metadata, rows):
    if not metadata:
        return
    checks = [("training_row_keys", lambda r:r["row_key"]),
              ("training_group_ids", lambda r:r.get("group_id", "")),
              ("training_ticket_hashes", lambda r:digest(r["ewoc_id"]) if r.get("ewoc_id") else ""),
              ("training_text_hashes", lambda r:digest(normalize(r["description"])))]
    for key,getter in checks:
        values = {getter(r) for r in rows} - {""}
        overlap = values & set(metadata.get(key,[]))
        if overlap:
            raise ValueError(f"Fine-tuning leakage: evaluation overlaps {key} ({len(overlap)} values).")


class QwenClassifier:
    def __init__(self, args, labels=None, definitions=None):
        self.args = args
        self.labels = labels or load_labels(args.labels)
        self.definitions = definitions if definitions is not None else load_definitions(args.definitions,self.labels)
        self.adapter_metadata = None
        self.lora_request = None
        base = local_model(args.model)
        extra = {}
        metadata_path = Path(base)/"ewoc_training_metadata.json"
        if args.adapter:
            adapter = local_model(args.adapter)
            metadata_path = Path(adapter)/"ewoc_training_metadata.json"
            if not metadata_path.exists():
                raise ValueError("Adapter lacks EWOC metadata. Use adapter/ exported by finetune_qwen.py.")
        if metadata_path.exists():
            self.adapter_metadata = json.loads(metadata_path.read_text())
            if self.adapter_metadata["label_signature"] != label_signature(self.labels,self.definitions):
                raise ValueError("Adapter taxonomy/definitions differ from inference. Use its original configuration.")
        if args.adapter:
            if model_stamp(base) != self.adapter_metadata["base_model_stamp"]:
                raise ValueError("Base model files changed since training. Use the original model directory.")
            extra = {"enable_lora":True, "max_lora_rank": self.adapter_metadata["lora_rank"]}
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        if args.adapter:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("ewoc", 1, adapter)
        self.llm = LLM(model=base, tokenizer=base, dtype="bfloat16", trust_remote_code=False,
                       max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                       seed=args.seed, enable_prefix_caching=True, generation_config="vllm", **extra)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0., max_tokens=16, seed=args.seed,
                                      structured_outputs=StructuredOutputsParams(
                                          choice=[str(k) for k in self.labels]))

    def classify_batch(self, descriptions, examples_by_prompt):
        if len(descriptions) != len(examples_by_prompt):
            raise ValueError("Descriptions/evidence length mismatch")
        if not descriptions:
            return []
        prompts, stats = [], []
        for text,examples in zip(descriptions,examples_by_prompt):
            prompt, ids, info = fit_prompt(self.tokenizer,text,examples,self.labels,
                                           self.definitions,self.args.max_model_len-32)
            # Token IDs avoid vLLM injecting a second set of special tokens.
            prompts.append({"prompt_token_ids":ids})
            stats.append(info)
        outputs = self.llm.generate(prompts,self.params,lora_request=self.lora_request,use_tqdm=True)
        results = []
        for out,examples,info in zip(outputs,examples_by_prompt,stats):
            raw = out.outputs[0].text.strip()
            type_id = int(raw) if raw.isdigit() and int(raw) in self.labels else None
            used = examples[:info["examples_used"]]
            counts = Counter(ex["type_id"] for ex in used)
            results.append({"type_id":type_id, "type_name":self.labels.get(type_id,"INVALID"),
                            "raw":raw,"finish_reason":out.outputs[0].finish_reason,
                            "prompt_stats":info,"retrieved_type_counts":dict(counts),
                            "evidence":[{"ewoc_id":ex.get("ewoc_id"),"row_key":ex["row_key"],
                                         "type_id":ex["type_id"],"description":ex["description"],
                                         "rrf_score":ex.get("rrf_score"),
                                         "rerank_score":ex.get("rerank_score")} for ex in used]})
        return results


def load_model(model_path="models/Qwen3-8B", **kwargs):
    """Small programmatic entry point; returns QwenClassifier (new API)."""
    import argparse
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args([])
    args.model = str(model_path)
    for key,value in kwargs.items():
        if not hasattr(args,key):
            raise ValueError(f"Unknown option: {key}")
        setattr(args,key,value)
    return QwenClassifier(args)
