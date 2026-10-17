"""BM25 + character TF-IDF + optional dense embeddings, RRF and reranking.

Indexes are local joblib files: load only artifacts you built/trust.
All neural model paths must already exist locally; no automatic downloads.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ewoc_common import QUERY_INSTRUCTION, digest, head_tail, normalize, write_json

INDEX_VERSION = 1


def local_model(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Local model directory missing: {resolved}. Obtain it through your approved channel.")
    return str(resolved)


def model_stamp(path):
    path = Path(path)
    return digest([(str(f.relative_to(path)), f.stat().st_size, f.stat().st_mtime_ns)
                   for f in sorted(path.rglob("*")) if f.is_file()])


def top_indices(scores, count, positive=False):
    scores = np.asarray(scores).reshape(-1)
    count = min(max(0, count), len(scores))
    if count == 0:
        return []
    # Deterministic tie breaking; corpus is tens of thousands of tickets, not billions.
    order = np.lexsort((np.arange(len(scores)), -scores))
    if positive:
        order = order[scores[order] > 0]
    return order[:count].tolist()


class HybridRetriever:
    def __init__(self, index_dir, device="cpu", reranker_path=None, rerank_device="cpu"):
        self.directory = Path(index_dir)
        self.data = joblib.load(self.directory/"lexical.joblib")
        if self.data["version"] != INDEX_VERSION:
            raise ValueError("Unsupported index version; rebuild the index.")
        self.rows = self.data["rows"]
        self.labels = {int(k):v for k,v in self.data["labels"].items()}
        self.device = device
        self.encoder = None
        self.reranker = None
        self.embeddings = None
        if self.data["embedding_model"]:
            self.embeddings = np.load(self.directory/"embeddings.npy", mmap_mode="r")
            if len(self.embeddings) != len(self.rows):
                raise ValueError("Index embedding row count mismatch; rebuild index.")
        if reranker_path:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(local_model(reranker_path), device=rerank_device,
                                        max_length=1024, local_files_only=True, trust_remote_code=False)

    @classmethod
    def build(cls, rows, labels, out_dir, embedding_model=None, device="cpu", batch_size=32):
        if not rows:
            raise ValueError("Cannot build an empty retrieval index")
        if any(r["type_id"] not in labels for r in rows):
            raise ValueError("Reference rows outside configured taxonomy")
        out = Path(out_dir)
        if (out/"lexical.joblib").exists():
            raise FileExistsError(f"Index exists: {out}. Use a new --out directory.")
        out.mkdir(parents=True, exist_ok=True)
        texts = [head_tail(r["description"], 6000) for r in rows]
        word = CountVectorizer(lowercase=True, ngram_range=(1,2), min_df=1,
                               max_features=160_000, token_pattern=r"(?u)\b\w+\b")
        counts = word.fit_transform(texts).tocsr().astype(np.float32)
        length = np.asarray(counts.sum(axis=1)).ravel()
        avg_length = max(float(length.mean()), 1.)
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(rows)-df+.5)/(df+.5))
        coo = counts.tocoo()
        k1, b = 1.5, .75
        weights = idf[coo.col]*coo.data*(k1+1)/(coo.data+k1*(1-b+b*length[coo.row]/avg_length))
        bm25 = sparse.csr_matrix((weights.astype(np.float32), (coo.row, coo.col)), shape=counts.shape)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1,
                               max_features=160_000, sublinear_tf=True, dtype=np.float32)
        char_matrix = char.fit_transform(texts).tocsr()
        embed_path = local_model(embedding_model) if embedding_model else None
        if embed_path:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(embed_path, device=device, local_files_only=True,
                                          trust_remote_code=False)
            encoder.max_seq_length = 1024
            # No retrieval instruction for documents. Qwen's query instruction is added at search time.
            embeddings = encoder.encode([head_tail(t, 3200) for t in texts], batch_size=batch_size,
                                         normalize_embeddings=True, convert_to_numpy=True,
                                         show_progress_bar=True).astype(np.float32)
            np.save(out/"embeddings.npy", embeddings)
        metadata = {"version": INDEX_VERSION, "rows": rows, "labels": labels,
                    "embedding_model": embed_path,
                    "embedding_stamp": model_stamp(embed_path) if embed_path else None,
                    "query_instruction": QUERY_INSTRUCTION, "word": word, "bm25": bm25,
                    "char": char, "char_matrix": char_matrix,
                    "fingerprint": digest([[r["row_key"], r["description"], r["type_id"], r.get("group_id")]
                                           for r in rows])}
        joblib.dump(metadata, out/"lexical.joblib", compress=3)
        write_json(out/"index_manifest.json", {k:v for k,v in metadata.items()
                                               if k not in {"rows","word","bm25","char","char_matrix"}} |
                   {"row_count":len(rows), "retrieval":"BM25 + char TF-IDF + dense cosine + reciprocal rank fusion"})
        return out

    def _encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            path = local_model(self.data["embedding_model"])
            if model_stamp(path) != self.data["embedding_stamp"]:
                raise ValueError("Embedding model changed since indexing. Rebuild index.")
            self.encoder = SentenceTransformer(path, device=self.device, local_files_only=True,
                                               trust_remote_code=False)
            self.encoder.max_seq_length = 1024
        return self.encoder

    def encode_queries(self, descriptions, batch_size=32):
        if self.embeddings is None:
            return [None]*len(descriptions)
        texts = ["Instruct: " + self.data["query_instruction"] + "\nQuery: " + head_tail(d,3200)
                 for d in descriptions]
        return self._encoder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=len(texts)>4)

    def retrieve(self, description, count=12, max_per_type=4, pool=80, rerank_top=48,
                 query_embedding=None, exclude_row_key=None, exclude_group=None):
        if count < 1 or max_per_type < 1 or pool < count or rerank_top < count:
            raise ValueError("Require count/max_per_type >=1 and pool/rerank_top >= count")
        text = head_tail(description,6000)
        q = self.data["word"].transform([text])
        q.data[:] = 1  # BM25 query presence rather than ticket ID repetition
        lexical = np.asarray((q @ self.data["bm25"].T).toarray()).ravel()
        char_q = self.data["char"].transform([text])
        char_scores = np.asarray((char_q @ self.data["char_matrix"].T).toarray()).ravel()
        ranks = [top_indices(lexical,pool,True), top_indices(char_scores,pool,True)]
        if self.embeddings is not None:
            if query_embedding is None:
                query_embedding = self.encode_queries([description])[0]
            ranks.append(top_indices(self.embeddings @ query_embedding,pool))
        fused = Counter()
        for ranked in ranks:
            for rank, idx in enumerate(ranked,1):
                row = self.rows[idx]
                if exclude_row_key and row["row_key"] == exclude_row_key:
                    continue
                if exclude_group and row.get("group_id") == exclude_group:
                    continue
                fused[idx] += 1/(60+rank)
        ordered = sorted(fused, key=lambda i:(-fused[i],i))
        rerank_scores = {}
        if self.reranker is not None and ordered:
            ordered = ordered[:rerank_top]
            scores = np.asarray(self.reranker.predict(
                [[head_tail(description,1800),head_tail(self.rows[i]["description"],1800)] for i in ordered],
                batch_size=16, show_progress_bar=False)).reshape(-1)
            rerank_scores = dict(zip(ordered, map(float,scores)))
            ordered.sort(key=lambda i:(-rerank_scores[i], -fused[i],i))
        selected, used, seen_text = [], Counter(), set()
        for idx in ordered:
            row = self.rows[idx]
            key = normalize(row["description"])
            if used[row["type_id"]] >= max_per_type or key in seen_text:
                continue
            selected.append(dict(row, rrf_score=float(fused[idx]), rerank_score=rerank_scores.get(idx)))
            used[row["type_id"]] += 1
            seen_text.add(key)
            if len(selected) >= count:
                break
        return selected

    def release_neural_models(self):
        # Useful when query evidence is precomputed on GPU before launching vLLM.
        self.encoder = self.reranker = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
