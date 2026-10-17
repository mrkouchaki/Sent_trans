"""Build retrieval index from TRAINING tickets only; run once per split/model."""
import argparse
from pathlib import Path

from ewoc_common import DEFAULT_LABELS, audit, load_labels, read_rows
from ewoc_retrieval import HybridRetriever
from ewoc_embeddings import KINDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default="data/train.csv")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default="indexes/lexical")
    p.add_argument("--embedding-model", default=None, help="Optional local MiniLM, BERT, or Qwen embedding model. Omit to disable dense retrieval.")
    p.add_argument("--embedding-kind", choices=KINDS, default="auto")
    p.add_argument("--embedding-max-length", type=int, default=None, help="Optional lower token cap; never exceeds the model's saved/native limit")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    rows = read_rows(args.train_csv, load_labels(args.labels))
    if not all(r.get("group_id") for r in rows):
        raise ValueError("Run prepare_data.py first; index requires grouped training CSV.")
    if audit(rows)["conflicting_descriptions"]:
        raise ValueError("Conflicting labels remain in training data.")
    path = HybridRetriever.build(rows, load_labels(args.labels), args.out,
                                 args.embedding_model, args.device,args.batch_size,
                                 args.embedding_kind,args.embedding_max_length)
    print(f"Built {path.resolve()} from {len(rows)} training rows")
    print("Dense channel:", "enabled" if args.embedding_model else "disabled (BM25 + character TF-IDF retrieval)")


if __name__ == "__main__":
    main()
