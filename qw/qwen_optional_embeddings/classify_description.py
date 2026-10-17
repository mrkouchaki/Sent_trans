"""Classify one description or a UTF-8 text file; emit JSON plus retrieval evidence."""
import argparse
import json
from pathlib import Path

from ewoc_common import load_labels, write_json
from ewoc_classifier import (QwenClassifier, add_inference_args, make_retriever, retrieve_many)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_inference_args(p)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--description")
    source.add_argument("--text-file", help="Supports multiline ticket descriptions")
    p.add_argument("--out", default=None, help="Optional complete JSON result including evidence")
    args = p.parse_args()
    description = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else args.description
    if description is None:
        description = input("EWOC ticket description: ").strip()
    if not description.strip():
        raise ValueError("Description must not be blank")
    labels = load_labels(args.labels)
    retriever = make_retriever(args,labels)
    evidence = retrieve_many(retriever,[{"description":description}],args)
    classifier = QwenClassifier(args,labels)
    result = classifier.classify_batch([description],evidence)[0]
    print(json.dumps({k:v for k,v in result.items() if k != "evidence"},indent=2))
    if args.out:
        write_json(args.out,result)
        print(f"Evidence saved: {args.out}")


if __name__ == "__main__":
    main()
