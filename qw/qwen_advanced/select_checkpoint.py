"""Select an epoch adapter by generated validation macro F1, in the inference env.

Runs each evaluation in a separate process so vLLM releases GPU memory between
checkpoints. Selection is restricted to the exact validation partition from training.
The untouched test set is never used for checkpoint selection.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ewoc_common import digest,load_labels,read_rows,write_json
from ewoc_classifier import add_inference_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_inference_args(p)
    p.add_argument("--run",default="runs/ewoc_lora")
    p.add_argument("--validation-csv",default="data/validation.csv")
    p.add_argument("--out",default="results/checkpoint_selection")
    args = p.parse_args()
    if args.adapter:
        raise ValueError("Use --run to select checkpoints, not --adapter")
    run = Path(args.run)
    metadata = json.loads((run/"run_metadata.json").read_text())
    rows = read_rows(args.validation_csv,load_labels(args.labels))
    if digest(sorted(r["row_key"] for r in rows)) != metadata["validation_row_fingerprint"]:
        raise ValueError("Selection must use the original complete validation partition, never the test partition")
    checkpoints = sorted((run/"checkpoints").glob("checkpoint-*"),
                         key=lambda d:int(d.name.rsplit("-",1)[1]))
    checkpoints = [d for d in checkpoints if (d/"adapter_config.json").exists()
                   and (d/"ewoc_training_metadata.json").exists()]
    if not checkpoints:
        raise ValueError("No EWOC epoch checkpoints found")
    selected_out = run/"adapter_best_f1"
    if selected_out.exists():
        raise FileExistsError(f"{selected_out} exists. Move it or use a separate run; selection never overwrites it.")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh --out directory for reproducible selection")
    out.mkdir(parents=True,exist_ok=True)
    results = []
    for checkpoint in checkpoints:
        prefix = out/checkpoint.name
        command = [sys.executable,str(Path(__file__).with_name("evaluate_classifier.py")),
                   "--model",args.model,"--adapter",str(checkpoint),"--labels",args.labels,
                   "--definitions",args.definitions,"--eval-csv",args.validation_csv,"--out",str(prefix),
                   "--index",args.index,"--retrieval-device",args.retrieval_device,
                   "--examples",str(args.examples),"--max-per-type",str(args.max_per_type),
                   "--pool",str(args.pool),"--rerank-top",str(args.rerank_top),
                   "--max-model-len",str(args.max_model_len),"--batch-size",str(args.batch_size),
                   "--gpu-memory-utilization",str(args.gpu_memory_utilization),"--seed",str(args.seed)]
        if args.no_rag:
            command.append("--no-rag")
        if args.reranker:
            command.extend(["--reranker",args.reranker])
        subprocess.run(command,check=True)
        report = json.loads(Path(str(prefix)+"_metrics.json").read_text())
        results.append({"checkpoint":str(checkpoint),"macro_f1":report["macro_f1"],
                        "accuracy":report["accuracy"],"weighted_f1":report["weighted_f1"]})
        write_json(out/"selection_progress.json",results)
    # Stable tie: earlier checkpoint, reducing unnecessary training if scores tie.
    best = max(results,key=lambda r:r["macro_f1"])
    source = Path(best["checkpoint"])
    selected_out.mkdir(parents=True)
    # Copy adapter + tokenizer + metadata, not multi-GB optimizer/scheduler state.
    for file in source.iterdir():
        if file.is_file() and (file.name.startswith("adapter_") or
                              file.name.startswith("tokenizer") or
                              file.name in {"special_tokens_map.json","added_tokens.json","vocab.json",
                                            "merges.txt","chat_template.jinja","labels.json",
                                            "label_definitions.json","ewoc_training_metadata.json"}):
            shutil.copy2(file,selected_out/file.name)
    selected_meta = json.loads((selected_out/"ewoc_training_metadata.json").read_text())
    selected_meta.update({"selection_metric":"generated validation macro F1",
                          "selected_checkpoint":str(source),"selection_scores":best,
                          "selection_inference_configuration":vars(args)})
    write_json(selected_out/"ewoc_training_metadata.json",selected_meta)
    write_json(out/"selection_result.json",{"best":best,"all_checkpoints":results,
                                            "selected_adapter":str(selected_out),"configuration":vars(args)})
    print(f"Selected adapter: {selected_out.resolve()} (validation macro F1 {best['macro_f1']:.4f})")
    print("Keep this inference configuration for final test evaluation.")


if __name__ == "__main__":
    main()
