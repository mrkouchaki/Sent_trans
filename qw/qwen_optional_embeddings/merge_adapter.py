"""Optional: merge LoRA into a BF16 base for ordinary vLLM serving (no adapter flag)."""
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
import argparse
import json
from pathlib import Path

from ewoc_common import write_json
from ewoc_retrieval import local_model, model_stamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="models/Qwen3-8B")
    p.add_argument("--adapter",default="runs/ewoc_lora/adapter")
    p.add_argument("--out",default="models/Qwen3-8B-EWOC")
    args = p.parse_args()
    base,adapter = local_model(args.model),local_model(args.adapter)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty/new output directory")
    metadata = json.loads((Path(adapter)/"ewoc_training_metadata.json").read_text())
    if model_stamp(base) != metadata["base_model_stamp"]:
        raise ValueError("Base files differ from the training base. Use the exact original local model.")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,
                                                device_map={"":0},local_files_only=True,
                                                trust_remote_code=False)
    model = PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()
    model.save_pretrained(out,safe_serialization=True,max_shard_size="4GB")
    AutoTokenizer.from_pretrained(adapter,local_files_only=True).save_pretrained(out)
    write_json(out/"ewoc_training_metadata.json",metadata)
    write_json(out/"labels.json",metadata["labels"])
    write_json(out/"label_definitions.json",metadata["definitions"])
    print(f"Merged BF16 model: {out.resolve()}")


if __name__ == "__main__":
    main()
