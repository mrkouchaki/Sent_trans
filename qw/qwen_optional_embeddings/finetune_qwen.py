"""Single-H100 BF16 LoRA or NF4 QLoRA for Qwen3-8B.

Completion-only supervision; real grouped train/validation CSVs; epoch checkpoints;
best adapter selected by validation loss. Test data is never opened by training.
Optional mixed RAG examples are retrieved only from the training partition.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ewoc_common import (DEFAULT_DEFINITIONS, DEFAULT_LABELS, assert_disjoint, digest,
                         label_signature, load_definitions, load_labels, normalize,
                         read_rows, write_json)
from ewoc_retrieval import HybridRetriever, local_model, model_stamp
from ewoc_sft_data import encode_completion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="models/Qwen3-8B")
    p.add_argument("--train-csv",default="data/train.csv")
    p.add_argument("--validation-csv",default="data/validation.csv")
    p.add_argument("--labels",default=str(DEFAULT_LABELS))
    p.add_argument("--definitions",default=str(DEFAULT_DEFINITIONS))
    p.add_argument("--output",default="runs/ewoc_lora")
    p.add_argument("--quantization",choices=["none","nf4"],default="none",
                   help="none=BF16 LoRA (H100 default); nf4=4-bit QLoRA with BF16 compute")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--gradient-accumulation",type=int,default=16)
    p.add_argument("--max-length",type=int,default=2048)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--rank",type=int,choices=[8,16,32,64],default=32)
    p.add_argument("--lora-alpha",type=int,default=64)
    p.add_argument("--lora-dropout",type=float,default=.05)
    p.add_argument("--max-per-class",type=int,default=None,help="Optional TRAIN-only cap; no oversampling")
    p.add_argument("--patience",type=int,default=2,help="Early stopping by validation loss")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--resume",default=None,help="Checkpoint directory, or latest; restores optimizer/scheduler")
    p.add_argument("--rag-index",default=None,help="Optional index built solely from this train.csv")
    p.add_argument("--rag-probability",type=float,default=.5,
                   help="With --rag-index, train a mix of retrieved and zero-shot prompts")
    p.add_argument("--rag-examples",type=int,default=4)
    p.add_argument("--retrieval-device",default="cpu")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE","1")) != 1:
        raise ValueError("This script targets one H100. Launch with python, not torchrun.")
    if not 0 <= args.rag_probability <= 1:
        raise ValueError("--rag-probability must be between 0 and 1")
    if min(args.epochs,args.batch_size,args.gradient_accumulation,args.patience) < 1:
        raise ValueError("Epochs, batch size, accumulation, patience must be positive")
    if args.max_length < 512 or args.learning_rate <= 0:
        raise ValueError("Use max-length >=512 and positive learning rate")
    if args.max_per_class is not None and args.max_per_class < 1:
        raise ValueError("--max-per-class must be positive")
    labels = load_labels(args.labels)
    definitions = load_definitions(args.definitions,labels)
    train = read_rows(args.train_csv,labels)
    validation = read_rows(args.validation_csv,labels)
    if not all(r.get("group_id") for r in train+validation):
        raise ValueError("Run prepare_data.py first; grouped CSVs are required")
    if any(r["is_synthetic"] for r in train+validation):
        raise ValueError("This workflow expects real tickets. Synthetic lineage is not validated here.")
    assert_disjoint(train,validation,"training versus validation")
    for name,rows in [("train",train),("validation",validation)]:
        missing = set(labels)-{r["type_id"] for r in rows}
        if missing:
            raise ValueError(f"{name} missing configured classes: {missing}")
    # Keep all source-training exposure hashes, including possible retrieved demonstrations.
    source_train = train[:]
    rng = random.Random(args.seed)
    if args.max_per_class:
        by_class = defaultdict(list)
        for row in train:
            by_class[row["type_id"]].append(row)
        train = []
        for key in sorted(by_class):
            rng.shuffle(by_class[key])
            train.extend(by_class[key][:args.max_per_class])
        rng.shuffle(train)
    base = local_model(args.model)
    output = Path(args.output)
    checkpoint_dir = output/"checkpoints"
    # Do not mix runs or silently overwrite an existing adapter.
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} is nonempty. Use --resume latest or a new --output.")
    output.mkdir(parents=True,exist_ok=True)
    metadata = {"base_model_path":base,"base_model_stamp":model_stamp(base),
                "label_signature":label_signature(labels,definitions),"labels":labels,
                "definitions":definitions,"lora_rank":args.rank,"quantization":args.quantization,
                "training_row_keys":sorted({r["row_key"] for r in source_train}),
                "training_group_ids":sorted({r["group_id"] for r in source_train}),
                "training_ticket_hashes":sorted({digest(r["ewoc_id"]) for r in source_train if r["ewoc_id"]}),
                "training_text_hashes":sorted({digest(normalize(r["description"])) for r in source_train}),
                "validation_row_fingerprint":digest(sorted(r["row_key"] for r in validation)),
                "selection_metric":"validation completion loss (not classification macro F1)",
                "training_class_counts":dict(Counter(r["type_id"] for r in train)),
                "rag_mixture":bool(args.rag_index),"config":vars(args)}
    run_config = {k:v for k,v in vars(args).items() if k != "resume"}
    metadata["run_signature"] = digest({"config":run_config,"train":metadata["training_row_keys"],
                                         "validation":metadata["validation_row_fingerprint"],
                                         "taxonomy":metadata["label_signature"],"base":metadata["base_model_stamp"]})
    metadata_path = output/"run_metadata.json"
    if args.resume:
        if not metadata_path.exists():
            raise ValueError("Resume requires existing run_metadata.json")
        previous = json.loads(metadata_path.read_text())
        if previous["run_signature"] != metadata["run_signature"]:
            raise ValueError("Resume data/model/config changed. Use original arguments or start a new run.")
    train_examples = [[] for _ in train]
    val_examples = [[] for _ in validation]
    if args.rag_index:
        retriever = HybridRetriever(args.rag_index,device=args.retrieval_device)
        if retriever.labels != labels:
            raise ValueError("RAG index taxonomy differs from training labels")
        source_keys = {r["row_key"] for r in source_train}
        if not {r["row_key"] for r in retriever.rows}.issubset(source_keys):
            raise ValueError("Training RAG index includes rows outside train.csv")
        assert_disjoint(retriever.rows,validation,"training RAG references versus validation")
        def evidence_for(rows):
            selected = [i for i in range(len(rows)) if rng.random() < args.rag_probability]
            result = [[] for _ in rows]
            vectors = retriever.encode_queries([rows[i]["description"] for i in selected]) if selected else []
            for i,q in zip(selected,vectors):
                r = rows[i]
                result[i] = retriever.retrieve(r["description"],count=args.rag_examples,max_per_type=2,
                                               query_embedding=q,exclude_row_key=r["row_key"],
                                               exclude_group=r["group_id"])
            return result
        train_examples = evidence_for(train)
        val_examples = evidence_for(validation)
        retriever.release_neural_models()
        metadata["training_rag_index_fingerprint"] = retriever.data["fingerprint"]
        if args.resume and previous.get("training_rag_index_fingerprint") != metadata["training_rag_index_fingerprint"]:
            raise ValueError("RAG index changed since checkpoint")
    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required (H100 supported).")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    class CompletionDataset(Dataset):
        def __init__(self,rows,evidence):
            self.items = []
            shortened = 0
            for row,examples in zip(rows,evidence):
                encoded,stats = encode_completion(tokenizer,row,examples,labels,definitions,args.max_length)
                self.items.append(encoded)
                shortened += stats["description_char_budget"] < min(6000,len(row["description"]))
            print(f"Tokenized {len(self.items)} tickets; further shortened {shortened} long descriptions")
        def __len__(self):
            return len(self.items)
        def __getitem__(self,index):
            return self.items[index]
    train_dataset = CompletionDataset(train,train_examples)
    val_dataset = CompletionDataset(validation,val_examples)
    def collate(items):
        length = ((max(len(i["input_ids"]) for i in items)+7)//8)*8
        batch = {}
        for key,pad in [("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100)]:
            batch[key] = torch.tensor([item[key]+[pad]*(length-len(item[key])) for item in items],dtype=torch.long)
        return batch
    model_kwargs = dict(local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,
                        device_map={"":0},attn_implementation="sdpa")
    if args.quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base,**model_kwargs)
    model.config.use_cache = False
    if args.quantization == "nf4":
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant":False})
    model = get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=args.rank,
                                          lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
                                          target_modules="all-linear",bias="none"))
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",warmup_ratio=.05,weight_decay=.01,max_grad_norm=1.,
        bf16=True,tf32=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        optim="adamw_torch",eval_strategy="epoch",save_strategy="epoch",
        save_total_limit=None,save_safetensors=True,save_only_model=False,
        load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,
        prediction_loss_only=True,logging_steps=10,report_to="none",push_to_hub=False,
        seed=args.seed,data_seed=args.seed,dataloader_num_workers=0,
        remove_unused_columns=False,label_names=["labels"])
    class EpochMetadataCallback(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            checkpoint = Path(args.output_dir)/f"checkpoint-{state.global_step}"
            write_json(checkpoint/"ewoc_training_metadata.json",dict(metadata,checkpoint_epoch=state.epoch))
            write_json(checkpoint/"labels.json",labels)
            write_json(checkpoint/"label_definitions.json",definitions)
    trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,
                      eval_dataset=val_dataset,data_collator=collate,processing_class=tokenizer,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience),EpochMetadataCallback()])
    resume = None
    if args.resume:
        resume = get_last_checkpoint(str(checkpoint_dir)) if args.resume == "latest" else args.resume
        if not resume or not Path(resume).is_dir():
            raise ValueError("No resumable checkpoint found")
        if Path(resume).resolve().parent != checkpoint_dir.resolve():
            raise ValueError("Resume checkpoint must be inside this run's checkpoints/ directory")
    for package in ["torch","transformers","peft","accelerate","bitsandbytes"]:
        try:
            metadata.setdefault("packages",{})[package] = version(package)
        except PackageNotFoundError:
            pass
    write_json(metadata_path,metadata)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_metrics("train",result.metrics)
    trainer.save_metrics("eval",trainer.evaluate())
    adapter_dir = output/"adapter"
    trainer.save_model(str(adapter_dir))  # best eval-loss checkpoint is loaded at this point
    tokenizer.save_pretrained(adapter_dir)
    metadata["best_checkpoint"] = trainer.state.best_model_checkpoint
    metadata["best_validation_loss"] = trainer.state.best_metric
    metadata["training_finished"] = True
    write_json(adapter_dir/"ewoc_training_metadata.json",metadata)
    write_json(adapter_dir/"labels.json",labels)
    write_json(adapter_dir/"label_definitions.json",definitions)
    write_json(output/"training_history.json",trainer.state.log_history)
    write_json(metadata_path,metadata)
    print(f"Best adapter: {adapter_dir.resolve()}")
    print(f"Every epoch checkpoint (optimizer + scheduler included): {checkpoint_dir.resolve()}")
    print("Evaluate macro F1 using evaluate_classifier.py; validation loss is not classification accuracy.")


if __name__ == "__main__":
    main()
