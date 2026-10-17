"""EWOC commands: prepare, index, predict, evaluate, test, train, select, merge, serve, check."""
from ewoc_core import (
    DEFAULT_DEFINITIONS,
    DEFAULT_LABELS,
    HybridRetriever,
    KINDS,
    QUERY_INSTRUCTION,
    QwenClassifier,
    add_inference_args,
    allowed_label_tokens,
    assert_adapter_holdout,
    assert_disjoint,
    audit,
    digest,
    encode_completion,
    encode_texts,
    fit_prompt,
    format_embedding_text,
    head_tail,
    inspect_embedding_profile,
    label_signature,
    load_definitions,
    load_embedding_encoder,
    load_labels,
    local_embedding_model,
    local_model,
    make_messages,
    make_retriever,
    model_stamp,
    normalize,
    read_rows,
    retrieve_many,
    template_key,
    write_json,
    write_rows,
)

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import json
import csv
import platform
from importlib.metadata import PackageNotFoundError, version
import os
import random
import shutil
import subprocess
import sys
from threading import Lock
import unittest
import evaluation
import training
from data_preparation import DataPreparer, prepare
from evaluation import metrics, select_evaluation_rows, export_metric_tables

# ====================== CONFIG: EDIT HERE ==========================
# Run `python ewoc.py`. Training/evaluation resolve their dependencies automatically.
ROOT = Path(__file__).resolve().parent
ACTION = 'test'                      # prepare/index/predict/evaluate/test/train/select/merge/serve/check
GPU_ID = '0'                         # Allocated physical GPU; existing CUDA_VISIBLE_DEVICES takes priority
RANDOM_SEED = 2026                   # Reproducible splitting, sampling and training
MODEL_PATH = ROOT / 'models/Qwen3-8B' # LOCAL folder with weights, config and tokenizer
ADAPTER_PATH = None                  # None=base; or ROOT/'runs/ewoc_lora/adapter_best_f1'
LABEL_CONFIG = ROOT / 'labels.json'  # Names + business definitions; review marked draft definitions
DEFINITIONS_PATH = None              # None=use labels.json; optional legacy definitions file

# Raw data and split policy: used automatically, including when ACTION='train'.
PREPARE = {
    'csv': ROOT / 'whole_ewoc_tickets.csv', # Original labeled CSV
    'out': ROOT / 'data',             # Generated train/validation/test CSVs
    'folds': 10,                      # 10≈80/10/10; 5≈60/20/20; need >=folds groups/class
    'seed': 2026,                     # Split seed is independent of training/sampling seed
    'group_templates': False,         # True=also group numeric template variants (heuristic)
}
TRAIN_CSV = None                     # None=prepare/reuse automatically; or custom grouped training CSV
VALIDATION_CSV = None                # None=automatic; set with custom TRAIN_CSV
TEST_CSV = None                      # None=prepared test split; or your independent labeled CSV

EMBEDDING = {
    'embedding_model': None,          # None=off; local models/bert, models/all-MiniLM-L6-v2, etc.
    'embedding_kind': 'auto',         # auto/bert/minilm/sentence-transformer/qwen (embedding checkpoint)
    'embedding_max_length': None,     # None=native cap; integer can lower it
    'device': 'cpu',                  # cpu/cuda:0; building dense index may use GPU
    'batch_size': 32,                 # Encoder batch size; lower if out of memory
}
# Automatically use a separate index folder when selecting an embedding model.
INDEX_PATH = ROOT / 'indexes' / ('lexical' if EMBEDDING['embedding_model'] is None
                                else Path(EMBEDDING['embedding_model']).name)
USE_RAG = True                       # False=classify directly with Qwen, without retrieval
RAG = {
    'reranker': None,                # None=off; or local BGE reranker folder
    'retrieval_device': 'cpu',       # CPU keeps inference GPU memory free; API requires cpu
    'examples': 12,                  # Maximum retrieved examples in inference prompts
    'max_per_type': 4,               # Maximum examples from each class
    'pool': 80,                      # Retrieval candidate pool
    'rerank_top': 48,                # Candidates scored by optional reranker
}
INFERENCE = {
    'backend': 'transformers',       # transformers=PyTorch locally; vllm=optional faster engine
    'inference_device': 'cuda:0',    # cuda:0/cpu (CPU is slower)
    'inference_dtype': 'bfloat16',   # bfloat16=H100; float16/float32 also supported
    'max_model_len': 8192,           # Prompt+completion budget; lower if memory is limited
    'gpu_memory_utilization': 0.80,  # vLLM only
    'batch_size': 4,                 # Evaluation chunk size; Transformers processes one prompt at a time
}
EVALUATION = {
    'eval_csv': VALIDATION_CSV,      # Validation is for choosing settings/checkpoints
    'out': ROOT / 'results/validation',
    'max_per_class': 200,            # None=all rows; otherwise sample up to this count per class
    'samples_by_class': {},          # Optional overrides: {1: 100, 2: 200, 40: 80}
    'strict_samples': False,         # True=fail if requested counts are unavailable
}
TEST = {
    'eval_csv': TEST_CSV,            # Final or independent test set; never starts training
    'out': ROOT / 'results/test',
    'max_per_class': 200,            # None=all; no sampling with replacement
    'samples_by_class': {},          # Optional per-class overrides, as above
    'strict_samples': False,         # False=use available rows and report shortages
}
FINETUNE = {
    'method': 'lora',                # lora=BF16 LoRA; qlora=4-bit NF4 LoRA (not full-weight training)
    'output': ROOT / 'runs/ewoc_lora',# New run folder; change for each experiment
    'epochs': 3,
    'batch_size': 2,                 # H100 starting value; lower if memory is limited
    'gradient_accumulation': 16,     # Effective batch=2*16 on one GPU
    'max_length': 2048,              # Training token budget; >=512
    'learning_rate': 1e-4,
    'rank': 32,                      # 8/16/32/64
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    'target_modules': 'all-linear', # Or a list of module names, e.g. ['q_proj','v_proj']
    'max_per_class': None,           # Optional TRAIN-only class cap; None=all real rows
    'patience': 2,                   # Early stopping by validation completion loss
    'resume': None,                  # None=new; 'latest'=resume this same run/config
    'rag_index': None,               # None=ordinary SFT; INDEX_PATH=auto-build/reuse RAG for training
    'rag_probability': 0.5,          # Fraction using RAG when rag_index is set
    'rag_examples': 4,
    'retrieval_device': 'cpu',
    'lr_scheduler_type': 'cosine',   # cosine/linear/constant
    'warmup_ratio': 0.05,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'logging_steps': 10,
    'optim': 'adamw_torch',          # Transformers optimizer name; torch AdamW by default
    'gradient_checkpointing': True, # Saves activation memory
    'tf32': True,                   # H100 TF32 matmul support
}
SELECTION = {
    'run': FINETUNE['output'],       # Select saved epochs by full validation macro F1
    'validation_csv': VALIDATION_CSV,
    'out': ROOT / 'results/checkpoint_selection', # Fresh output directory required
}
MERGE = {
    'adapter': ROOT / 'runs/ewoc_lora/adapter', # Or adapter_best_f1
    'out': ROOT / 'models/Qwen3-8B-EWOC',      # New merged BF16 model folder
}
PREDICT = {
    'description': None,            # None=interactive input; or a ticket string
    'text_file': None,              # Optional local UTF-8 file instead of description
    'out': ROOT / 'results/prediction.json',
}
SERVER = {'host': '127.0.0.1', 'port': 8000} # Local endpoint; one worker
# Offline loading is enforced in ewoc_core.py; no login, Hub ID or download.
# ==================================================================


def configured_defaults(command):
    """One source of defaults for scripts and optional CLI, without model imports."""
    shared = {'model': MODEL_PATH, 'labels': LABEL_CONFIG, 'definitions': DEFINITIONS_PATH,
              'seed': RANDOM_SEED}
    inference = dict(shared, adapter=ADAPTER_PATH, index=INDEX_PATH, no_rag=not USE_RAG,
                     **RAG, **INFERENCE)
    sections = {
        'prepare': dict(labels=LABEL_CONFIG, **PREPARE),
        'index': dict(train_csv=TRAIN_CSV, labels=LABEL_CONFIG, out=INDEX_PATH, **EMBEDDING),
        'predict': dict(inference, **PREDICT),
        'evaluate': dict(inference, **EVALUATION),
        'test': dict(inference, **TEST),
        'train': dict(shared, train_csv=TRAIN_CSV, validation_csv=VALIDATION_CSV, **FINETUNE),
        'select': dict(inference, **SELECTION),
        'merge': dict(model=MODEL_PATH, **MERGE),
        'serve': dict(inference, **SERVER),
        'check': {},
    }
    # argparse paths remain strings, making provenance JSON and subprocesses portable.
    return {k: str(v) if isinstance(v, Path) else v for k, v in sections[command].items()}


def parse_settings(parser, command, argv=None, overrides=None):
    defaults = configured_defaults(command)
    defaults.update(overrides or {})
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for option in parser._actions:
        value = getattr(args, option.dest, None)
        if option.choices and value is not None and value not in option.choices:
            raise ValueError(f'Invalid CONFIG value for {option.dest}: {value}; options: {option.choices}')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
    return args


def data_preparer(labels_path):
    return DataPreparer(PREPARE['csv'], PREPARE['out'], labels_path,
                        PREPARE['folds'], PREPARE['seed'], PREPARE['group_templates'])


def embedding_settings():
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in EMBEDDING.items()}


def resolve_retrieval_dependency(args):
    if args.no_rag:
        return
    # A deployed, resolved index is usable without carrying the original source CSV.
    if not Path(PREPARE['csv']).is_file() and (Path(args.index)/'lexical.joblib').is_file():
        return
    train_csv = TRAIN_CSV
    if train_csv is None:
        train_csv = data_preparer(args.labels).ensure().train_csv
    labels = load_labels(args.labels)
    rows = read_rows(train_csv, labels)
    args.index = str(HybridRetriever.ensure(rows, labels, args.index, **embedding_settings()))


def resolve_evaluation_dependencies(args, command):
    if args.eval_csv is None:
        dataset = data_preparer(args.labels).ensure()
        args.eval_csv = str(dataset.test_csv if command == 'test' else dataset.validation_csv)
    resolve_retrieval_dependency(args)


def prepare_main(argv=None):
    p = argparse.ArgumentParser(description='Audit CSV and create reproducible grouped train/validation/test splits.', prog='ewoc.py prepare')
    p.add_argument('--csv', default='whole_ewoc_tickets.csv')
    p.add_argument('--out', default='data')
    p.add_argument('--labels', default=str(DEFAULT_LABELS))
    p.add_argument('--folds', type=int, default=10, help='Two folds held out; default approximately 80/10/10')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--group-templates', action='store_true')
    args = parse_settings(p, 'prepare', argv)
    result = DataPreparer(args.csv, args.out, args.labels, args.folds, args.seed,
                          args.group_templates).ensure()
    print(f'Prepared splits and audit: {result.directory}')
    return result

def index_main(argv=None):
    p = argparse.ArgumentParser(description='Build retrieval index from TRAINING tickets only; run once per split/model.', prog='ewoc.py index')
    p.add_argument('--train-csv', default='data/train.csv')
    p.add_argument('--labels', default=str(DEFAULT_LABELS))
    p.add_argument('--out', default='indexes/lexical')
    p.add_argument('--embedding-model', default=None, help='Optional local MiniLM, BERT, or Qwen embedding model. Omit to disable dense retrieval.')
    p.add_argument('--embedding-kind', choices=KINDS, default='auto')
    p.add_argument('--embedding-max-length', type=int, default=None, help="Optional lower token cap; never exceeds the model's saved/native limit")
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32)
    args = parse_settings(p, 'index', argv)
    if args.train_csv is None:
        args.train_csv = str(data_preparer(args.labels).ensure().train_csv)
    rows = read_rows(args.train_csv, load_labels(args.labels))
    if not all(r.get('group_id') for r in rows):
        raise ValueError('Custom training CSV needs GROUP_ID; leave TRAIN_CSV=None for automatic preparation.')
    path = HybridRetriever.ensure(rows, load_labels(args.labels), args.out,
                                 args.embedding_model, args.device, args.batch_size,
                                 args.embedding_kind, args.embedding_max_length)
    print(f'Ready index: {path}')
    return path

def predict_main(argv=None):
    p = argparse.ArgumentParser(description='Classify one description or a UTF-8 text file; emit JSON plus retrieval evidence.', prog='ewoc.py predict')
    add_inference_args(p)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--description')
    source.add_argument('--text-file', help='Supports multiline ticket descriptions')
    p.add_argument('--out', default=None, help='Optional complete JSON result including evidence')
    args = parse_settings(p, 'predict', argv)
    description = Path(args.text_file).read_text(encoding='utf-8') if args.text_file else args.description
    if args.text_file and args.description is not None:
        raise ValueError('Set only one of PREDICT description or text_file')
    if description is None:
        description = input('EWOC ticket description: ').strip()
    if not description.strip():
        raise ValueError('Description must not be blank')
    labels = load_labels(args.labels)
    resolve_retrieval_dependency(args)
    retriever = make_retriever(args, labels)
    evidence = retrieve_many(retriever, [{'description': description}], args)
    classifier = QwenClassifier(args, labels)
    result = classifier.classify_batch([description], evidence)[0]
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}, indent=2))
    if args.out:
        write_json(args.out, result)
        print(f'Evidence saved: {args.out}')

def evaluate_main(argv=None, _command='evaluate', settings=None):
    p = argparse.ArgumentParser(description='Evaluate base/LoRA Qwen with or without retrieval on a disjoint held-out CSV.', prog='ewoc.py evaluate')
    add_inference_args(p)
    p.add_argument('--eval-csv', default='data/validation.csv')
    p.add_argument('--out', default='results/base_rag_validation')
    p.add_argument('--max-per-class', '--samples-per-class', dest='max_per_class', type=int, default=None,
                   help='Up to this many real holdout rows per class; sampling is without replacement')
    p.add_argument('--samples-by-class', default=None, help='Optional JSON overrides, e.g. {"1":100,"2":200}')
    p.add_argument('--strict-samples', action=argparse.BooleanOptionalAction, help='Fail if any requested class count is unavailable')
    p.add_argument('--all-samples', dest='max_per_class', action='store_const', const=None, help='Use all rows per class')
    args = parse_settings(p, _command, argv, settings)
    resolve_evaluation_dependencies(args, _command)
    return evaluation.run_evaluation(args)


def test_main(argv=None):
    """A separate model-test run; this never starts fine-tuning."""
    return evaluate_main(argv, _command='test')

def train_main(argv=None):
    p = argparse.ArgumentParser(description='Single-H100 BF16 LoRA or NF4 QLoRA for Qwen3-8B.', prog='ewoc.py train')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--train-csv', default='data/train.csv')
    p.add_argument('--validation-csv', default='data/validation.csv')
    p.add_argument('--labels', default=str(DEFAULT_LABELS))
    p.add_argument('--definitions', default=None, help='Optional legacy definition file; otherwise use --labels')
    p.add_argument('--output', default='runs/ewoc_lora')
    p.add_argument('--method', choices=['lora', 'qlora'], default='lora')
    p.add_argument('--quantization', choices=['none', 'nf4'], default=None, help='none=BF16 LoRA (H100 default); nf4=4-bit QLoRA with BF16 compute')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--gradient-accumulation', type=int, default=16)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--learning-rate', type=float, default=0.0001)
    p.add_argument('--rank', type=int, choices=[8, 16, 32, 64], default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--max-per-class', type=int, default=None, help='Optional TRAIN-only cap; no oversampling')
    p.add_argument('--patience', type=int, default=2, help='Early stopping by validation loss')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--resume', default=None, help='Checkpoint directory, or latest; restores optimizer/scheduler')
    p.add_argument('--rag-index', default=None, help='Optional index built solely from this train.csv')
    p.add_argument('--rag-probability', type=float, default=0.5, help='With --rag-index, train a mix of retrieved and zero-shot prompts')
    p.add_argument('--rag-examples', type=int, default=4)
    p.add_argument('--retrieval-device', default='cpu')
    p.add_argument('--target-modules', nargs='+')
    p.add_argument('--lr-scheduler-type', choices=['cosine', 'linear', 'constant'])
    p.add_argument('--warmup-ratio', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--max-grad-norm', type=float)
    p.add_argument('--logging-steps', type=int)
    p.add_argument('--optim')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction)
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction)
    args = parse_settings(p, 'train', argv)
    return training.TrainingPipeline(args, data_preparer(args.labels), embedding_settings()).run()

def select_main(argv=None):
    p = argparse.ArgumentParser(description='Select an epoch adapter by generated validation macro F1, in the inference env.', prog='ewoc.py select')
    add_inference_args(p)
    p.add_argument('--run', default='runs/ewoc_lora')
    p.add_argument('--validation-csv', default='data/validation.csv')
    p.add_argument('--out', default='results/checkpoint_selection')
    args = parse_settings(p, 'select', argv)
    if args.adapter:
        raise ValueError('Use --run to select checkpoints, not --adapter')
    run = Path(args.run)
    metadata = json.loads((run / 'run_metadata.json').read_text())
    if args.validation_csv is None:
        args.validation_csv = metadata.get('config', {}).get('validation_csv')
        if args.validation_csv is None:
            args.validation_csv = str(data_preparer(args.labels).ensure().validation_csv)
    resolve_retrieval_dependency(args)
    rows = read_rows(args.validation_csv, load_labels(args.labels))
    if digest(sorted((r['row_key'] for r in rows))) != metadata['validation_row_fingerprint']:
        raise ValueError('Selection must use the original complete validation partition, never the test partition')
    checkpoints = sorted((run / 'checkpoints').glob('checkpoint-*'), key=lambda d: int(d.name.rsplit('-', 1)[1]))
    checkpoints = [d for d in checkpoints if (d / 'adapter_config.json').exists() and (d / 'ewoc_training_metadata.json').exists()]
    if not checkpoints:
        raise ValueError('No EWOC epoch checkpoints found')
    selected_out = run / 'adapter_best_f1'
    if selected_out.exists():
        raise FileExistsError(f'{selected_out} exists. Move it or use a separate run; selection never overwrites it.')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh --out directory for reproducible selection')
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in checkpoints:
        prefix = out / checkpoint.name
        command = [sys.executable, str(Path(__file__).resolve()), 'evaluate', '--all-samples', '--samples-by-class', '{}', '--no-strict-samples', '--backend', args.backend, '--inference-device', args.inference_device, '--inference-dtype', args.inference_dtype, '--model', args.model, '--adapter', str(checkpoint), '--labels', args.labels, '--definitions', args.definitions or args.labels, '--eval-csv', args.validation_csv, '--out', str(prefix), '--index', args.index, '--retrieval-device', args.retrieval_device, '--examples', str(args.examples), '--max-per-type', str(args.max_per_type), '--pool', str(args.pool), '--rerank-top', str(args.rerank_top), '--max-model-len', str(args.max_model_len), '--batch-size', str(args.batch_size), '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--seed', str(args.seed)]
        command.append('--no-rag' if args.no_rag else '--rag')
        if args.reranker:
            command.extend(['--reranker', args.reranker])
        subprocess.run(command, check=True)
        report = json.loads(Path(str(prefix) + '_metrics.json').read_text())
        results.append({'checkpoint': str(checkpoint), 'macro_f1': report['macro_f1'], 'accuracy': report['accuracy'], 'weighted_f1': report['weighted_f1']})
        write_json(out / 'selection_progress.json', results)
    best = max(results, key=lambda r: r['macro_f1'])
    source = Path(best['checkpoint'])
    selected_out.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and (file.name.startswith('adapter_') or file.name.startswith('tokenizer') or file.name in {'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja', 'labels.json', 'label_definitions.json', 'ewoc_training_metadata.json'}):
            shutil.copy2(file, selected_out / file.name)
    selected_meta = json.loads((selected_out / 'ewoc_training_metadata.json').read_text())
    selected_meta.update({'selection_metric': 'generated validation macro F1', 'selected_checkpoint': str(source), 'selection_scores': best, 'selection_inference_configuration': vars(args)})
    write_json(selected_out / 'ewoc_training_metadata.json', selected_meta)
    write_json(out / 'selection_result.json', {'best': best, 'all_checkpoints': results, 'selected_adapter': str(selected_out), 'configuration': vars(args)})
    print(f"Selected adapter: {selected_out.resolve()} (validation macro F1 {best['macro_f1']:.4f})")
    print('Keep this inference configuration for final test evaluation.')

def merge_main(argv=None):
    p = argparse.ArgumentParser(description='Optional: merge LoRA into a BF16 base for ordinary vLLM serving (no adapter flag).', prog='ewoc.py merge')
    p.add_argument('--model', default='models/Qwen3-8B')
    p.add_argument('--adapter', default='runs/ewoc_lora/adapter')
    p.add_argument('--out', default='models/Qwen3-8B-EWOC')
    args = parse_settings(p, 'merge', argv)
    base, adapter = (local_model(args.model), local_model(args.adapter))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty/new output directory')
    metadata = json.loads((Path(adapter) / 'ewoc_training_metadata.json').read_text())
    if model_stamp(base) != metadata['base_model_stamp']:
        raise ValueError('Base files differ from the training base. Use the exact original local model.')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(adapter, local_files_only=True).save_pretrained(out)
    write_json(out / 'ewoc_training_metadata.json', metadata)
    write_json(out / 'labels.json', metadata['labels'])
    write_json(out / 'label_definitions.json', metadata['definitions'])
    print(f'Merged BF16 model: {out.resolve()}')

def serve_main(argv=None):
    p = argparse.ArgumentParser(description='Optional local FastAPI endpoint loading Qwen and the index once per process.', prog='ewoc.py serve')
    add_inference_args(p)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    args = parse_settings(p, 'serve', argv)
    if args.retrieval_device != 'cpu':
        raise ValueError('API retrieval uses CPU; reserve the GPU for vLLM. Use --retrieval-device cpu.')
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
    labels = load_labels(args.labels)
    resolve_retrieval_dependency(args)
    retriever = make_retriever(args, labels)
    classifier = QwenClassifier(args, labels)
    lock = Lock()
    app = FastAPI(title='EWOC Qwen Classifier', version='1.0')

    class Request(BaseModel):
        description: str = Field(min_length=1, max_length=50000)
        include_evidence: bool = False

    @app.get('/health')
    def health():
        return {'ready': True, 'rag': retriever is not None, 'adapter': bool(args.adapter), 'classes': len(labels)}

    @app.post('/classify')
    def classify(request: Request):
        if not request.description.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail='Description cannot be blank')
        with lock:
            examples = retriever.retrieve(request.description, count=args.examples, max_per_type=args.max_per_type, pool=args.pool, rerank_top=args.rerank_top) if retriever else []
            result = classifier.classify_batch([request.description], [examples])[0]
        if not request.include_evidence:
            result.pop('evidence', None)
        return result
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
def check_main(argv=None):
    parser = argparse.ArgumentParser(description="Run all CPU integrity checks")
    parser.parse_args(argv)
    import test_workflows
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_workflows)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "prepare": prepare_main, "index": index_main,
        "predict": predict_main, "evaluate": evaluate_main,
        "test": test_main,
        "train": train_main, "select": select_main,
        "merge": merge_main, "serve": serve_main,
        "check": check_main,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(commands))
    if not argv:
        if ACTION not in commands:
            raise ValueError(f'Unknown ACTION: {ACTION}')
        print(f'ACTION={ACTION}; settings from the CONFIG section')
        return commands[ACTION]([])
    if argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    return commands[args.command](argv[1:])


if __name__ == "__main__":
    sys.modules["ewoc"] = sys.modules[__name__]
    main()
