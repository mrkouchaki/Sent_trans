"""Independent model evaluation: edit the settings below and run this file.

No fine-tuning is performed. Use a labeled holdout CSV, including a new custom
test set, with DESCRIPTION and TYPE_ID. CLI flags override these settings.
"""
from pathlib import Path
import json
import os
import sys
from ewoc import configured_defaults, evaluate_main, GPU_ID

ROOT = Path(__file__).resolve().parent
DEFAULTS = configured_defaults('test')

# ----------------------- EDIT THESE SETTINGS -----------------------
# All settings inherit ewoc.py's CONFIG. Optional test-only overrides here.
MODEL_PATH = DEFAULTS['model']
ADAPTER_PATH = DEFAULTS['adapter']    # None=base; or local trained adapter folder
TEST_CSV = DEFAULTS['eval_csv']       # Or your independent labeled CSV
OUTPUT_PREFIX = DEFAULTS['out']
LABEL_CONFIG = DEFAULTS['labels']

N_ROWS_PER_TYPE = DEFAULTS['max_per_class'] # None=all; default 200
ROWS_PER_TYPE = DEFAULTS['samples_by_class'] # Optional: {1: 100, 40: 80}
STRICT_SAMPLE_COUNTS = DEFAULTS['strict_samples'] # True=require requested counts
RANDOM_SEED = DEFAULTS['seed']

USE_RAG = not DEFAULTS['no_rag']      # False=direct classification
INDEX_PATH = DEFAULTS['index']        # Inherits selected lexical/dense index
FEW_SHOT_EXAMPLES = DEFAULTS['examples']
MAX_EXAMPLES_PER_TYPE = DEFAULTS['max_per_type']
RERANKER_PATH = DEFAULTS['reranker']   # Optional local BGE folder
# Backend, dtype, device and every training/embedding setting live in ewoc.py.
# ------------------------------------------------------------------


def settings_arguments():
    args = ['--model', str(MODEL_PATH), '--eval-csv', str(TEST_CSV),
            '--out', str(OUTPUT_PREFIX), '--labels', str(LABEL_CONFIG),
            '--seed', str(RANDOM_SEED), '--examples', str(FEW_SHOT_EXAMPLES),
            '--max-per-type', str(MAX_EXAMPLES_PER_TYPE)]
    if N_ROWS_PER_TYPE is not None:
        args += ['--samples-per-class', str(N_ROWS_PER_TYPE)]
    else:
        args.append('--all-samples')
    args += ['--samples-by-class', json.dumps(ROWS_PER_TYPE)]
    args.append('--strict-samples' if STRICT_SAMPLE_COUNTS else '--no-strict-samples')
    if USE_RAG:
        args += ['--rag', '--index', str(INDEX_PATH)]
    else:
        args.append('--no-rag')
    if ADAPTER_PATH:
        args += ['--adapter', str(ADAPTER_PATH)]
    if RERANKER_PATH:
        args += ['--reranker', str(RERANKER_PATH)]
    return args


def main(argv=None):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
    # Repeated value options use the final CLI value, so --eval-csv, --out,
    # --samples-per-class, --model and --adapter can be overridden on demand.
    return evaluate_main(settings_arguments() + list(sys.argv[1:] if argv is None else argv),
                         _command='test', settings={'adapter': ADAPTER_PATH, 'reranker': RERANKER_PATH})


if __name__ == '__main__':
    main()
