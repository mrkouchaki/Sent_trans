"""CPU regression tests. Run python -m unittest test_workflows -v."""
import argparse
from collections import Counter
from pathlib import Path
import json
import csv
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import ewoc
import evaluation
import data_preparation
import training
from ewoc import *

LABELS = {1: 'Call Test Validation', 2: 'E911 Provisioning', 40: 'BMC/WEA'}

def row(text, label=1, ticket='', group=''):
    return {'description': text, 'type_id': label, 'ewoc_id': ticket, 'group_id': group, 'row_key': digest([text, label, ticket])[:24], 'is_synthetic': False}

def corpus():
    return [row(f'{task} requested for uniquely named node {i:03d}', label, f'{label}-{i}') for label, task in [(1, 'Validate completed emergency calls'), (2, 'Provision emergency routing entries'), (40, 'BMC WEA broadcast configuration')] for i in range(80)]

class TinyTokenizer:
    eos_token = '\x03'

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def apply_chat_template(self, messages, **kwargs):
        return ''.join((m['role'] + ':' + m['content'] + '\n' for m in messages)) + 'assistant:<think></think>\n'

class Checks(unittest.TestCase):

    def test_plain_language_model_rejected_as_dense_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'config.json').write_text('{"architectures":["Qwen3ForCausalLM"]}')
            with self.assertRaisesRegex(ValueError, 'generation models'):
                local_embedding_model(d)

    def test_embedding_pipeline_can_share_causal_architecture(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'config.json').write_text('{"architectures":["Qwen3ForCausalLM"]}')
            (root / 'modules.json').write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
            (root / '1_Pooling').mkdir()
            (root / '1_Pooling' / 'config.json').write_text('{"pooling_mode_lasttoken":true}')
            self.assertEqual(local_embedding_model(d), str(root.resolve()))

    def test_grouped_partitions_are_disjoint_and_reproducible(self):
        a, _, _ = prepare(corpus(), folds=5)
        b, _, _ = prepare(corpus(), folds=5)
        self.assertEqual(a, b)
        for key in a:
            self.assertEqual({r['type_id'] for r in a[key]}, set(LABELS))
        assert_disjoint(a['train'], a['test'])
        assert_disjoint(a['train'], a['validation'])
        assert_disjoint(a['test'], a['validation'])

    def test_conflicting_input_is_quarantined(self):
        rows = corpus() + [row('same ambiguous input', 1), row('same ambiguous input', 2)]
        parts, conflicts, report = prepare(rows, folds=5)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(report['conflicting_rows_quarantined'], 2)
        self.assertFalse(any((r['description'] == 'same ambiguous input' for part in parts.values() for r in part)))

    def test_duplicate_normalization_and_id_bridges(self):
        rows = corpus() + [row('One common description', 1, 'BRIDGE-A'), row(' one  COMMON description ', 1, 'BRIDGE-B'), row('Another description joined by ticket ID', 1, 'BRIDGE-B')]
        parts, _, report = prepare(rows, folds=5)
        self.assertEqual(report['exact_duplicate_rows_removed'], 1)
        selected = [r for part in parts.values() for r in part if r['ewoc_id'].startswith('BRIDGE')]
        self.assertEqual(len({r['group_id'] for r in selected}), 1)

    def test_template_grouping_links_identifier_variations(self):
        rows = corpus() + [row('Unique special task for NODE12345678', 1, 'x'), row('Unique special task for NODE98765432', 1, 'y')]
        parts, _, _ = prepare(rows, folds=5, group_templates=True)
        selected = [r for part in parts.values() for r in part if r['ewoc_id'] in {'x', 'y'}]
        self.assertEqual(len({r['group_id'] for r in selected}), 1)

    def test_overlap_rejected(self):
        a = row('same input', ticket='A', group='G')
        with self.assertRaises(ValueError):
            assert_disjoint([a], [dict(a, row_key='other')])
        with self.assertRaises(ValueError):
            assert_adapter_holdout({'training_text_hashes': [digest(normalize(a['description']))]}, [a])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rows.csv'
            rows = corpus()[:5]
            write_rows(path, rows)
            self.assertEqual(read_rows(path, LABELS), rows)

    def test_lexical_retrieval_and_group_exclusion(self):
        rows = [row('Validate completed emergency calls', 1, 'A', 'G1'), row('Verify completed emergency test calls', 1, 'B', 'G2'), row('Provision PSAP routing entries', 2, 'C', 'G3'), row('BMC WEA broadcast configuration update', 40, 'D', 'G4')]
        with tempfile.TemporaryDirectory() as d:
            HybridRetriever.build(rows, LABELS, d)
            retriever = HybridRetriever(d)
            examples = retriever.retrieve('BMC WEA broadcast configuration', count=1, max_per_type=1)
            self.assertEqual(examples[0]['type_id'], 40)
            excluded = retriever.retrieve('Verify completed emergency calls', count=2, max_per_type=1, exclude_group='G1')
            self.assertTrue(all((r['group_id'] != 'G1' for r in excluded)))
            self.assertEqual(len({r['type_id'] for r in excluded}), len(excluded))

    def test_index_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            HybridRetriever.build(corpus(), LABELS, d)
            with self.assertRaises(FileExistsError):
                HybridRetriever.build(corpus(), LABELS, d)

    def test_prompt_only_loss_is_masked(self):
        tokenizer = TinyTokenizer()
        encoded, _ = encode_completion(tokenizer, row('Broadcast configuration', 40), [], LABELS, {}, 4096)
        supervised = [v for v in encoded['labels'] if v != -100]
        self.assertEqual(supervised, list(b'40\x03'))
        self.assertTrue(all((x == -100 for x in encoded['labels'][:-3])))
        self.assertEqual(len(encoded['input_ids']), len(encoded['labels']))

    def test_context_budget_removes_examples_before_ticket(self):
        tokenizer = TinyTokenizer()
        examples = [row('Example ' * 100, 1) for _ in range(12)]
        _, ids, stats = fit_prompt(tokenizer, 'Important action at the END', examples, LABELS, {}, 2200)
        self.assertLess(stats['examples_used'], 12)
        self.assertLessEqual(len(ids), 2200)
        self.assertIn(b'Important action', bytes(ids))

    def test_long_query_keeps_start_and_end(self):
        tokenizer = TinyTokenizer()
        text = 'START action ' + 'x ' * 5000 + ' END action'
        _, ids, _ = fit_prompt(tokenizer, text, [], LABELS, {}, 2000)
        self.assertLessEqual(len(ids), 2000)
        self.assertIn(b'START action', bytes(ids))
        self.assertIn(b'END action', bytes(ids))

    def test_invalid_predictions_count_as_errors(self):
        report = metrics([1, 2, 40], [1, -1, 2], LABELS)
        self.assertAlmostEqual(report['accuracy'], 1 / 3)
        self.assertEqual(report['invalid_predictions'], 1)
        self.assertEqual(sum(map(sum, report['confusion_matrix'])), 3)

class Tokenizer:

    def encode(self, text, add_special_tokens=False):
        return list(str(text).encode())

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode()

    def num_special_tokens_to_add(self, pair=False):
        return 2

class Encoder:

    def __init__(self, limit=256):
        self.max_seq_length = limit
        self.tokenizer = Tokenizer()
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        values = np.array([[1.0, text.count('call') + 1.0, text.count('routing') + 1.0] for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

def fixture(path, model_type='bert', pipeline=True):
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({'model_type': model_type, 'architectures': ['Qwen3ForCausalLM' if model_type == 'qwen3' else 'BertModel']}))
    if pipeline:
        (path / 'modules.json').write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
        (path / '1_Pooling').mkdir()
        (path / '1_Pooling' / 'config.json').write_text('{"pooling_mode_mean_tokens":true}')

class OptionalEmbeddings(unittest.TestCase):

    def test_minilm_has_plain_query_format(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'minilm'
            fixture(model)
            profile = inspect_embedding_profile(model, 'minilm')
            self.assertIsNone(profile['query_instruction'])
            self.assertEqual(format_embedding_text(Encoder(), 'call validation', profile, True), 'call validation')

    def test_raw_bert_explicitly_uses_mean_pooling(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'bert'
            fixture(model, pipeline=False)
            profile = inspect_embedding_profile(model)
            self.assertEqual(profile['pipeline'], 'bert-mean-pooling')
            self.assertIsNone(profile['query_instruction'])

    def test_qwen_generation_rejected_but_embedding_pipeline_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'qwen'
            fixture(model, 'qwen3', False)
            with self.assertRaises(ValueError):
                inspect_embedding_profile(model)
            (model / 'modules.json').write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
            (model / '1_Pooling').mkdir()
            (model / '1_Pooling' / 'config.json').write_text('{}')
            profile = inspect_embedding_profile(model)
            self.assertEqual(profile['kind'], 'qwen')
            self.assertTrue(format_embedding_text(Encoder(512), 'call validation', profile, True).startswith('Instruct: '))
            self.assertEqual(format_embedding_text(Encoder(), 'call validation', profile, False), 'call validation')

    def test_native_limit_is_never_increased(self):
        fake = types.SimpleNamespace(SentenceTransformer=lambda *a, **kw: Encoder(256), models=object())
        with patch.dict(sys.modules, {'sentence_transformers': fake}):
            encoder = load_embedding_encoder('unused', {'pipeline': 'saved', 'kind': 'minilm', 'requested_max_length': 1024})
            self.assertEqual(encoder.max_seq_length, 256)
            encoder = load_embedding_encoder('unused', {'pipeline': 'saved', 'kind': 'minilm', 'requested_max_length': 128})
            self.assertEqual(encoder.max_seq_length, 128)

    def test_long_minilm_input_keeps_both_ends(self):
        formatted = format_embedding_text(Encoder(128), 'START ' + 'x' * 5000 + ' END', {'query_instruction': None}, True)
        self.assertTrue(formatted.startswith('START'))
        self.assertTrue(formatted.endswith('END'))
        self.assertLessEqual(len(formatted.encode()) + 2, 128)

    def test_dense_index_persists_adapter_and_uses_same_query_format(self):
        with tempfile.TemporaryDirectory() as d:
            model, index = (Path(d) / 'minilm', Path(d) / 'index')
            fixture(model)
            encoder = Encoder(256)
            rows = [row('validate emergency call', 1, 'A', 'G1'), row('provision routing entries', 2, 'B', 'G2')]
            with patch('ewoc_core.load_embedding_encoder', return_value=encoder):
                HybridRetriever.build(rows, {1: 'Calls', 2: 'Provisioning'}, index, model, embedding_kind='minilm')
                retrieved = HybridRetriever(index)
                query = retrieved.encode_queries(['validate emergency call'])
                hits = retrieved.retrieve('validate emergency call', count=1, query_embedding=query[0])
                self.assertEqual(hits[0]['type_id'], 1)
                self.assertEqual(retrieved.data['embedding_profile']['effective_max_length'], 256)
                self.assertEqual(retrieved.embeddings.shape, (2, 3))
                self.assertFalse(any((text.startswith('Instruct:') for text in encoder.seen)))
                self.assertIn('no vector database', json.loads((index / 'index_manifest.json').read_text())['vector_backend'])


class ConsolidatedChecks(unittest.TestCase):
    def test_unified_names_and_definitions_config(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d)/'labels.json'
            write_json(config, {'1': {'name': 'Calls', 'definition': 'Validated call results'}})
            self.assertEqual(load_labels(config), {1: 'Calls'})
            self.assertEqual(load_definitions(config), {1: 'Validated call results'})
            write_json(config, {'1': 'Calls'})
            self.assertEqual(load_labels(config), {1: 'Calls'})

    def test_selection_invokes_unified_evaluate_command(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root/'run'
            run.mkdir()
            data = [row('Held-out call verification', 1, 'VAL', 'V')]
            write_rows(root/'validation.csv', data)
            write_json(root/'labels.json', {'1': {'name': 'Calls', 'definition': ''}})
            write_json(run/'run_metadata.json', {'validation_row_fingerprint': digest(sorted(r['row_key'] for r in data))})
            for step in [1, 2]:
                checkpoint = run/'checkpoints'/f'checkpoint-{step}'
                checkpoint.mkdir(parents=True)
                write_json(checkpoint/'adapter_config.json', {})
                write_json(checkpoint/'ewoc_training_metadata.json', {'checkpoint': step})
                (checkpoint/'adapter_model.safetensors').write_bytes(str(step).encode())
            def fake_evaluate(command, check=True):
                self.assertEqual(command[2], 'evaluate')
                self.assertEqual(Path(command[1]).name, 'ewoc.py')
                prefix = command[command.index('--out')+1]
                step = int(Path(command[command.index('--adapter')+1]).name.rsplit('-', 1)[1])
                write_json(prefix+'_metrics.json', {'macro_f1': step*.4, 'accuracy': .5, 'weighted_f1': .5})
            with patch('subprocess.run', side_effect=fake_evaluate):
                select_main(['--run', str(run), '--validation-csv', str(root/'validation.csv'),
                             '--labels', str(root/'labels.json'), '--no-rag', '--out', str(root/'selection')])
            self.assertEqual((run/'adapter_best_f1'/'adapter_model.safetensors').read_bytes(), b'2')

    def test_api_routes_survive_consolidation(self):
        routes = {}
        class App:
            def __init__(self, **kwargs): pass
            def get(self, path):
                return lambda func: routes.setdefault(path, func)
            def post(self, path):
                def register(func):
                    self_type = func.__annotations__['request']
                    if not isinstance(self_type, type):
                        raise AssertionError('Request annotation must resolve to the local request class')
                    routes[path] = func
                    return func
                return register
        fake_classifier = types.SimpleNamespace(classify_batch=lambda *a: [{'type_id': 1, 'evidence': []}])
        fakes = {'fastapi': types.SimpleNamespace(FastAPI=App),
                 'pydantic': types.SimpleNamespace(BaseModel=object, Field=lambda **kw: None),
                 'uvicorn': types.SimpleNamespace(run=lambda *a, **kw: None)}
        with patch.dict(sys.modules, fakes), patch('ewoc'+'.QwenClassifier', return_value=fake_classifier):
            serve_main(['--no-rag'])
        self.assertTrue(routes['/health']()['ready'])
        result = routes['/classify'](types.SimpleNamespace(description='Verify emergency calls', include_evidence=False))
        self.assertEqual(result, {'type_id': 1})


class EvaluationChecks(unittest.TestCase):
    def test_per_class_counts_overrides_and_reproducibility(self):
        selected, counts = select_evaluation_rows(corpus(), LABELS, 20, {2: 7}, 2026)
        again, _ = select_evaluation_rows(corpus(), LABELS, 20, {2: 7}, 2026)
        self.assertEqual(selected, again)
        self.assertEqual(Counter(r['type_id'] for r in selected), {1: 20, 2: 7, 40: 20})
        self.assertEqual(len({r['row_key'] for r in selected}), len(selected))
        self.assertFalse(any(r['shortage'] for r in counts))

    def test_shortages_are_recorded_or_rejected_without_duplicates(self):
        data = [row('One holdout', 1)]
        selected, counts = select_evaluation_rows(data, {1: 'Calls'}, 200)
        self.assertEqual(len(selected), 1)
        self.assertEqual(counts[0]['used'], 1)
        self.assertTrue(counts[0]['shortage'])
        with self.assertRaises(ValueError):
            select_evaluation_rows(data, {1: 'Calls'}, 200, strict=True)
        with self.assertRaises(ValueError):
            select_evaluation_rows(data, {1: 'Calls'}, 0)
        with self.assertRaises(ValueError):
            select_evaluation_rows(data, {1: 'Calls'}, 1, {99: 2})

    def test_precision_recall_and_exported_confusion_matrix(self):
        actual, predicted = [1, 1, 2, 2, 40, 40], [1, 2, 2, -1, 40, 1]
        report = metrics(actual, predicted, LABELS)
        self.assertAlmostEqual(report['accuracy'], .5)
        self.assertAlmostEqual(report['macro_precision'], 2/3)
        self.assertAlmostEqual(report['macro_recall'], .5)
        self.assertEqual(sum(map(sum, report['confusion_matrix'])), 6)
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'run'
            export_metric_tables(prefix, report, LABELS)
            with Path(str(prefix)+'_summary.csv').open() as f:
                summary = list(csv.DictReader(f))
            self.assertAlmostEqual(float(summary[0]['accuracy']), .5)
            with Path(str(prefix)+'_per_class_metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertTrue({'precision', 'recall', 'f1', 'support'}.issubset(rows[0]))
            with Path(str(prefix)+'_confusion_matrix.csv').open() as f:
                matrix = list(csv.reader(f))
            self.assertEqual(sum(sum(map(int, row[1:])) for row in matrix[1:]), 6)

    def test_separate_model_test_accepts_custom_csv_and_exports_all_metrics(self):
        import io
        from contextlib import redirect_stdout
        data = [r for r in corpus() if r['ewoc_id'].endswith(('-0', '-1', '-2'))]
        actual_by_text = {r['description']: r['type_id'] for r in data}
        def predict(descriptions, evidence):
            output = []
            for description in descriptions:
                actual = actual_by_text[description]
                predicted = 2 if actual == 1 else actual
                output.append({'type_id': predicted, 'type_name': LABELS[predicted], 'raw': str(predicted),
                               'finish_reason': 'stop', 'prompt_stats': {'examples_used': 0}, 'evidence': []})
            return output
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_rows(root/'custom_test.csv', data)
            write_json(root/'labels.json', {str(k): {'name': v, 'definition': ''} for k,v in LABELS.items()})
            prefix = root/'results'/'test'
            fake = types.SimpleNamespace(classify_batch=predict)
            with patch('evaluation.QwenClassifier', return_value=fake), \
                 patch('ewoc'+'.train_main', side_effect=AssertionError('Test must not train')), \
                 redirect_stdout(io.StringIO()) as output:
                test_main(['--eval-csv', str(root/'custom_test.csv'), '--labels', str(root/'labels.json'),
                           '--samples-per-class', '2', '--no-rag', '--out', str(prefix)])
            report = json.loads(Path(str(prefix)+'_metrics.json').read_text())
            self.assertEqual(report['rows'], 6)
            self.assertAlmostEqual(report['accuracy'], 2/3)
            self.assertEqual([r['used'] for r in report['sample_counts']], [2, 2, 2])
            for suffix in ['metrics.json', 'predictions.csv', 'errors.csv', 'evidence.jsonl',
                           'summary.csv', 'per_class_metrics.csv', 'confusion_matrix.csv', 'sample_counts.csv']:
                self.assertTrue(Path(str(prefix)+'_'+suffix).exists(), suffix)
            self.assertIn('precision', output.getvalue())
            self.assertIn('Confusion matrix', output.getvalue())

    def test_draft_definitions_are_marked_and_included_in_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d)/'labels.json'
            write_json(config, {'1': {'name': 'Calls', 'definition': 'Review call results',
                                     'definition_status': 'draft_needs_review'},
                                '2': {'name': 'Provisioning', 'definition': 'Approved fixture definition',
                                      'definition_status': 'approved'}})
            labels = load_labels(config)
            definitions = load_definitions(config, labels)
        self.assertTrue(definitions[1].startswith('[DRAFT: needs domain review]'))
        self.assertEqual(definitions[2], 'Approved fixture definition')
        messages = make_messages('Verify completed test calls', [], labels, definitions)
        for definition in definitions.values():
            self.assertIn(definition, messages[0]['content'])
        self.assertIn('provisional guidance', messages[0]['content'])

    def test_standalone_settings_control_sample_counts(self):
        import test_evaluate as separate
        with patch.object(separate, 'N_ROWS_PER_TYPE', 25), \
             patch.object(separate, 'ROWS_PER_TYPE', {1: 10}), \
             patch.object(separate, 'USE_RAG', False):
            args = separate.settings_arguments()
        self.assertEqual(args[args.index('--samples-per-class')+1], '25')
        self.assertEqual(json.loads(args[args.index('--samples-by-class')+1]), {'1': 10})
        self.assertIn('--no-rag', args)


class ConfigurationChecks(unittest.TestCase):
    def test_action_runs_without_flags_and_does_not_train_by_default(self):
        with patch('ewoc'+'.ACTION', 'test'), patch('ewoc'+'.test_main') as test, \
             patch('ewoc'+'.train_main', side_effect=AssertionError('Unexpected training')):
            main([])
        test.assert_called_once_with([])

    def test_config_changes_reach_parser_and_optional_flags_override(self):
        parser = argparse.ArgumentParser()
        add_inference_args(parser)
        with patch.dict(INFERENCE, {'backend': 'vllm', 'batch_size': 7}), \
             patch('ewoc'+'.USE_RAG', False):
            configured = parse_settings(parser, 'predict', [])
            overridden = parse_settings(parser, 'predict', ['--backend', 'transformers', '--rag'])
        self.assertEqual((configured.backend, configured.batch_size, configured.no_rag), ('vllm', 7, True))
        self.assertEqual((overridden.backend, overridden.no_rag), ('transformers', False))
        with patch.dict(INFERENCE, {'backend': 'typo'}), self.assertRaises(ValueError):
            parse_settings(parser, 'predict', [])

    def test_unlimited_sampling_and_empty_overrides_reach_standalone_runner(self):
        import test_evaluate as separate
        with patch.object(separate, 'N_ROWS_PER_TYPE', None), \
             patch.object(separate, 'ROWS_PER_TYPE', {}), \
             patch.object(separate, 'ADAPTER_PATH', None), \
             patch.object(separate, 'evaluate_main') as evaluate:
            separate.main([])
        arguments = evaluate.call_args.args[0]
        self.assertIn('--all-samples', arguments)
        self.assertEqual(arguments[arguments.index('--samples-by-class')+1], '{}')
        self.assertIsNone(evaluate.call_args.kwargs['settings']['adapter'])

    def test_constrained_labels_cover_multitoken_and_prefix_ids(self):
        sequences = [[10], [10, 20], [30, 40]]
        self.assertEqual(allowed_label_tokens(sequences, [], 99), [10, 30])
        self.assertEqual(allowed_label_tokens(sequences, [10], 99), [20, 99])
        self.assertEqual(allowed_label_tokens(sequences, [30], 99), [40])
        self.assertEqual(allowed_label_tokens(sequences, [10, 20], 99), [99])
        with self.assertRaises(ValueError):
            allowed_label_tokens(sequences, [88], 99)

    def test_transformers_backend_loads_local_files_and_generates_allowed_label(self):
        from contextlib import nullcontext
        class LocalTokenizer(TinyTokenizer):
            eos_token_id = 3
            pad_token_id = 3
            def decode(self, ids, skip_special_tokens=True):
                return bytes(int(v) for v in ids if int(v) != 3).decode()
        tokenizer = LocalTokenizer()
        def generate(**kwargs):
            values = kwargs['input_ids'][0].tolist()
            fn = kwargs['prefix_allowed_tokens_fn']
            self.assertIn(ord('4'), fn(0, np.asarray(values)))
            values.append(ord('4'))
            self.assertEqual(fn(0, np.asarray(values)), [ord('0')])
            values.append(ord('0'))
            self.assertEqual(fn(0, np.asarray(values)), [3])
            return np.asarray([values + [3]])
        model = types.SimpleNamespace(to=lambda device: None, eval=lambda: None, generate=generate)
        tokenizer_loader = unittest.mock.Mock(return_value=tokenizer)
        model_loader = unittest.mock.Mock(return_value=model)
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
            bfloat16='bfloat16', long='long', ones_like=np.ones_like,
            tensor=lambda values, **kw: np.asarray(values), inference_mode=nullcontext)
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_loader),
            GenerationConfig=lambda **kw: types.SimpleNamespace(**kw))
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
            args = argparse.Namespace(**dict(configured_defaults('predict'), model=d, adapter=None))
            classifier = QwenClassifier(args, labels=LABELS, definitions={})
            result = classifier.classify_batch(['BMC/WEA broadcast request'], [[]])[0]
        self.assertEqual(result['type_id'], 40)
        self.assertEqual(result['finish_reason'], 'constrained_label')
        for loader in [tokenizer_loader, model_loader]:
            self.assertTrue(loader.call_args.kwargs['local_files_only'])
            self.assertFalse(loader.call_args.kwargs['trust_remote_code'])
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['HF_HUB_DISABLE_TELEMETRY'], '1')

    def test_config_only_prepare_index_and_rag_evaluation(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            write_rows(root/'source.csv', corpus())
            write_json(root/'labels.json', {str(k): {'name': v, 'definition': ''} for k,v in LABELS.items()})
            with patch('ewoc'+'.LABEL_CONFIG', root/'labels.json'), \
                 patch.dict(PREPARE, {'csv': root/'source.csv', 'out': root/'data', 'folds': 5}), \
                 patch('ewoc'+'.TRAIN_CSV', None), \
                 patch('ewoc'+'.INDEX_PATH', root/'index'), \
                 patch('ewoc'+'.USE_RAG', True), \
                 patch.dict(EMBEDDING, {'embedding_model': None}), \
                 patch.dict(TEST, {'eval_csv': None, 'out': root/'results/test',
                                   'max_per_class': 2, 'samples_by_class': {40: 1}}):
                # The model-test action itself prepares splits and builds its index.
                fake = types.SimpleNamespace(classify_batch=lambda descriptions, evidence: [
                    {'type_id': 1, 'type_name': LABELS[1], 'raw': '1', 'finish_reason': 'stop',
                     'prompt_stats': {'examples_used': len(ex)}, 'evidence': ex} for ex in evidence])
                with patch('evaluation.QwenClassifier', return_value=fake):
                    test_main([])
            report = json.loads((root/'results/test_metrics.json').read_text())
            self.assertEqual(report['rows'], 5)
            self.assertIsNotNone(report['index_fingerprint'])
            self.assertEqual([r['used'] for r in report['sample_counts']], [2, 2, 1])
            self.assertTrue((root/'results/test_confusion_matrix.csv').exists())


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_rows(self.root/'source.csv', corpus())
        write_json(self.root/'labels.json', {str(k): {'name': v, 'definition': ''}
                                           for k,v in LABELS.items()})
        self.preparer = data_preparation.DataPreparer(
            self.root/'source.csv', self.root/'data', self.root/'labels.json', folds=5)

    def train_args(self, **overrides):
        values = dict(configured_defaults('train'), model=str(self.root/'model'),
                      labels=str(self.root/'labels.json'), train_csv=None, validation_csv=None,
                      output=str(self.root/'run'), resume=None, rag_index=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_first_train_prepares_before_model_work_and_next_run_reuses(self):
        seen = []
        def fit(args):
            seen.append((args.train_csv, args.method))
            train = read_rows(args.train_csv, LABELS)
            validation = read_rows(args.validation_csv, LABELS)
            assert_disjoint(train, validation)
            self.assertTrue(args.prepared_data_signature)
            return 'trained'
        with patch.dict(ewoc.PREPARE, {'csv': self.root/'source.csv', 'out': self.root/'data', 'folds': 5}), \
             patch.object(ewoc, 'LABEL_CONFIG', self.root/'labels.json'), \
             patch.object(ewoc, 'TRAIN_CSV', None), patch.object(ewoc, 'VALIDATION_CSV', None), \
             patch.dict(ewoc.FINETUNE, {'output': self.root/'run', 'method': 'lora', 'resume': None, 'rag_index': None}), \
             patch('training.run_training', side_effect=fit):
            self.assertEqual(ewoc.train_main([]), 'trained')
            first = Path(seen[0][0])
            timestamp = first.stat().st_mtime_ns
            with patch('data_preparation.prepare', side_effect=AssertionError('Unexpected re-split')), \
                 patch.dict(ewoc.FINETUNE, {'output': self.root/'run2', 'method': 'qlora'}):
                self.assertEqual(ewoc.train_main([]), 'trained')
            self.assertEqual(first.stat().st_mtime_ns, timestamp)
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertEqual([item[1] for item in seen], ['lora', 'qlora'])

    def test_changed_source_creates_new_splits_without_overwriting_previous(self):
        first = self.preparer.ensure()
        old = first.train_csv.read_bytes()
        write_rows(self.root/'source.csv', corpus()+[row('Additional genuine call result', 1, 'NEW', 'NEW')])
        second = self.preparer.ensure()
        self.assertNotEqual(first.signature, second.signature)
        self.assertEqual(first.train_csv.read_bytes(), old)
        self.assertFalse(second.reused)

    def test_split_seed_change_creates_a_new_cache_version(self):
        first = self.preparer.ensure()
        other = data_preparation.DataPreparer(self.root/'source.csv', self.root/'data',
                                              self.root/'labels.json', folds=5, seed=7)
        second = other.ensure()
        self.assertNotEqual(first.signature, second.signature)
        self.assertTrue(first.train_csv.is_file())

    def test_corrupt_cache_is_quarantined_and_rebuilt(self):
        first = self.preparer.ensure()
        original = first.train_csv.read_bytes()
        first.train_csv.write_text('broken cache')
        second = self.preparer.ensure()
        self.assertEqual(first.signature, second.signature)
        self.assertFalse(second.reused)
        self.assertEqual(second.train_csv.read_bytes(), original)
        self.assertEqual(len(list((self.root/'data/prepared').glob('invalid_*'))), 1)

    def test_failed_preparation_publishes_no_partial_dataset(self):
        with patch('data_preparation.prepare', side_effect=ValueError('invalid fixture')), self.assertRaises(ValueError):
            self.preparer.ensure()
        self.assertEqual(list((self.root/'data/prepared').iterdir()), [])

    def test_resume_rejects_changed_source_before_building_or_training(self):
        original = self.preparer.ensure()
        run = self.root/'run'
        write_json(run/'run_metadata.json', {'prepared_data_signature': original.signature,
                    'data_file_hashes': {'train': data_preparation.file_hash(original.train_csv),
                                        'validation': data_preparation.file_hash(original.validation_csv)}})
        write_rows(self.root/'source.csv', corpus()+[row('Different real source', 2, 'NEW', 'NEW')])
        pipeline = training.TrainingPipeline(self.train_args(resume='latest'), self.preparer, embedding_settings())
        with patch('training.run_training') as fit, self.assertRaisesRegex(ValueError, 'Resume stopped'):
            pipeline.run()
        fit.assert_not_called()
        self.assertEqual(len(list((self.root/'data/prepared').iterdir())), 1)

    def test_custom_grouped_splits_are_validated_without_raw_preparation(self):
        dataset = self.preparer.ensure()
        args = self.train_args(train_csv=str(dataset.train_csv), validation_csv=str(dataset.validation_csv))
        with patch.object(self.preparer, 'ensure', side_effect=AssertionError('Must use supplied splits')), \
             patch('training.run_training', return_value='custom'):
            self.assertEqual(training.TrainingPipeline(args, self.preparer, embedding_settings()).run(), 'custom')
        bad = self.train_args(train_csv=str(dataset.train_csv), validation_csv=str(dataset.train_csv))
        with patch('training.run_training') as fit, self.assertRaisesRegex(ValueError, 'Leakage'):
            training.TrainingPipeline(bad, self.preparer, embedding_settings()).run()
        fit.assert_not_called()

    def test_rag_training_automatically_builds_and_reuses_index(self):
        def fit(args):
            retriever = HybridRetriever(args.rag_index)
            self.assertEqual(len(retriever.rows), len(read_rows(args.train_csv, LABELS)))
            assert_disjoint(retriever.rows, read_rows(args.validation_csv, LABELS))
            return args.rag_index
        settings = dict(embedding_settings(), embedding_model=None)
        with patch('training.run_training', side_effect=fit):
            first = training.TrainingPipeline(self.train_args(rag_index=str(self.root/'index')),
                                               self.preparer, settings).run()
            timestamp = (Path(first)/'lexical.joblib').stat().st_mtime_ns
            with patch.object(HybridRetriever, 'build', side_effect=AssertionError('Unexpected index rebuild')):
                second = training.TrainingPipeline(self.train_args(rag_index=str(self.root/'index')),
                                                    self.preparer, settings).run()
        self.assertEqual(first, second)
        self.assertEqual((Path(second)/'lexical.joblib').stat().st_mtime_ns, timestamp)

    def test_tampered_index_is_rebuilt_from_training_only(self):
        dataset = self.preparer.ensure()
        rows = read_rows(dataset.train_csv, LABELS)
        first = HybridRetriever.ensure(rows, LABELS, self.root/'index')
        (first/'lexical.joblib').write_bytes(b'partial write')
        second = HybridRetriever.ensure(rows, LABELS, self.root/'index')
        self.assertEqual(len(HybridRetriever(second).rows), len(rows))
        self.assertEqual(first, second)

    def test_training_body_preserves_methods_epoch_metadata_exports_and_resume(self):
        import io
        from contextlib import redirect_stdout
        base = self.root/'model'
        write_json(base/'config.json', {'model_type': 'qwen3'})
        captured = []
        class TrainingTokenizer(TinyTokenizer):
            pad_token_id = 0
            def save_pretrained(self, directory):
                write_json(Path(directory)/'tokenizer_config.json', {})
        model = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True),
                                      print_trainable_parameters=lambda: None)
        model_loader = unittest.mock.Mock(return_value=model)
        prepare_quantized = unittest.mock.Mock(side_effect=lambda model, **kw: model)
        lora_configs = []
        def attach_lora(model, config):
            lora_configs.append(config)
            return model
        class FakeTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.state = types.SimpleNamespace(global_step=1, epoch=1.0, best_metric=.3,
                    best_model_checkpoint=str(Path(kwargs['args'].output_dir)/'checkpoint-1'),
                    log_history=[{'epoch': 1.0, 'eval_loss': .3}])
            def train(self, resume_from_checkpoint=None):
                captured.append((self.kwargs, resume_from_checkpoint))
                checkpoint = Path(self.state.best_model_checkpoint)
                checkpoint.mkdir(parents=True, exist_ok=True)
                write_json(checkpoint/'adapter_config.json', {})
                for callback in self.kwargs['callbacks']:
                    if hasattr(callback, 'on_save'):
                        callback.on_save(self.kwargs['args'], self.state, None)
                batch = self.kwargs['data_collator']([self.kwargs['train_dataset'][0]])
                if not (batch['labels'] == -100).any():
                    raise AssertionError('Prompt loss must be masked')
                return types.SimpleNamespace(metrics={'train_loss': .2})
            def save_metrics(self, name, metrics): pass
            def evaluate(self): return {'eval_loss': .3}
            def save_model(self, directory):
                write_json(Path(directory)/'adapter_config.json', {})
        torch = types.SimpleNamespace(
            bfloat16='bf16', long=np.int64, tensor=lambda values, **kw: np.asarray(values),
            cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))))
        transformers = types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_loader),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: TrainingTokenizer()),
            BitsAndBytesConfig=lambda **kw: kw, EarlyStoppingCallback=lambda **kw: object(),
            Trainer=FakeTrainer, TrainerCallback=object, TrainingArguments=lambda **kw: types.SimpleNamespace(**kw),
            set_seed=lambda value: None)
        modules = {'torch': torch, 'torch.utils': types.SimpleNamespace(),
                   'torch.utils.data': types.SimpleNamespace(Dataset=object),
                   'peft': types.SimpleNamespace(LoraConfig=lambda **kw: kw, get_peft_model=attach_lora,
                                                prepare_model_for_kbit_training=prepare_quantized),
                   'transformers': transformers,
                   'transformers.trainer_utils': types.SimpleNamespace(
                       get_last_checkpoint=lambda directory: str(Path(directory)/'checkpoint-1'))}
        with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            for method in ['lora', 'qlora']:
                output = self.root/f'run_{method}'
                args = self.train_args(method=method, quantization=None, output=str(output))
                training.TrainingPipeline(args, self.preparer, embedding_settings()).run()
                metadata = json.loads((output/'adapter/ewoc_training_metadata.json').read_text())
                self.assertTrue(metadata['training_finished'])
                self.assertEqual(metadata['quantization'], 'nf4' if method == 'qlora' else 'none')
                self.assertEqual(metadata['prepared_data_signature'], args.prepared_data_signature)
                self.assertTrue((output/'checkpoints/checkpoint-1/ewoc_training_metadata.json').is_file())
                self.assertTrue((output/'training_history.json').is_file())
                resumed = self.train_args(method=method, quantization=None, output=str(output), resume='latest')
                training.TrainingPipeline(resumed, self.preparer, embedding_settings()).run()
                self.assertEqual(captured[-1][1], str(output/'checkpoints/checkpoint-1'))
        self.assertEqual(prepare_quantized.call_count, 2)  # QLoRA initial run and resume only.
        for kwargs, _ in captured:
            self.assertEqual(kwargs['args'].save_strategy, 'epoch')
            self.assertIsNone(kwargs['args'].save_total_limit)
            self.assertFalse(kwargs['args'].push_to_hub)
        self.assertTrue(all(c['target_modules'] == 'all-linear' for c in lora_configs))
        self.assertTrue(all(call.kwargs['local_files_only'] for call in model_loader.call_args_list))

