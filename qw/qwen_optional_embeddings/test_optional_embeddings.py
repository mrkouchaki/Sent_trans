"""CPU adapter and persistence tests. Neural encoders are mocked, not benchmarked."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ewoc_embeddings import inspect_embedding_profile, load_embedding_encoder, format_embedding_text
from ewoc_retrieval import HybridRetriever
from selfcheck import row


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
        values = np.array([[1., text.count('call')+1., text.count('routing')+1.] for text in texts],dtype=np.float32)
        return values/np.linalg.norm(values,axis=1,keepdims=True)


def fixture(path, model_type="bert", pipeline=True):
    path.mkdir(parents=True)
    (path/"config.json").write_text(json.dumps({"model_type":model_type,"architectures":["Qwen3ForCausalLM" if model_type=="qwen3" else "BertModel"]}))
    if pipeline:
        (path/"modules.json").write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
        (path/"1_Pooling").mkdir()
        (path/"1_Pooling"/"config.json").write_text('{"pooling_mode_mean_tokens":true}')


class OptionalEmbeddings(unittest.TestCase):
    def test_minilm_has_plain_query_format(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)/"minilm"
            fixture(model)
            profile = inspect_embedding_profile(model,"minilm")
            self.assertIsNone(profile["query_instruction"])
            self.assertEqual(format_embedding_text(Encoder(),"call validation",profile,True),"call validation")

    def test_raw_bert_explicitly_uses_mean_pooling(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)/"bert"
            fixture(model,pipeline=False)
            profile = inspect_embedding_profile(model)
            self.assertEqual(profile["pipeline"],"bert-mean-pooling")
            self.assertIsNone(profile["query_instruction"])

    def test_qwen_generation_rejected_but_embedding_pipeline_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)/"qwen"
            fixture(model,"qwen3",False)
            with self.assertRaises(ValueError):
                inspect_embedding_profile(model)
            (model/"modules.json").write_text('[{"type":"sentence_transformers.models.Pooling","path":"1_Pooling"}]')
            (model/"1_Pooling").mkdir()
            (model/"1_Pooling"/"config.json").write_text('{}')
            profile = inspect_embedding_profile(model)
            self.assertEqual(profile["kind"],"qwen")
            self.assertTrue(format_embedding_text(Encoder(512),"call validation",profile,True).startswith("Instruct: "))
            self.assertEqual(format_embedding_text(Encoder(),"call validation",profile,False),"call validation")

    def test_native_limit_is_never_increased(self):
        fake = types.SimpleNamespace(SentenceTransformer=lambda *a,**kw:Encoder(256), models=object())
        with patch.dict(sys.modules,{"sentence_transformers":fake}):
            encoder = load_embedding_encoder("unused",{"pipeline":"saved","kind":"minilm","requested_max_length":1024})
            self.assertEqual(encoder.max_seq_length,256)
            encoder = load_embedding_encoder("unused",{"pipeline":"saved","kind":"minilm","requested_max_length":128})
            self.assertEqual(encoder.max_seq_length,128)

    def test_long_minilm_input_keeps_both_ends(self):
        formatted = format_embedding_text(Encoder(128),"START "+"x"*5000+" END",{"query_instruction":None},True)
        self.assertTrue(formatted.startswith("START"))
        self.assertTrue(formatted.endswith("END"))
        self.assertLessEqual(len(formatted.encode())+2,128)

    def test_dense_index_persists_adapter_and_uses_same_query_format(self):
        with tempfile.TemporaryDirectory() as d:
            model,index = Path(d)/"minilm",Path(d)/"index"
            fixture(model)
            encoder = Encoder(256)
            rows = [row("validate emergency call",1,"A","G1"),row("provision routing entries",2,"B","G2")]
            with patch("ewoc_retrieval.load_embedding_encoder",return_value=encoder):
                HybridRetriever.build(rows,{1:"Calls",2:"Provisioning"},index,model,embedding_kind="minilm")
                retrieved = HybridRetriever(index)
                query = retrieved.encode_queries(["validate emergency call"])
                hits = retrieved.retrieve("validate emergency call",count=1,query_embedding=query[0])
                self.assertEqual(hits[0]["type_id"],1)
                self.assertEqual(retrieved.data["embedding_profile"]["effective_max_length"],256)
                self.assertEqual(retrieved.embeddings.shape,(2,3))
                self.assertFalse(any(text.startswith("Instruct:") for text in encoder.seen))
                self.assertIn("no vector database",json.loads((index/"index_manifest.json").read_text())["vector_backend"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
