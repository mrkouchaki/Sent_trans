"""CPU checks for split leakage, retrieval, metrics, and completion-only labels.

Run: python selfcheck.py. No model files, GPU, torch or vLLM required.
These checks do not benchmark GPU inference or fine-tuning.
"""
import tempfile
import unittest
from pathlib import Path

from ewoc_common import (assert_disjoint,digest,fit_prompt,load_labels,normalize,
                         read_rows,write_rows)
from ewoc_sft_data import encode_completion
from prepare_data import prepare
from ewoc_retrieval import HybridRetriever
from ewoc_classifier import assert_adapter_holdout
from evaluate_classifier import metrics


LABELS = {1:"Call Test Validation",2:"E911 Provisioning",40:"BMC/WEA"}


def row(text,label=1,ticket="",group=""):
    return {"description":text,"type_id":label,"ewoc_id":ticket,
            "group_id":group,"row_key":digest([text,label,ticket])[:24],"is_synthetic":False}


def corpus():
    return [row(f"{task} requested for uniquely named node {i:03d}",label,f"{label}-{i}")
            for label,task in [(1,"Validate completed emergency calls"),
                               (2,"Provision emergency routing entries"),
                               (40,"BMC WEA broadcast configuration")]
            for i in range(80)]


class TinyTokenizer:
    eos_token = "\x03"
    def encode(self,text,add_special_tokens=False):
        return list(text.encode("utf-8"))
    def apply_chat_template(self,messages,**kwargs):
        return "".join(m["role"]+":"+m["content"]+"\n" for m in messages)+"assistant:<think></think>\n"


class Checks(unittest.TestCase):
    def test_grouped_partitions_are_disjoint_and_reproducible(self):
        a,_,_ = prepare(corpus(),folds=5)
        b,_,_ = prepare(corpus(),folds=5)
        self.assertEqual(a,b)
        for key in a:
            self.assertEqual({r["type_id"] for r in a[key]},set(LABELS))
        assert_disjoint(a["train"],a["test"])
        assert_disjoint(a["train"],a["validation"])
        assert_disjoint(a["test"],a["validation"])

    def test_conflicting_input_is_quarantined(self):
        rows = corpus()+[row("same ambiguous input",1),row("same ambiguous input",2)]
        parts,conflicts,report = prepare(rows,folds=5)
        self.assertEqual(len(conflicts),2)
        self.assertEqual(report["conflicting_rows_quarantined"],2)
        self.assertFalse(any(r["description"]=="same ambiguous input" for part in parts.values() for r in part))

    def test_duplicate_normalization_and_id_bridges(self):
        rows = corpus()+[row("One common description",1,"BRIDGE-A"),
                         row(" one  COMMON description ",1,"BRIDGE-B"),
                         row("Another description joined by ticket ID",1,"BRIDGE-B")]
        parts,_,report = prepare(rows,folds=5)
        self.assertEqual(report["exact_duplicate_rows_removed"],1)
        selected = [r for part in parts.values() for r in part if r["ewoc_id"].startswith("BRIDGE")]
        self.assertEqual(len({r["group_id"] for r in selected}),1)

    def test_template_grouping_links_identifier_variations(self):
        rows = corpus()+[row("Unique special task for NODE12345678",1,"x"),
                         row("Unique special task for NODE98765432",1,"y")]
        # Base corpus's short numeric suffixes remain independent under the conservative heuristic.
        parts,_,_ = prepare(rows,folds=5,group_templates=True)
        selected = [r for part in parts.values() for r in part if r["ewoc_id"] in {"x","y"}]
        self.assertEqual(len({r["group_id"] for r in selected}),1)

    def test_overlap_rejected(self):
        a = row("same input",ticket="A",group="G")
        with self.assertRaises(ValueError):
            assert_disjoint([a],[dict(a,row_key="other")])
        with self.assertRaises(ValueError):
            assert_adapter_holdout({"training_text_hashes":[digest(normalize(a["description"]))]},[a])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"rows.csv"
            rows = corpus()[:5]
            write_rows(path,rows)
            self.assertEqual(read_rows(path,LABELS),rows)

    def test_lexical_retrieval_and_group_exclusion(self):
        rows = [row("Validate completed emergency calls",1,"A","G1"),
                row("Verify completed emergency test calls",1,"B","G2"),
                row("Provision PSAP routing entries",2,"C","G3"),
                row("BMC WEA broadcast configuration update",40,"D","G4")]
        with tempfile.TemporaryDirectory() as d:
            HybridRetriever.build(rows,LABELS,d)
            retriever = HybridRetriever(d)
            examples = retriever.retrieve("BMC WEA broadcast configuration",count=1,max_per_type=1)
            self.assertEqual(examples[0]["type_id"],40)
            excluded = retriever.retrieve("Verify completed emergency calls",count=2,max_per_type=1,
                                           exclude_group="G1")
            self.assertTrue(all(r["group_id"]!="G1" for r in excluded))
            self.assertEqual(len({r["type_id"] for r in excluded}),len(excluded))

    def test_index_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            HybridRetriever.build(corpus(),LABELS,d)
            with self.assertRaises(FileExistsError):
                HybridRetriever.build(corpus(),LABELS,d)

    def test_prompt_only_loss_is_masked(self):
        tokenizer = TinyTokenizer()
        encoded,_ = encode_completion(tokenizer,row("Broadcast configuration",40),[],LABELS,{},4096)
        supervised = [v for v in encoded["labels"] if v != -100]
        self.assertEqual(supervised,list(b"40\x03"))
        self.assertTrue(all(x==-100 for x in encoded["labels"][:-3]))
        self.assertEqual(len(encoded["input_ids"]),len(encoded["labels"]))

    def test_context_budget_removes_examples_before_ticket(self):
        tokenizer = TinyTokenizer()
        examples = [row("Example "*100,1) for _ in range(12)]
        _,ids,stats = fit_prompt(tokenizer,"Important action at the END",examples,LABELS,{},2200)
        self.assertLess(stats["examples_used"],12)
        self.assertLessEqual(len(ids),2200)
        self.assertIn(b"Important action",bytes(ids))

    def test_long_query_keeps_start_and_end(self):
        tokenizer = TinyTokenizer()
        text = "START action "+"x "*5000+" END action"
        _,ids,_ = fit_prompt(tokenizer,text,[],LABELS,{},2000)
        self.assertLessEqual(len(ids),2000)
        self.assertIn(b"START action",bytes(ids))
        self.assertIn(b"END action",bytes(ids))

    def test_invalid_predictions_count_as_errors(self):
        report = metrics([1,2,40],[1,-1,2],LABELS)
        self.assertAlmostEqual(report["accuracy"],1/3)
        self.assertEqual(report["invalid_predictions"],1)
        self.assertEqual(sum(map(sum,report["confusion_matrix"])),3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
