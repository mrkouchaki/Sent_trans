"""Completion-only token construction, independent of torch for CPU testing."""
from ewoc_common import fit_prompt


def encode_completion(tokenizer, row, examples, labels, definitions, max_length=2048):
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define an EOS token")
    target = tokenizer.encode(str(row["type_id"])+tokenizer.eos_token, add_special_tokens=False)
    if not target:
        raise ValueError("Empty completion")
    _, prompt_ids, stats = fit_prompt(tokenizer,row["description"],examples,labels,definitions,
                                     max_length-len(target))
    input_ids = prompt_ids+target
    return {"input_ids":input_ids,"attention_mask":[1]*len(input_ids),
            "labels":[-100]*len(prompt_ids)+target}, stats
