import numpy as np
import torch
from .prompts import LETTERS, build_messages


def _prep_tokenizer(tok):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

@torch.no_grad()
def mcq_predict(model, tok, examples, batch_size=8, retrieved=None):
    tok = _prep_tokenizer(tok)
    letter_ids = {}
    for L in LETTERS:
        ids = tok.encode(" " + L, add_special_tokens=False)
        assert len(ids) == 1, f"'{L}' is not a single token"
        letter_ids[L] = ids[0]
    preds = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i: i + batch_size]
        texts = [
            tok.apply_chat_template(
                build_messages(ex, retrieved[i + j] if retrieved else None),
                tokenize=False, add_generation_prompt=True,
            ) + "Answer:"
            for j, ex in enumerate(batch)
        ]
        enc = tok(texts, return_tensors="pt", padding=True,
        add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        for row, ex in zip(logits, batch):
            cand = LETTERS[: len(ex["options"])]
            scores = torch.stack([row[letter_ids[L]] for L in cand])
            preds.append(cand[int(scores.argmax())])
    return preds

@torch.no_grad()
def generate_answers(model, tok, message_lists, max_new_tokens=256, batch_size=8):
    tok = _prep_tokenizer(tok)
    outs = []
    for i in range(0, len(message_lists), batch_size):
        batch = message_lists[i : i + batch_size]
        texts = [tok.apply_chat_template(m, tokenize=False,
        add_generation_prompt=True) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
        add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
        )
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
        skip_special_tokens=True)
    return [o.strip() for o in outs]


_rouge, _bertscore = None, None

def text_metrics(preds, refs):
    import evaluate
    global _rouge, _bertscore
    _rouge = _rouge or evaluate.load("rouge")
    _bertscore = _bertscore or evaluate.load("bertscore")
    r = _rouge.compute(predictions=preds, references=refs, use_stemmer=True, use_aggregator=False)
    b = _bertscore.compute(predictions=preds, references=refs, lang="en", rescale_with_baseline=True)
    return {
        "rougeL": float(np.mean(r["rougeL"])),
        "bertscore_f1": float(np.mean(b["f1"])),
        "per_example_rougeL": r["rougeL"],
        "per_example_bertscore": b["f1"],
    }

def bootstrap_ci(correct, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    means = [rng.choice(correct, len(correct)).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

