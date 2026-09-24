"""Tiny random-init BERT models + synthetic MS MARCO-shaped data, so rerank/export tests
run offline in seconds (no hub download, no real data)."""
from __future__ import annotations

import random
from pathlib import Path

WORDS = ("the a of is what how cat dog fish bird tree river city capital country peru lima "
         "france paris water blood flow heart run fast slow big small red blue green time year "
         "day night food eat drink sleep book read write code search engine rank query passage").split()


def write_vocab(path: Path) -> Path:
    toks = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ",", "?", ":", "'"]
    toks += WORDS + ["##s", "##ing", "##ed"]
    path.write_text("\n".join(toks) + "\n")
    return path


def tiny_tokenizer(d: Path):
    from transformers import BertTokenizer

    d.mkdir(parents=True, exist_ok=True)
    write_vocab(d / "vocab.txt")
    # transformers 5: the constructor takes ``vocab=``; from_pretrained reads vocab.txt
    tok = BertTokenizer.from_pretrained(str(d), do_lower_case=True)
    tok.save_pretrained(str(d))
    return tok


def tiny_config(vocab_size: int, num_labels: int = 1):
    from transformers import BertConfig

    return BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
                      num_labels=num_labels)


def tiny_cross_encoder(d: Path, seed: int = 0) -> Path:
    import torch
    from transformers import BertForSequenceClassification

    tok = tiny_tokenizer(d)
    torch.manual_seed(seed)
    BertForSequenceClassification(tiny_config(tok.vocab_size)).save_pretrained(str(d))
    return d


def tiny_bert(d: Path, seed: int = 0) -> Path:
    import torch
    from transformers import BertModel

    tok = tiny_tokenizer(d)
    torch.manual_seed(seed)
    BertModel(tiny_config(tok.vocab_size), add_pooling_layer=False).save_pretrained(str(d))
    return d


def sentence(rng: random.Random, lo: int = 3, hi: int = 20) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))


def synthetic_corpus(d: Path, n_docs: int = 400, n_queries: int = 40, seed: int = 0) -> dict:
    """collection.tsv, queries/qrels for train + val, a mined jsonl and a val BM25-ish run."""
    import json

    rng = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    docs = {str(i): sentence(rng, 5, 30) for i in range(n_docs)}
    docs["399"] = docs["3"]  # a duplicate text of a positive (pid 3 is q0's positive below)
    (d / "collection.tsv").write_text("".join(f"{p}\t{t}\n" for p, t in docs.items()))
    qs, qrels, mined, run = {}, [], [], []
    for i in range(n_queries):
        qid = str(1000 + i)
        pos = str(3 + 7 * i % 390)
        qs[qid] = sentence(rng, 2, 8)
        qrels.append(f"{qid} 0 {pos} 1\n")
        judged_extra = str((int(pos) + 1) % 390)  # a second judged pid that also sits in the pools
        qrels.append(f"{qid} 0 {judged_extra} 1\n")
        pool = [str(rng.randrange(n_docs)) for _ in range(30)] + [pos, judged_extra, "399"]
        mined.append({"qid": qid, "pos": [pos], "bm25": pool[:20] + [pos, "399"],
                      "dense": pool[10:] + [judged_extra], "random": [str(rng.randrange(n_docs)) for _ in range(20)]})
        cands = [pos] + [str(rng.randrange(n_docs)) for _ in range(9)]
        for r, p in enumerate(dict.fromkeys(cands)):
            run.append(f"{qid} Q0 {p} {r + 1} {10 - r:.3f} tiny\n")
    (d / "queries.tsv").write_text("".join(f"{q}\t{t}\n" for q, t in qs.items()))
    (d / "qrels.tsv").write_text("".join(qrels))
    (d / "mined.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mined))
    (d / "val.trec").write_text("".join(run))
    return {"docs": docs, "queries": qs, "mined": mined}
