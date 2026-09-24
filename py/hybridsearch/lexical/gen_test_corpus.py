"""Deterministic test corpus for the lexical engine's ctest gates.

Writes engine/tests/data/lexical/{corpus.tsv, queries.tsv}. The corpus is synthetic but
shaped like MS MARCO: Zipfian vocabulary with inflections (so the Porter stemmer merges
forms), stop words, possessives, numbers with separators, hyphens, a little non-ASCII,
duplicated passages (score ties), stop-word-only passages (length 0), and a few very long
passages (length quantisation far from exact). Global IDs are sparse and increasing.

The Lucene golden files next to them are produced by Lucene itself, not by this script:
  D=engine/tests/data/lexical
  tools/lucene_ref/run.sh analyze < $D/corpus.tsv | cut -f1,2 > $D/corpus.lucene_analyzed.tsv
  tools/lucene_ref/run.sh analyze < $D/tricky.tsv > $D/tricky.lucene.tsv
  tools/lucene_ref/run.sh score $D/corpus.tsv $D/queries.tsv 20 0.9 0.4 > $D/lucene_bm25_k20.tsv
  tools/lucene_ref/run.sh smallfloat | awk -F'\t' '$1=="b" || NR%50==0' > $D/lucene_smallfloat.tsv
(tricky.tsv holds hand-written edge cases: long tokens, emoji, CJK, Thai, Hebrew, numbers.)

Usage: python -m hybridsearch.lexical.gen_test_corpus
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "engine/tests/data/lexical"
SEED = 20260923

STEMS = """
run walk connect generate relate condition agree feed hope care pony sky cat dog house water
cell energy protein acid heart blood river mountain city country state price cost market bank
computer program system network data model train learn test result number value power light
sound music film book write read speak language word name call form work play move live die
grow change turn start end open close build design plan study teach school student class
temperature pressure weather climate rain snow storm wind cloud sun moon star planet earth
""".split()
SUFFIXES = ["", "", "", "s", "ing", "ed", "er", "ers", "ation", "ational", "ness", "ly", "ful", "able", "'s"]
STOP = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no",
        "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to",
        "was", "will", "with"]
EXTRAS = ["3.14", "1,000", "2019", "U.S.", "e-mail", "state-of-the-art", "café", "naïve", "Zürich", "東京",
          "\U0001F600", "O'Neil", "rock'n'roll", "x_y", "ab12cd", "100%", "$5.00", "C++", "Mr.", "HTTP://www.example.com/a?b=c"]


def main() -> None:
    rng = random.Random(SEED)
    words: list[str] = []
    for s in STEMS:
        for suf in SUFFIXES:
            if rng.random() < 0.5:
                words.append(s + suf)
    rng.shuffle(words)
    # Zipf weights over the vocabulary.
    weights = [1.0 / (i + 1) ** 1.05 for i in range(len(words))]

    def sentence(n: int) -> str:
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.28:
                w = rng.choice(STOP)
            elif r < 0.31:
                w = rng.choice(EXTRAS)
            else:
                w = rng.choices(words, weights)[0]
            if rng.random() < 0.08:
                w = w.capitalize()
            out.append(w)
        text = " ".join(out)
        if rng.random() < 0.5:
            text += rng.choice([".", "!", "?", ";", " ..."])
        return text

    docs: list[str] = []
    for i in range(2500):
        r = rng.random()
        if r < 0.01:
            text = " ".join(rng.choice(STOP) for _ in range(rng.randint(1, 6)))  # length 0 after analysis
        elif r < 0.02:
            text = sentence(rng.randint(400, 1500))  # long: coarse SmallFloat buckets
        elif r < 0.06 and docs:
            text = rng.choice(docs)  # exact duplicate: guaranteed score ties
        else:
            text = sentence(rng.randint(5, 70))
        docs.append(text)
    OUT.mkdir(parents=True, exist_ok=True)
    pid = 0
    with (OUT / "corpus.tsv").open("w", encoding="utf-8") as f:
        for text in docs:
            pid += rng.randint(1, 9)
            f.write(f"{pid}\t{text}\n")
    with (OUT / "queries.tsv").open("w", encoding="utf-8") as f:
        for q in range(300):
            n = rng.randint(1, 8)
            terms = []
            for _ in range(n):
                r = rng.random()
                if r < 0.15:
                    terms.append(rng.choice(STOP))
                elif r < 0.2:
                    terms.append("zzqx" + str(rng.randint(0, 9)))  # out of vocabulary
                elif r < 0.25 and terms:
                    terms.append(terms[-1])  # repeated term: boost 2
                else:
                    terms.append(rng.choices(words, weights)[0])
            f.write(f"{1000 + q}\t{' '.join(terms)}\n")
    print(f"wrote {len(docs)} docs and 300 queries to {OUT}")


if __name__ == "__main__":
    main()
