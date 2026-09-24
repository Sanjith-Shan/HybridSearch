"""Per-(query, doc) score agreement between our raw runs and Anserini's run (streams one query at a time).

Anserini's printed scores are rounded to 1e-4 by ScoreTiesAdjusterReranker (minus 1e-6 per tie), so the
comparison is round(ours, 4) vs round(anserini, 4); textbook scores are divided by (k1 + 1) = 1.9 first.
Usage: python -m hybridsearch.lexical.score_agreement OURS_LUCENE.raw OURS_TEXTBOOK.raw ANSERINI.trec OUT.json
"""
import itertools
import json
import sys


def groups(path, parse):
    with open(path) as f:
        for q, it in itertools.groupby((parse(l) for l in f), key=lambda x: x[0]):
            yield q, {d: s for _, d, s in it}


def raw(l):
    q, d, _r, s, _b = l.split("\t")
    return q, d, float(s)


def trec(l):
    q, _, d, _r, s, _t = l.split()
    return q, d, float(s)


def main():
    lucene, textbook, ans, out = sys.argv[1:5]
    res = {}
    for name, path, scale in (("lucene", lucene, 1.0), ("textbook", textbook, 1.9)):
        a_it = groups(ans, trec)
        n = same = only_ours = only_ans = mism_perturbed = 0
        maxd = 0.0
        for (q, ours), (qa, a) in zip(groups(path, raw), a_it):
            if q != qa:
                raise SystemExit(f"query order differs: {q} vs {qa}")
            for d, s in ours.items():
                if d not in a:
                    only_ours += 1
                    continue
                diff = abs(round(s / scale, 4) - round(a[d], 4))
                n += 1
                same += diff <= 1.01e-4
                if diff > 1.01e-4 and abs(a[d] * 1e4 - round(a[d] * 1e4)) > 1e-3:
                    mism_perturbed += 1  # Anserini's printed score carries its -1e-6-per-tie offset
                maxd = max(maxd, diff)
            only_ans += sum(1 for d in a if d not in ours)
        res[name] = {"pairs_in_both": n, "pairs_only_ours": only_ours, "pairs_only_anserini": only_ans,
                     "within_1e-4": same, "mismatches_where_anserini_score_is_tie_perturbed": mism_perturbed, "frac_within_1e-4": same / n, "max_abs_diff": round(maxd, 6)}
    res["note"] = ("Anserini scores are rounded to 1e-4 and then lowered by 1e-6 per position in a tie run "
                   "(ScoreTiesAdjusterReranker); textbook scores are divided by k1+1=1.9 before comparing.")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
