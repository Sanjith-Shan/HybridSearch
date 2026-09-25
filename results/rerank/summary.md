| model.candidates | split | queries | depth | backend | reranked | first stage | file |
|---|---|---|---|---|---|---|---|
| ours-ort-fp32.bm25full-q200 | dev | 200 | 100 | ort/cpu | RR@10 0.1014, nDCG@10 0.1237, R@100 0.6450 | RR@10 0.1906, nDCG@10 0.2262, R@100 0.6450 | results/rerank/eval/ours-ort-fp32.bm25full-q200.dev.json |
| ours-ort-int8.bm25full-q200 | dev | 200 | 100 | ort/cpu | RR@10 0.1050, nDCG@10 0.1301, R@100 0.6450 | RR@10 0.1906, nDCG@10 0.2262, R@100 0.6450 | results/rerank/eval/ours-ort-int8.bm25full-q200.dev.json |
| ours.bm25full-q200 | dev | 200 | 100 | torch/mps | RR@10 0.1014, nDCG@10 0.1237, R@100 0.6450 | RR@10 0.1906, nDCG@10 0.2262, R@100 0.6450 | results/rerank/eval/ours.bm25full-q200.dev.json |
| public.bm25full-q200 | dev | 200 | 100 | torch/mps | RR@10 0.3780, nDCG@10 0.4233, R@100 0.6450 | RR@10 0.1906, nDCG@10 0.2262, R@100 0.6450 | results/rerank/eval/public.bm25full-q200.dev.json |
| ours-ort-fp32.bm25full | dl19 | 43 | 100 | ort/cpu | nDCG@10 0.3950, RR(rel=2)@10 0.5708, R(rel=2)@100 0.4910 | nDCG@10 0.5058, RR(rel=2)@10 0.7024, R(rel=2)@100 0.4910 | results/rerank/eval/ours-ort-fp32.bm25full.dl19.json |
| ours-ort-int8.bm25full | dl19 | 43 | 100 | ort/cpu | nDCG@10 0.3922, RR(rel=2)@10 0.5532, R(rel=2)@100 0.4910 | nDCG@10 0.5058, RR(rel=2)@10 0.7024, R(rel=2)@100 0.4910 | results/rerank/eval/ours-ort-int8.bm25full.dl19.json |
| ours.bm25full | dl19 | 43 | 100 | torch/mps | nDCG@10 0.3950, RR(rel=2)@10 0.5708, R(rel=2)@100 0.4910 | nDCG@10 0.5058, RR(rel=2)@10 0.7024, R(rel=2)@100 0.4910 | results/rerank/eval/ours.bm25full.dl19.json |
| public.bm25full | dl19 | 43 | 100 | torch/mps | nDCG@10 0.7267, RR(rel=2)@10 0.8651, R(rel=2)@100 0.4910 | nDCG@10 0.5058, RR(rel=2)@10 0.7024, R(rel=2)@100 0.4910 | results/rerank/eval/public.bm25full.dl19.json |
| ours.bm25full | dl20 | 54 | 100 | torch/mps | nDCG@10 0.2902, RR(rel=2)@10 0.3587, R(rel=2)@100 0.5599 | nDCG@10 0.4796, RR(rel=2)@10 0.6533, R(rel=2)@100 0.5599 | results/rerank/eval/ours.bm25full.dl20.json |
| public.bm25full | dl20 | 54 | 100 | torch/mps | nDCG@10 0.6747, RR(rel=2)@10 0.8460, R(rel=2)@100 0.5599 | nDCG@10 0.4796, RR(rel=2)@10 0.6533, R(rel=2)@100 0.5599 | results/rerank/eval/public.bm25full.dl20.json |
