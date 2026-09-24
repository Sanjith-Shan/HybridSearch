#!/usr/bin/env bash
# M5 full-scale reranker training on a rented A100 (RunPod). PREPARED, NOT RUN.
#
# This script runs ON THE POD. It never creates or destroys a pod: per
# campaign/specs/RUNPOD_SHARED.md exactly one owner session does that. If the "Current pod"
# block there says "none", stop; do not deploy your own.
#
# From the Mac (the owner session supplies <ip> <port>):
#   rsync -avz --exclude .git --exclude .venv --exclude 'data/raw' --exclude 'data/embeddings' \
#     --exclude 'data/indexes' --exclude 'data/models' ~/Documents/HybridSearch/ \
#     root@<ip>:/workspace/HybridSearch/ -e "ssh -p <port>"
#   # the pod needs: data/rerank/mined/train50k.jsonl, data/subset/{train50k,train_tune},
#   # data/runs/reference/anserini-bm25-default.train_tune.trec and collection.tsv (3 GB):
#   rsync -avz data/raw/msmarco/collection.tsv root@<ip>:/workspace/HybridSearch/data/raw/msmarco/ -e "ssh -p <port>"
#   ssh root@<ip> -p <port> "cd /workspace/HybridSearch && bash scripts/full_scale/train_reranker_a100.sh mixed"
#   rsync -avz -e "ssh -p <port>" root@<ip>:/workspace/HybridSearch/data/rerank/runs/a100_mixed/best/ \
#     ~/Documents/HybridSearch/data/rerank/runs/a100_mixed/best/
#   rsync -avz -e "ssh -p <port>" root@<ip>:/workspace/HybridSearch/data/rerank/runs/a100_mixed/{train_summary.json,train_summary.json.meta.json,log.jsonl} \
#     ~/Documents/HybridSearch/data/rerank/runs/a100_mixed/
# Then evaluate/export on the Mac exactly as for the MPS runs (scripts/rerank_eval.sh).
set -euo pipefail
STRATEGY=${1:-mixed}            # random | bm25-hard | dense-hard | mixed  (use the MPS ablation winner)
STEPS=${STEPS:-5000}
cd "$(dirname "$0")/../.."
python3 -m venv .venv-pod --system-site-packages     # reuse the image's CUDA torch
. .venv-pod/bin/activate
pip install -q "transformers>=4.44" pyyaml numpy scipy tqdm safetensors
pip install -q -e . --no-deps
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import torch; assert torch.cuda.is_available(), "no CUDA"; print(torch.__version__, torch.cuda.get_device_name(0))
PY
OUT=data/rerank/runs/a100_${STRATEGY%-hard}
# resumable: rerun the same command after a disconnect and it continues from $OUT/last
RESUME=""; [ -f "$OUT/last/trainer_state.pt" ] && RESUME="--resume"
python -m hybridsearch.rerank.train --config py/hybridsearch/rerank/configs/a100.yaml \
  --set strategy=$STRATEGY max_steps=$STEPS out_dir=$OUT \
        "label=A100 SXM4 80GB RunPod bf16, $STRATEGY negatives" $RESUME 2>&1 | tee -a "$OUT.log"
rm -rf "$OUT/last"   # optimizer state is 2x the model; only best/ comes home
ls -la "$OUT/best"
