#!/usr/bin/env bash
# M3 on the 1M-passage subset: exact ground truth, in-memory Vamana (recall/QPS + dev MRR),
# DiskANN-style SSD index (partitioned build, cold/warm sweeps, SSD reads, dev MRR), and
# the hnswlib / FAISS comparison. Every result lands in results/vector/ with a sidecar.
#
# Usage: scripts/vector/run_1m.sh [step ...]
#   steps: gt vamana vamana_dev disk disk_part disk_sweep disk_dev compare mrr plots
#   disk      = SSD index written from the saved in-memory Vamana graph (single partition)
#   disk_part = the paper's memory-limited build: overlapping k-means partitions (1 GB budget), merge
# Env: HS_THREADS (default 4), N (default all rows), LABEL_N (for file names),
#      DISKV=disk|disk_part  which SSD index disk_sweep / disk_dev / plots use (default disk)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$here"
export HS_THREADS="${HS_THREADS:-4}" PYTHONPATH=py
PY=.venv/bin/python; B=engine/build-vec
E=data/embeddings/1m; BASE=$E/passages.fbin; DOCIDS=data/subset/1m/docids.u64bin
N="${N:-0}"; TAG="${LABEL_N:-1m}"
IDX=data/indexes/vector/$TAG; R=results/vector; mkdir -p "$IDX" "$R"
DISKV="${DISKV:-disk}"; DP=${DISKV/_/}_${TAG}   # result-file prefix: disk_1m or diskpart_1m
TT=$E/queries.train_tune.fbin; DEV=$E/queries.dev.fbin
GT_TT=$IDX/gt.train_tune.bin; GT_DEV=$IDX/gt.dev.bin
maxn=(); [ "$N" != 0 ] && maxn=(--max-n "$N")
meta() { $PY -m hybridsearch.bench.meta "$1" --command "scripts/vector/run_1m.sh $step" --threads "$HS_THREADS" --cache "$2" \
           --set "base=$BASE${N:+ first $N rows}" "${@:3}" >/dev/null; }
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(gt vamana vamana_dev disk disk_sweep disk_dev compare mrr plots)
for step in "${steps[@]}"; do
  echo "=== $step $(date +%T)"
  case $step in
  gt)
    $B/hs_vec_groundtruth --base $BASE ${N:+--max-base $N} --queries $TT --k 100 --out $GT_TT | tee $R/gt_${TAG}.json
    meta $R/gt_${TAG}.json n/a "queries=train_tune"
    if [ -f $E/gt.dev.top100.u64bin ] && [ "$N" = 0 ]; then
      $B/hs_vec_groundtruth --base $BASE --queries $DEV --max-queries 300 --k 10 --out $IDX/gt.dev300.ours.bin >/dev/null
      $PY -m hybridsearch.bench.gt_from_pids $E/gt.dev.top100.u64bin $E/gt.dev.top100.scores.fbin $DOCIDS $GT_DEV \
         --check $IDX/gt.dev300.ours.bin | tee $R/gt_crosscheck_${TAG}.json
      meta $R/gt_crosscheck_${TAG}.json n/a "what=hs_vec_groundtruth vs hybridsearch.dense.flat (faiss IndexFlatIP) on 300 dev queries"
    else
      $B/hs_vec_groundtruth --base $BASE ${N:+--max-base $N} --queries $DEV --k 100 --out $GT_DEV
    fi ;;
  vamana)
    $B/hs_vec_build --base $BASE "${maxn[@]}" --out $IDX/vamana_R64_L100_a1.2.graph --R 64 --L 100 --alpha 1.2 \
       --stats $R/vamana_${TAG}.build.jsonl --verbose
    meta $R/vamana_${TAG}.build.jsonl n/a
    rm -f $R/vamana_${TAG}.search.jsonl
    $B/hs_vec_search --base $BASE "${maxn[@]}" --graph $IDX/vamana_R64_L100_a1.2.graph --queries $TT --gt $GT_TT \
       --L 10,15,20,30,40,60,80,120,160,240,320 --label vamana-R64 --out $R/vamana_${TAG}.search.jsonl
    meta $R/vamana_${TAG}.search.jsonl warm "queries=train_tune (2000)" "query_threads=1" ;;
  vamana_dev)
    rm -f $R/vamana_${TAG}.dev.jsonl; mkdir -p $R/runs
    $B/hs_vec_search --base $BASE "${maxn[@]}" --graph $IDX/vamana_R64_L100_a1.2.graph --queries $DEV --gt $GT_DEV \
       --L 10,20,40,80,160 --label vamana-R64-dev --out $R/vamana_${TAG}.dev.jsonl \
       --trec-prefix $R/runs/vamana_${TAG}.dev --qids $E/queries.dev.qids.txt --docids $DOCIDS
    meta $R/vamana_${TAG}.dev.jsonl warm "queries=dev (6980)" ;;
  disk)
    $B/hs_vec_disk_build --base $BASE "${maxn[@]}" --docids $DOCIDS --out $IDX/disk --R 64 --L 100 --alpha 1.2 \
       --pq-M 96 --pq-sample 200000 --graph $IDX/vamana_R64_L100_a1.2.graph 2>&1 | tee $R/disk_${TAG}.build.log
    cp $IDX/disk/build.json $R/disk_${TAG}.build.json
    meta $R/disk_${TAG}.build.json n/a "graph=reused from step vamana (build time of the graph is in vamana_${TAG}.build.jsonl)" ;;
  disk_part)
    $B/hs_vec_disk_build --base $BASE "${maxn[@]}" --docids $DOCIDS --out $IDX/disk_part --R 64 --L 100 --alpha 1.2 \
       --pq-M 96 --pq-sample 200000 --ram-gb 1.0 2>&1 | tee $R/diskpart_${TAG}.build.log
    cp $IDX/disk_part/build.json $R/diskpart_${TAG}.build.json
    meta $R/diskpart_${TAG}.build.json n/a "ram_budget=1.0 GB for partition vectors+graph" ;;
  disk_sweep)
    out=$R/${DP}.sweep.jsonl; rm -f $out
    for W in 2 4 8; do
      $B/hs_vec_disk_search --index $IDX/$DISKV --queries $TT --gt $GT_TT --L 10,20,30,50,75,100,150,200 --W $W \
        --mode cold --label ${DISKV/_/} --out $out
    done
    $B/hs_vec_disk_search --index $IDX/$DISKV --queries $TT --gt $GT_TT --L 10,20,30,50,75,100,150,200 --W 4 \
      --mode cold --cache 20000 --label ${DISKV/_/} --out $out
    $B/hs_vec_disk_search --index $IDX/$DISKV --queries $TT --gt $GT_TT --L 10,20,50,100,200 --W 4 \
      --mode cold --io seq --label ${DISKV/_/}-seqio --out $out
    $B/hs_vec_disk_search --index $IDX/$DISKV --queries $TT --gt $GT_TT --L 10,20,30,50,75,100,150,200 --W 4 \
      --mode warm --label ${DISKV/_/} --out $out
    meta $out "cold (F_NOCACHE) and warm (page cache) rows, see mode field" "queries=train_tune (2000)" "query_threads=1" ;;
  disk_dev)
    out=$R/${DP}.dev.jsonl; rm -f $out; mkdir -p $R/runs
    $B/hs_vec_disk_search --index $IDX/$DISKV --queries $DEV --gt $GT_DEV --L 10,20,50,100,200 --W 4 --mode cold \
       --label ${DISKV/_/}-dev --out $out --trec-prefix $R/runs/${DP}.dev --qids $E/queries.dev.qids.txt
    meta $out cold "queries=dev (6980)" ;;
  compare)
    out=$R/ann_compare_${TAG}.jsonl; rm -f $out
    $PY -m hybridsearch.bench.ann_compare --base $BASE ${N:+--max-n $N} --queries $TT --gt $GT_TT --out $out \
       --build-threads $HS_THREADS
    meta $out warm "queries=train_tune (2000)" "search_threads=1" ;;
  mrr)
    ref=data/runs/reference/dense-flat-1m.dev.trec
    [ -f $ref ] || { $PY -m hybridsearch.bench.gt_to_trec $GT_DEV $E/queries.dev.qids.txt $DOCIDS $R/runs/exact_${TAG}.dev.trec exact; ref=$R/runs/exact_${TAG}.dev.trec; }
    $PY -m hybridsearch.bench.mrr_points --runs $R/runs/vamana_${TAG}.dev.L*.trec $R/runs/disk*_${TAG}.dev.W4.L*.trec \
       --qrels data/subset/1m/qrels.dev.tsv --reference $ref \
       --points $R/vamana_${TAG}.dev.jsonl $R/disk*_${TAG}.dev.jsonl --out $R/mrr_dev_${TAG}.json
    meta $R/mrr_dev_${TAG}.json n/a "reference=$ref" ;;
  plots)
    $PY -m hybridsearch.bench.plot_vector compare $R/ann_compare_${TAG}.png $R/vamana_${TAG}.search.jsonl $R/ann_compare_${TAG}.jsonl \
       <($PY -c "import json,sys; [print(l.strip()) for l in open('$R/${DP}.sweep.jsonl') if json.loads(l)['W']==4 and json.loads(l)['cache_nodes']==0 and json.loads(l)['io']=='pool']")
    $PY -m hybridsearch.bench.plot_vector disk $R/${DP} $R/${DP}.sweep.jsonl $R/mrr_dev_${TAG}.json ;;
  *) echo "unknown step $step"; exit 1 ;;
  esac
done
