// Constant-arrival-rate load against the broker's /api/search.
//
// Open-loop on purpose: k6's constant-arrival-rate executor starts iterations on
// a schedule regardless of how long earlier ones took, so a slow response does
// not hide the requests that should have been sent behind it (coordinated
// omission). If k6 runs out of VUs it counts dropped_iterations; any run with
// dropped iterations is invalid and bench/load/summarize.py refuses it.
//
//   k6 run -e RATE=200 -e DURATION=60s -e QUERIES=data/raw/msmarco/queries.dev.small.tsv \
//          --out json=results/load/raw.json bench/load/search.js
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";

const BASE = __ENV.BASE_URL || "http://localhost:8080";
const RATE = parseInt(__ENV.RATE || "100", 10);
const DURATION = __ENV.DURATION || "60s";
const MODE = __ENV.MODE || "hybrid";
const RERANK = __ENV.RERANK || "true";
const DEADLINE = __ENV.DEADLINE_MS || "300";

const queries = new SharedArray("queries", () =>
  open(__ENV.QUERIES || "../../data/raw/msmarco/queries.dev.small.tsv")
    .split("\n")
    .filter((l) => l.length > 0)
    .map((l) => {
      const tab = l.indexOf("\t");
      return { qid: l.slice(0, tab), text: l.slice(tab + 1) };
    })
);

export const options = {
  discardResponseBodies: true,
  scenarios: {
    search: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: Math.max(50, RATE),
      maxVUs: Math.max(500, RATE * 5),
    },
  },
  summaryTrendStats: ["min", "med", "p(90)", "p(99)", "p(99.9)", "max"],
};

export default function () {
  const q = queries[Math.floor(Math.random() * queries.length)];
  const url =
    `${BASE}/api/search?q=${encodeURIComponent(q.text)}` +
    `&mode=${MODE}&rerank=${RERANK}&deadlineMs=${DEADLINE}&k=10`;
  const res = http.get(url, { tags: { name: "search" }, timeout: "10s" });
  check(res, { "status 200": (r) => r.status === 200 });
}
