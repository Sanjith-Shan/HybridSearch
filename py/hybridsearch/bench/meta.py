"""Write the .meta.json sidecar for a result file produced by a C++ hs_vec_* tool.

    python -m hybridsearch.bench.meta results/vector/x.jsonl --command "..." \
        --cache cold --threads 4 [--build-dir engine/build-vec] [--set key=value ...]

Records CPU, cores, OS, compiler + flags (from the CMake cache), SIMD path, git SHA,
pinned=false, load average at write time, and the dev-signal-only label.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from ..eval.results import REPO, machine_meta


def compiler_info(build_dir: Path) -> dict:
    cache = build_dir / "CMakeCache.txt"
    info = {"build_dir": str(build_dir.relative_to(REPO)) if build_dir.is_relative_to(REPO) else str(build_dir)}
    if cache.exists():
        txt = cache.read_text()
        def get(key):
            m = re.search(rf"^{key}:[A-Z]+=(.*)$", txt, re.M)
            return m.group(1) if m else None
        cxx = get("CMAKE_CXX_COMPILER")
        info["cxx"] = cxx
        info["build_type"] = get("CMAKE_BUILD_TYPE")
        info["cxx_flags_release"] = get("CMAKE_CXX_FLAGS_RELEASE")
        native = get("HS_NATIVE")
        info["native_flag"] = "-mcpu=native" if native in ("ON", "1", "TRUE") and os.uname().machine == "arm64" else (
            "-march=native" if native in ("ON", "1", "TRUE") else None)
        if cxx:
            try:
                info["cxx_version"] = subprocess.check_output([cxx, "--version"], text=True).splitlines()[0]
            except Exception:
                pass
    info["simd"] = "neon" if os.uname().machine == "arm64" else "avx2-or-scalar"
    return info


def memory_state() -> dict:
    out = {}
    for key, cmd in (("swapusage", ["sysctl", "-n", "vm.swapusage"]), ("memory_pressure", ["memory_pressure"])):
        try:
            txt = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            out[key] = txt.strip().splitlines()[-1] if key == "memory_pressure" else txt.strip()
        except Exception:
            pass
    return out


def sidecar(result: str, command: str, cache: str, threads: int | None, build_dir: str,
            extra: dict | None = None) -> Path:
    load = os.getloadavg()
    meta = machine_meta(
        command=command,
        cache=cache,
        threads=threads,
        compiler=compiler_info((REPO / build_dir).resolve()),
        load_average_1_5_15=[round(x, 2) for x in load],
        memory_at_write_time=memory_state(),
        timing_label=("dev-signal-only (macOS, threads not pinned; machine shared with other jobs, "
                      f"load avg {load[0]:.1f} on {os.cpu_count()} cores at write time; see memory_at_write_time "
                      "for swap: page-ins inflate latency of RAM-resident indexes)"),
    )
    if extra:
        meta.update(extra)
    out = Path(str(result) + ".meta.json")
    out.write_text(json.dumps(meta, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hybridsearch.bench.meta")
    ap.add_argument("result")
    ap.add_argument("--command", required=True)
    ap.add_argument("--cache", default="n/a", help="warm | cold | n/a")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--build-dir", default="engine/build-vec")
    ap.add_argument("--set", nargs="*", default=[], help="extra key=value pairs")
    a = ap.parse_args(argv)
    extra = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        extra[k] = v
    print(sidecar(a.result, a.command, a.cache, a.threads, a.build_dir, extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
