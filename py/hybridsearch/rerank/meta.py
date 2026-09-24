"""``.meta.json`` sidecars for every number written under ``results/``.

Every result file gets a sidecar naming the hardware, OS, library versions, date and the
exact command that produced it, plus task-specific fields (steps, examples seen, ...).
macOS timings are labelled ``dev-signal-only`` because threads cannot be pinned.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def hardware() -> dict:
    cpu = _sh(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else ""
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info = {
        "cpu": cpu or platform.processor(),
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()} ({platform.platform()})",
        "python": platform.python_version(),
        "pinned_threads": False,
        # other agents share this laptop; load at write time says how contended timings were
        "loadavg_1_5_15": [round(x, 2) for x in os.getloadavg()],
    }
    if sys.platform == "darwin":
        info["timing_label"] = "dev-signal-only (macOS, threads cannot be pinned)"
        info["mem_bytes"] = int(_sh(["sysctl", "-n", "hw.memsize"]) or 0)
    return info


def versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "sentence_transformers", "onnx", "onnxruntime", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


def git_sha() -> str:
    repo = Path(__file__).resolve().parents[3]
    return _sh(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]) or "no-commit"


def write_meta(result_path: str | os.PathLike, **fields) -> Path:
    """Write ``<result_path>.meta.json``. ``fields`` are merged over the standard block."""
    meta = {
        "result": str(result_path),
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "hardware": hardware(),
        "versions": versions(),
        "git_sha": git_sha(),
    }
    meta.update(fields)
    p = Path(f"{result_path}.meta.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return p
