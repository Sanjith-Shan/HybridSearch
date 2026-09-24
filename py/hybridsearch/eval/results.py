"""Write a result file under results/ with a .meta.json sidecar (hardware, OS, date,
command, git SHA), per the repo rule that every reported number lives in results/."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, cwd=REPO).strip()
    except Exception:
        return None


def machine_meta(**extra) -> dict:
    meta = {
        "cpu": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "cores": os.cpu_count(),
        "mem_bytes": int(_sh(["sysctl", "-n", "hw.memsize"]) or 0),
        "os": f"{platform.system()} {platform.release()} (macOS {platform.mac_ver()[0]})",
        "python": platform.python_version(),
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "pinned": False,
        "timing_label": "dev-signal-only (macOS, shared & overloaded machine, threads not pinned)",
    }
    meta.update(extra)
    return meta


def write_result(rel_path: str, payload, command: str, **meta_extra) -> Path:
    out = RESULTS / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        out.write_text(payload)
    else:
        out.write_text(json.dumps(payload, indent=2) + "\n")
    meta = machine_meta(command=command, **meta_extra)
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
