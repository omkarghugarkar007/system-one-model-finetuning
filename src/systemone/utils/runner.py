"""Run bookkeeping: every experiment writes a directory that explains itself.

A research project's unit of output is not a number, it is a reproducible run.
Each one gets `runs/<date>-<slug>/` holding:

    manifest.json   config, git commit, versions, device, seeds, timings
    metrics.json    the numbers, flat and machine-readable
    report.txt      what a human reads
    *.npz           bulk arrays, gitignored

The manifest records the environment because "it worked on the M4 in fp32" and
"it worked on a T4 in fp16" are different claims about a calibrated model, and
the difference will not be visible in the metrics.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path

import numpy as np

__all__ = ["Run", "git_commit", "json_safe"]


def git_commit(default: str = "unknown") -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5).stdout
            return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except Exception:                                        # noqa: BLE001
        pass
    return default


def json_safe(obj):
    """Make numpy / dataclasses / paths survive json.dump without surprises."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return obj


class Run:
    """One experiment's output directory.

    Use as a context manager so wall-clock and failure state are recorded even
    when the experiment raises -- a crashed run that leaves no trace is the
    most expensive kind.
    """

    def __init__(self, name: str, config: dict | None = None,
                 root: str | Path = "runs", tag: str = ""):
        slug = f"{date.today().isoformat()}-{name}" + (f"-{tag}" if tag else "")
        self.dir = Path(root) / slug
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = Path(root) / f"{slug}-{n}"
        self.dir.mkdir(parents=True)
        self.name = name
        self.config = dict(config or {})
        self.metrics: dict = {}
        self.lines: list[str] = []
        self.t0 = time.time()
        self.status = "running"

    # ------------------------------------------------------------- recording
    def log(self, line: str = "", echo: bool = True):
        self.lines.append(line)
        if echo:
            print(line, flush=True)

    def section(self, title: str):
        self.log()
        self.log(title)
        self.log("=" * len(title))

    def metric(self, key: str, value, **extra):
        self.metrics[key] = json_safe(value)
        if extra:
            self.metrics.setdefault("_detail", {})[key] = json_safe(extra)

    def array(self, name: str, **arrays):
        np.savez_compressed(self.dir / f"{name}.npz", **arrays)

    def artifact(self, name: str, obj):
        (self.dir / name).write_text(json.dumps(json_safe(obj), indent=2))

    # --------------------------------------------------------------- manifest
    def _manifest(self) -> dict:
        import torch  # noqa: PLC0415
        env = {"python": platform.python_version(),
               "platform": platform.platform(),
               "numpy": np.__version__}
        try:
            env["torch"] = torch.__version__
            env["mps"] = torch.backends.mps.is_available()
            env["cuda"] = torch.cuda.is_available()
        except Exception:                                    # noqa: BLE001
            pass
        return {"name": self.name, "status": self.status,
                "git_commit": git_commit(),
                "started": self.t0, "wall_seconds": round(time.time() - self.t0, 1),
                "config": json_safe(self.config), "env": env}

    def close(self, status: str = "ok"):
        self.status = status
        (self.dir / "manifest.json").write_text(
            json.dumps(self._manifest(), indent=2))
        (self.dir / "metrics.json").write_text(
            json.dumps(json_safe(self.metrics), indent=2))
        (self.dir / "report.txt").write_text("\n".join(self.lines) + "\n")
        return self.dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.log(f"\nFAILED: {exc_type.__name__}: {exc}")
            self.close("failed")
        else:
            self.close("ok")
        return False
