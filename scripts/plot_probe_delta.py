#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot delta_k curves from `probe_vanilla_lp_projection.py` text log.

Reads the textual probe log (output of ``probe_vanilla_lp_projection.py``)
and renders the per-iteration ``delta_k`` for every probed sample so we can
visually inspect the FP potential function.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SAMPLE_HDR_RE = re.compile(r"probing sample_id=(\S+)")
ITER_RE = re.compile(r"iter (\d+): status=(\S+), delta_k=([0-9.eE+-]+|None|nan), term=")


def parse(log_path: Path):
    """Return {sample_id: [(iter, status, delta_k), ...]}."""
    blocks = {}
    current = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = SAMPLE_HDR_RE.search(line)
            if m:
                current = m.group(1)
                blocks[current] = []
                continue
            m = ITER_RE.search(line)
            if m and current is not None:
                it = int(m.group(1))
                status = m.group(2)
                raw = m.group(3)
                if raw.lower() in ("none", "nan"):
                    delta = float("nan")
                else:
                    delta = float(raw)
                blocks[current].append((it, status, delta))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="probe_vanilla*.log path")
    parser.add_argument("--output", required=True, help="output PNG path")
    args = parser.parse_args()

    log_path = Path(args.input)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parents[1] / log_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[1] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = parse(log_path)
    if not blocks:
        print(f"[error] no sample blocks found in {log_path}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    sids = list(blocks.keys())
    for i, sid in enumerate(sids):
        rows = blocks[sid]
        if not rows:
            continue
        iters = [r[0] for r in rows]
        deltas = [r[2] for r in rows]
        color = cmap(i / max(len(sids) - 1, 1))
        ax.plot(
            iters, deltas, marker="o", markersize=4, linewidth=1.5,
            label=f"sample {sid}", color=color, alpha=0.85,
        )
        # mark NON-OPTIMAL projection iterations
        for it, status, dv in rows:
            if status != "OPTIMAL":
                ax.scatter([it], [dv], color="red", marker="x", s=80, zorder=5)

    ax.set_xlabel("FP iteration $k$")
    ax.set_ylabel(r"$\delta_k$ (L1 distance from rounded $y$ to LP projection)")
    ax.set_title("Vanilla FP $\\delta_k$ trajectory (case14, probe samples)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
