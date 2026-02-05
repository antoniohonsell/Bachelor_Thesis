#!/usr/bin/env python3
"""
stitching_trend.py

Your folder layout:
  activation_stitching_out_cifar10_resnet20_1/
  activation_stitching_out_cifar10_resnet20_2/
  activation_stitching_out_cifar10_resnet20_8/
  activation_stitching_out_cifar10_resnet20_16/
(each contains one or more results.json somewhere inside)

This script:
- recursively finds results.json under --root
- infers w from the *final number* in the parent folder name (e.g. "..._16" -> w=16)
- reads stitching.metrics.test.loss_perm
- prints a table: mean loss_perm for cuts 1..9, for w in {1,2,8,16}
- also prints counts per cell (how many files contributed)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

TARGET_WS = [1, 2, 8, 16]
TARGET_CUTS = list(range(1, 10))  # cuts 1..9

LOSS_PATH = ("stitching", "metrics", "test", "loss_perm")
CUTS_PATH = ("stitching", "cuts")

# still supported (if you ever store w inside json)
W_KEY_CANDIDATES_DEFAULT = ["w", "window", "width_multiplier"]

# match folder suffix "..._16" (or "-16")
SUFFIX_INT_RE = re.compile(r"(?:_|-)(\d+)$")


def _get_nested(d: Any, path: Iterable[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _to_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, str):
            s = x.strip().lower()
            for prefix in ("w=", "window=", "width_multiplier="):
                if s.startswith(prefix):
                    s = s[len(prefix) :]
            return int(s)
    except Exception:
        return None
    return None


def _extract_w_from_path(file_path: Path) -> Optional[int]:
    """
    Infer w from the right-most directory name that ends with _<int> or -<int>.
    Example: activation_stitching_out_cifar10_resnet20_16 -> 16
    """
    # iterate directories from deepest to shallowest
    for part in reversed(file_path.parts[:-1]):  # exclude filename
        m = SUFFIX_INT_RE.search(part)
        if not m:
            continue
        w = int(m.group(1))
        if w in TARGET_WS:
            return w
    return None


def _extract_w(obj: Dict[str, Any], w_keys: List[str], file_path: Path) -> Optional[int]:
    # 1) from JSON (direct)
    for k in w_keys:
        if k in obj:
            w = _to_int(obj[k])
            if w is not None:
                return w

    # 2) from JSON (inside typical config containers)
    for container in ("params", "param", "config", "cfg", "args", "hparams"):
        if container in obj and isinstance(obj[container], dict):
            for k in w_keys:
                if k in obj[container]:
                    w = _to_int(obj[container][k])
                    if w is not None:
                        return w

    # 3) from folder suffix (your layout)
    return _extract_w_from_path(file_path)


def _iter_results_json_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files = list(root.rglob("results.json"))
    return sorted(files)


def _as_float_list(x: Any) -> Optional[List[float]]:
    if isinstance(x, list) and all(isinstance(v, (int, float)) for v in x):
        return [float(v) for v in x]
    return None


def _collect_one_file(
    file_path: Path,
    w_keys: List[str],
    acc: DefaultDict[Tuple[int, int], List[float]],
    verbose: bool = False,
) -> None:
    try:
        obj = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to read {file_path}: {e}", file=sys.stderr)
        return

    if not isinstance(obj, dict):
        if verbose:
            print(f"[WARN] {file_path} is not a JSON object", file=sys.stderr)
        return

    w = _extract_w(obj, w_keys, file_path)
    if w not in TARGET_WS:
        if verbose:
            print(f"[SKIP] {file_path} (w={w})", file=sys.stderr)
        return

    loss_perm = _get_nested(obj, LOSS_PATH)
    loss_list = _as_float_list(loss_perm)
    if loss_list is None:
        if verbose:
            print(f"[WARN] {file_path} missing stitching.metrics.test.loss_perm", file=sys.stderr)
        return

    cuts = _get_nested(obj, CUTS_PATH)
    if isinstance(cuts, list) and all(isinstance(c, (int, float)) for c in cuts):
        cuts_list = [int(c) for c in cuts]
    else:
        # fallback: indices are "cuts"
        cuts_list = list(range(len(loss_list)))

    idx_of = {c: i for i, c in enumerate(cuts_list)}

    for cut in TARGET_CUTS:
        i = idx_of.get(cut, None)
        if i is None or i < 0 or i >= len(loss_list):
            continue
        acc[(w, cut)].append(loss_list[i])


def _print_table(means: Dict[Tuple[int, int], float], counts: Dict[Tuple[int, int], int]) -> None:
    headers = ["cut"] + [f"w={w}" for w in TARGET_WS]
    rows: List[List[str]] = []

    for cut in TARGET_CUTS:
        row = [str(cut)]
        for w in TARGET_WS:
            key = (w, cut)
            row.append(f"{means[key]:.6g}" if key in means else "-")
        rows.append(row)

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(cells: List[str]) -> str:
        return " | ".join(cells[i].rjust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)

    print(fmt_line(headers))
    print(sep)
    for r in rows:
        print(fmt_line(r))

    print("\nCounts (n results.json contributing to each mean):")
    print(fmt_line(headers))
    print(sep)
    for cut in TARGET_CUTS:
        r = [str(cut)]
        for w in TARGET_WS:
            r.append(str(counts.get((w, cut), 0)))
        print(fmt_line(r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Root directory that contains activation_stitching_out_* folders (default: current directory).",
    )
    ap.add_argument(
        "--w-keys",
        default=",".join(W_KEY_CANDIDATES_DEFAULT),
        help='Comma-separated JSON keys to try for w (default: "w,window,width_multiplier").',
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    w_keys = [s.strip() for s in args.w_keys.split(",") if s.strip()]

    files = _iter_results_json_files(root)
    if not files:
        print(f"No results.json found under: {root}", file=sys.stderr)
        return 2

    acc: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)
    for fp in files:
        _collect_one_file(fp, w_keys, acc, verbose=args.verbose)

    means: Dict[Tuple[int, int], float] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for key, vals in acc.items():
        if vals:
            means[key] = statistics.fmean(vals)
            counts[key] = len(vals)

    if not means:
        print(
            "No usable data found.\n"
            "Expected to infer w from folder suffix (e.g. ..._1, ..._2, ..._8, ..._16)\n"
            "and to find stitching.metrics.test.loss_perm in each results.json.",
            file=sys.stderr,
        )
        return 3

    _print_table(means, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())