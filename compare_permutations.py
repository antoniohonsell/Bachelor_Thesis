#!/usr/bin/env python3
"""
compare_permutations.py

Compare permutations found by:
  - weight matching (expects .pkl saved as {perm_name: np.ndarray})
  - activation matching (accepts .json / .pt / .pkl)

Metrics per perm key (layer / perm group):
  (A) Combinatorial agreement:
      - Hamming agreement between P_wgt and P_act
      - Fixed-point fraction of Q = inv(P_wgt) o P_act
      - Cycle structure of Q
      - (optional) Cayley distance = n - #cycles

  (B) Geometric / representation agreement (optional):
      - CKA(A_features, permute(B_features, P_wgt))
      - CKA(A_features, permute(B_features, P_act))

Notes on permutation convention:
  p[i] = j means "A unit i matches B unit j" (A -> B assignment).


HOW TO USE IT:
python compare_permutations.py \
  --act-perm path/to/activation/permutations.pt \
  --wgt-perm path/to/weight/permutation_seed0.pkl \
  --out-json path/to/perm_compare_report.json

WITH OPTIONAL CKA:
python compare_permutations.py \
  --act-perm ... --wgt-perm ... \
  --features-a path/to/features_A.pt \
  --features-b path/to/features_B.pt \
  --unit-dim 1
  
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch


# -------------------------
# Optional: import CKA from "elsewhere"
# -------------------------
def _load_cka_fn():
    """
    Try to import a cka(x, y) callable from your repo.
    If you keep it somewhere else, just change this function.
    """
    # Example (your repo has AlignmentMetrics.cka)
    try:
        from metrics_platonic import AlignmentMetrics  # type: ignore
        return AlignmentMetrics.cka
    except Exception:
        pass

    # Fallback: if you have a standalone cka.py on PYTHONPATH with `def cka(x, y): ...`
    try:
        from cka import cka  # type: ignore
        return cka
    except Exception as e:
        raise ImportError(
            "Could not import a CKA function. Edit _load_cka_fn() to point to your implementation."
        ) from e


# -------------------------
# IO helpers
# -------------------------
def _resolve_rel(base_path: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_path)), maybe_rel))


def _torch_load(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def _load_any(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext in [".pt", ".pth"]:
        return _torch_load(path)
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file extension: {ext} (path={path})")


def _extract_perm_payload(obj: Any, src_path: str) -> Any:
    """
    Support a few common wrappers:
      - direct dict of permutations
      - dict with key 'permutations'
      - results dict with 'permutations_files' pointing to pickle/pt/json
    """
    if isinstance(obj, dict):
        if "permutations_files" in obj and isinstance(obj["permutations_files"], dict):
            pf = obj["permutations_files"]
            # prefer pickle if available, then pt, then json
            for k in ["pickle", "pt", "json"]:
                if k in pf and pf[k]:
                    perm_path = _resolve_rel(src_path, str(pf[k]))
                    return _extract_perm_payload(_load_any(perm_path), perm_path)

        if "permutations" in obj:
            return obj["permutations"]

    return obj


def _to_long_1d(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, list):
        t = torch.tensor(x)
    else:
        raise TypeError(f"Unsupported permutation value type: {type(x)}")

    if t.ndim != 1:
        raise ValueError(f"Permutation must be 1D, got shape {tuple(t.shape)}")
    return t.to(dtype=torch.long)


def _normalize_perm_dict(obj: Any, src_path: str) -> Dict[str, torch.Tensor]:
    """
    Returns dict[str, LongTensor].
    Keys are stringified, but we also attempt to reconcile common key mismatches later.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict-like permutations in {src_path}, got {type(obj)}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        out[str(k)] = _to_long_1d(v)
    return out


def load_permutations(path: str) -> Dict[str, torch.Tensor]:
    raw = _load_any(path)
    payload = _extract_perm_payload(raw, path)
    return _normalize_perm_dict(payload, path)


# -------------------------
# Permutation math
# -------------------------
def is_valid_permutation(p: torch.Tensor) -> bool:
    n = int(p.numel())
    if n == 0:
        return False
    if p.min().item() < 0 or p.max().item() >= n:
        return False
    s = torch.sort(p).values
    return torch.equal(s, torch.arange(n, dtype=torch.long))


def invert_perm(p_a_to_b: torch.Tensor) -> torch.Tensor:
    """
    p maps A-index -> B-index.
    Returns inv mapping B-index -> A-index.
    """
    n = int(p_a_to_b.numel())
    inv = torch.empty(n, dtype=torch.long)
    inv[p_a_to_b] = torch.arange(n, dtype=torch.long)
    return inv


def compose_q(p_wgt: torch.Tensor, p_act: torch.Tensor) -> torch.Tensor:
    """
    Q = inv(P_wgt) o P_act
    where P_wgt, P_act: A -> B (same convention).
    Q: A -> A
    """
    if p_wgt.numel() != p_act.numel():
        raise ValueError(f"Size mismatch: |P_wgt|={p_wgt.numel()} vs |P_act|={p_act.numel()}")
    inv_w = invert_perm(p_wgt)
    return inv_w[p_act]


def cycles_of_perm(q: torch.Tensor) -> List[List[int]]:
    """
    q is a permutation over {0..n-1}. Returns list of cycles (as lists of ints).
    """
    n = int(q.numel())
    visited = [False] * n
    cycles: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        cur = i
        cyc: List[int] = []
        while not visited[cur]:
            visited[cur] = True
            cyc.append(cur)
            cur = int(q[cur].item())
        cycles.append(cyc)
    return cycles


def cycle_summary(q: torch.Tensor) -> Dict[str, Any]:
    cycs = cycles_of_perm(q)
    lengths = sorted([len(c) for c in cycs], reverse=True)
    hist: Dict[int, int] = {}
    for L in lengths:
        hist[L] = hist.get(L, 0) + 1
    n = int(q.numel())
    return {
        "n": n,
        "num_cycles": len(cycs),
        "cycle_lengths_desc": lengths,
        "cycle_histogram": {str(k): v for k, v in sorted(hist.items())},
        "max_cycle": max(lengths) if lengths else 0,
        "cayley_distance": n - len(cycs),  # minimal #transpositions to reach identity
    }


# -------------------------
# Key reconciliation (MLP: "1,2,3" vs "P1,P2,P3")
# -------------------------
_P_INT = re.compile(r"^P(\d+)$")


def reconcile_keys(
    act: Dict[str, torch.Tensor],
    wgt: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
    """
    Returns possibly-remapped (act, wgt, strategy).
    """
    if set(act).intersection(set(wgt)):
        return act, wgt, "direct"

    # Try mapping activation keys "1" -> "P1" (common for MLP activation matching output)
    act_digits = all(k.isdigit() for k in act.keys())
    wgt_pints = all(_P_INT.match(k) for k in wgt.keys())
    if act_digits and wgt_pints:
        act2 = {f"P{k}": v for k, v in act.items()}
        if set(act2).intersection(set(wgt)):
            return act2, wgt, "act_digit_to_P"

    # Try mapping weight keys "P1" -> "1"
    if wgt_pints:
        wgt2 = {_P_INT.match(k).group(1): v for k, v in wgt.items()}  # type: ignore[union-attr]
        if set(act).intersection(set(wgt2)):
            return act, wgt2, "wgt_P_to_digit"

    return act, wgt, "none"


# -------------------------
# Optional: CKA feature handling
# -------------------------
def activation_to_2d(out: torch.Tensor, unit_dim: int = 1) -> torch.Tensor:
    """
    Same idea as activation_permutation_stitching.activation_to_2d :contentReference[oaicite:3]{index=3}
    """
    if out.ndim < 2:
        raise ValueError(f"Expected activation with ndim>=2, got {tuple(out.shape)}")
    if unit_dim < 0:
        unit_dim = out.ndim + unit_dim
    x = torch.movedim(out, unit_dim, -1)
    return x.reshape(-1, x.shape[-1])


def load_features(path: str) -> Dict[str, torch.Tensor]:
    """
    Expects a dict-like mapping layer_key -> activation tensor.
    Supports .pt/.pkl/.json but typical is torch.save(dict).
    """
    raw = _load_any(path)
    if isinstance(raw, dict) and "features" in raw and isinstance(raw["features"], dict):
        raw = raw["features"]
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict of features in {path}, got {type(raw)}")
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        out[str(k)] = v.detach().cpu()
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-perm", type=str, required=True,
                    help="Path to activation-matching permutations (.json/.pt/.pkl).")
    ap.add_argument("--wgt-perm", type=str, required=True,
                    help="Path to weight-matching permutations (.pkl).")
    ap.add_argument("--out-json", type=str, default=None,
                    help="If set, save a JSON report to this path.")

    # Optional CKA inputs
    ap.add_argument("--features-a", type=str, default=None,
                    help="Optional: path to saved A features dict[layer_key -> tensor].")
    ap.add_argument("--features-b", type=str, default=None,
                    help="Optional: path to saved B features dict[layer_key -> tensor].")
    ap.add_argument("--unit-dim", type=int, default=1,
                    help="Feature dimension corresponding to units/channels (default 1).")

    args = ap.parse_args()

    act = load_permutations(args.act_perm)
    wgt = load_permutations(args.wgt_perm)

    act, wgt, strat = reconcile_keys(act, wgt)

    keys = sorted(set(act).intersection(set(wgt)))
    if not keys:
        raise RuntimeError(
            "No overlapping permutation keys between activation and weight matching.\n"
            f"Reconciliation strategy tried: {strat}\n"
            f"Activation keys (sample): {list(act.keys())[:10]}\n"
            f"Weight keys (sample): {list(wgt.keys())[:10]}"
        )

    report: Dict[str, Any] = {
        "act_perm_path": args.act_perm,
        "wgt_perm_path": args.wgt_perm,
        "key_reconciliation": strat,
        "num_common_keys": len(keys),
        "per_key": {},
    }

    # Optional CKA
    do_cka = (args.features_a is not None) and (args.features_b is not None)
    cka_fn = _load_cka_fn() if do_cka else None
    feats_a = load_features(args.features_a) if do_cka else {}
    feats_b = load_features(args.features_b) if do_cka else {}

    print(f"[INFO] common keys = {len(keys)} (reconcile={strat})")

    for k in keys:
        p_act = act[k]
        p_wgt = wgt[k]

        if not is_valid_permutation(p_act):
            raise ValueError(f"[{k}] activation permutation is not a valid permutation.")
        if not is_valid_permutation(p_wgt):
            raise ValueError(f"[{k}] weight permutation is not a valid permutation.")
        if p_act.numel() != p_wgt.numel():
            raise ValueError(f"[{k}] size mismatch: {p_act.numel()} vs {p_wgt.numel()}")

        # agreement / fixed points
        agreement = float((p_act == p_wgt).float().mean().item())

        # Q = inv(P_wgt) o P_act
        q = compose_q(p_wgt, p_act)
        fixed = float((q == torch.arange(q.numel())).float().mean().item())

        # cycle structure
        cs = cycle_summary(q)

        entry: Dict[str, Any] = {
            "n": int(p_act.numel()),
            "hamming_agreement": agreement,
            "fixed_point_fraction_of_Q": fixed,
            "cycle_summary_of_Q": cs,
        }

        # Optional CKA: compare A vs permuted-B representations under each permutation
        if do_cka:
            if k not in feats_a or k not in feats_b:
                entry["cka"] = {"skipped": True, "reason": "missing features for this key"}
            else:
                xa = activation_to_2d(feats_a[k], unit_dim=args.unit_dim)
                xb = activation_to_2d(feats_b[k], unit_dim=args.unit_dim)

                d = int(p_act.numel())
                if xa.shape[1] != d or xb.shape[1] != d:
                    entry["cka"] = {
                        "skipped": True,
                        "reason": f"feature dim mismatch: xa={tuple(xa.shape)}, xb={tuple(xb.shape)}, perm_dim={d}",
                    }
                else:
                    xb_wgt = xb[:, p_wgt]
                    xb_act = xb[:, p_act]
                    cka_wgt = float(cka_fn(xa, xb_wgt).item())
                    cka_act = float(cka_fn(xa, xb_act).item())
                    entry["cka"] = {
                        "cka_A_vs_Bperm_wgt": cka_wgt,
                        "cka_A_vs_Bperm_act": cka_act,
                        "delta_act_minus_wgt": cka_act - cka_wgt,
                    }

        report["per_key"][k] = entry

        print(
            f"[{k}] n={entry['n']}  agree={agreement:.4f}  fixed(Q)={fixed:.4f}  "
            f"num_cycles={cs['num_cycles']}  max_cycle={cs['max_cycle']}  cayley={cs['cayley_distance']}"
        )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] wrote {args.out_json}")

    # Also print a quick overall summary
    avg_agree = float(np.mean([report["per_key"][k]["hamming_agreement"] for k in keys]))
    avg_fixed = float(np.mean([report["per_key"][k]["fixed_point_fraction_of_Q"] for k in keys]))
    print(f"[SUMMARY] mean agreement={avg_agree:.4f} | mean fixed(Q)={avg_fixed:.4f}")


if __name__ == "__main__":
    main()