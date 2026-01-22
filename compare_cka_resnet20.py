# compare_cka_resnet20.py
#
# Compares the 4 ResNet20 CKA results saved by measure_alignment_platonic.py
# (expects .npz files containing: scores, layers, split, metric, pairing, ckpt_a, ckpt_b)


# USAGE :
"""
python compare_cka_resnet20.py --dir ./results/alignment_resnet --save ./results/alignment_resnet/cka_resnet20_compare.png
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def _as_py(x):
    """Convert np scalar/bytes -> python type."""
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    return x


def load_result(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)

    scores = np.array(d["scores"], dtype=float)
    layers = np.array(d["layers"]).astype(str).tolist()

    meta = {
        "split": str(_as_py(d.get("split", "unknown"))),
        "metric": str(_as_py(d.get("metric", "unknown"))),
        "pairing": str(_as_py(d.get("pairing", "unknown"))),
        "ckpt_a": str(_as_py(d.get("ckpt_a", ""))),
        "ckpt_b": str(_as_py(d.get("ckpt_b", ""))),
    }

    return {"path": npz_path, "scores": scores, "layers": layers, "meta": meta}


def infer_label_from_filename(npz_path: str) -> str:
    name = os.path.basename(npz_path).replace(".npz", "")

    # format: base_a__vs__base_b__val__cka__diagonal
    if "__vs__" not in name:
        return name

    left, rest = name.split("__vs__", 1)
    parts = rest.split("__")
    base_b = parts[0] if len(parts) > 0 else "?"
    split = parts[1] if len(parts) > 1 else "?"
    metric = parts[2] if len(parts) > 2 else "?"
    pairing = parts[3] if len(parts) > 3 else "?"

    # dataset
    if ("CIFAR100" in left) or ("CIFAR100" in base_b):
        dataset = "CIFAR100"
    elif ("CIFAR10" in left) or ("CIFAR10" in base_b):
        dataset = "CIFAR10"
    else:
        dataset = "UNKNOWN"

    # disjoint?
    disjoint = ("subsetA" in left) or ("subsetB" in left) or ("subsetA" in base_b) or ("subsetB" in base_b)

    # seeds/subsets
    def find_seed(s):
        m = re.search(r"seed(\d+)", s)
        return m.group(1) if m else "?"

    def find_subset(s):
        m = re.search(r"subset([AB])", s)
        return m.group(1) if m else None

    seed_l = find_seed(left)
    seed_r = find_seed(base_b)
    sub_l = find_subset(left)
    sub_r = find_subset(base_b)

    if disjoint:
        detail = f"seed{seed_l}: {sub_l} vs {sub_r}"
        mode = "disjoint"
    else:
        detail = f"seed{seed_l} vs seed{seed_r}"
        mode = "non-disjoint"

    return f"{dataset} | {mode} | {detail} | {split}/{metric}/{pairing}"


def align_scores_to_layers(ref_layers, layers, scores):
    """If layer lists differ, align by layer name intersection."""
    if layers == ref_layers:
        return scores

    m = {ln: sc for ln, sc in zip(layers, scores)}
    out = []
    for ln in ref_layers:
        if ln not in m:
            raise ValueError(f"Layer '{ln}' missing from one file. Have: {layers}")
        out.append(m[ln])
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="./results/alignment_resnet",
                    help="Folder containing the saved .npz alignment results")
    ap.add_argument("--files", nargs="*", default=None,
                    help="Optional explicit list of 4 .npz files (overrides auto-discovery)")
    ap.add_argument("--save", type=str, default=None,
                    help="If set, save the plot to this path (e.g., ./cka_compare.png)")
    args = ap.parse_args()

    if args.files and len(args.files) > 0:
        paths = args.files
    else:
        # auto-discover *resnet20* CKA diagonal results
        pat = os.path.join(args.dir, "resnet20_*__cka__diagonal.npz")
        paths = sorted(glob.glob(pat))

    if len(paths) != 4:
        raise SystemExit(
            f"Expected 4 files, found {len(paths)}.\n"
            f"Auto pattern: {os.path.join(args.dir, 'resnet20_*__cka__diagonal.npz')}\n"
            f"Tip: pass --files <f1> <f2> <f3> <f4>"
        )

    results = [load_result(p) for p in paths]

    # Validate metadata
    for r in results:
        split = r["meta"]["split"]
        metric = r["meta"]["metric"]
        if "cka" not in metric:
            raise ValueError(f"{r['path']} has metric={metric}, expected cka-like.")
        if split not in ("train", "val", "test", "unknown"):
            raise ValueError(f"{r['path']} has split={split} (unexpected).")

    # Use first file's layer order as reference
    ref_layers = results[0]["layers"]
    for r in results:
        r["scores"] = align_scores_to_layers(ref_layers, r["layers"], r["scores"])
        r["layers"] = ref_layers

    # Print a compact summary
    print("CKA comparison (mean over layers):")
    for r in results:
        label = infer_label_from_filename(r["path"])
        print(f"  {label}")
        print(f"    mean={r['scores'].mean():.6f} | per-layer={np.round(r['scores'], 6)}")

    # Plot
    x = np.arange(len(ref_layers))
    plt.figure(figsize=(10, 4.5))

    # Order: CIFAR10 non-disjoint, CIFAR10 disjoint, CIFAR100 non-disjoint, CIFAR100 disjoint (if present)
    def sort_key(p):
        s = os.path.basename(p)
        ds = 0 if "CIFAR10" in s else 1
        dj = 1 if "subset" in s else 0
        return (ds, dj, s)

    results_sorted = sorted(results, key=lambda r: sort_key(r["path"]))

    for r in results_sorted:
        label = infer_label_from_filename(r["path"])
        label = f"{label} | mean={r['scores'].mean():.3f}"
        plt.plot(x, r["scores"], marker="o", linewidth=2, label=label)

    plt.xticks(x, ref_layers, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer")
    plt.ylabel("CKA")
    plt.title("ResNet20 CKA comparison (diagonal pairing)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.tight_layout()

    if args.save is not None:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"\nSaved figure to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
