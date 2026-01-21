"""
How to run :

python train_loss_comparison.py \
  --out_dir ./runs_resnet18_cifar10_two_subsets \
  --base_seed 7 \
  --mode separate

"""

import os
import argparse
import torch
import matplotlib.pyplot as plt


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_series(obj, candidate_keys):
    """
    Extract a per-epoch series from common history formats:
      - dict with list values: {"train_loss":[...], ...}
      - nested dict: {"history": {...}} or {"history": [...]}
      - list of dict per epoch: [{"train_loss":...}, ...]
    Returns: list[float]
    """
    if isinstance(obj, dict):
        # direct
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], (list, tuple)):
                return [float(x) for x in obj[k]]

        # nested
        for nested_key in ["history", "train", "logs", "metrics"]:
            if nested_key in obj:
                try:
                    return extract_series(obj[nested_key], candidate_keys)
                except KeyError:
                    pass

    # list of per-epoch dicts
    if isinstance(obj, (list, tuple)):
        series = []
        found_any = False
        for item in obj:
            if isinstance(item, dict):
                val = None
                for k in candidate_keys:
                    if k in item:
                        val = item[k]
                        break
                series.append(float("nan") if val is None else float(val))
                found_any = found_any or (val is not None)
            else:
                series.append(float("nan"))
        if found_any:
            return series

    raise KeyError(f"None of keys {candidate_keys} found in object type {type(obj)}")


def pad_to_len(xs, n):
    xs = list(xs)
    if len(xs) < n:
        xs = xs + [float("nan")] * (n - len(xs))
    return xs


def plot_two_curves(epochs, y1, y2, label1, label2, title, out_path):
    plt.figure()
    plt.plot(epochs, y1, label=label1)
    plt.plot(epochs, y2, label=label2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_four_curves(epochs, trainA, trainB, valA, valB, title, out_path):
    plt.figure()
    plt.plot(epochs, trainA, label="Train loss - Subset A")
    plt.plot(epochs, trainB, label="Train loss - Subset B")
    plt.plot(epochs, valA,   label="Val loss - Subset A")
    plt.plot(epochs, valB,   label="Val loss - Subset B")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Root runs directory (contains seed_X folders).")
    parser.add_argument("--base_seed", type=int, required=True,
                        help="Seed folder to inspect, e.g. 7 for seed_7.")
    parser.add_argument("--mode", choices=["separate", "combined"], default="separate",
                        help="separate: 2 plots (train and val). combined: 1 plot with 4 curves.")
    args = parser.parse_args()

    seed_dir = os.path.join(args.out_dir, f"seed_{args.base_seed}")
    hist_a_path = os.path.join(seed_dir, "subset_A", "history.pt")
    hist_b_path = os.path.join(seed_dir, "subset_B", "history.pt")

    if not os.path.exists(hist_a_path):
        raise FileNotFoundError(f"Missing: {hist_a_path}")
    if not os.path.exists(hist_b_path):
        raise FileNotFoundError(f"Missing: {hist_b_path}")

    A = safe_torch_load(hist_a_path)
    B = safe_torch_load(hist_b_path)

    # Keys: adjust here if your train_loop uses different names
    train_keys = ["train_loss", "loss_train", "train/loss", "trainLoss", "train_loss_epoch"]
    val_keys   = ["val_loss", "loss_val", "valid_loss", "val/loss", "validation_loss"]

    trainA = extract_series(A, train_keys)
    trainB = extract_series(B, train_keys)
    valA   = extract_series(A, val_keys)
    valB   = extract_series(B, val_keys)

    # Align lengths (in case one history is shorter)
    n = max(len(trainA), len(trainB), len(valA), len(valB))
    epochs = list(range(1, n + 1))
    trainA = pad_to_len(trainA, n)
    trainB = pad_to_len(trainB, n)
    valA   = pad_to_len(valA, n)
    valB   = pad_to_len(valB, n)

    if args.mode == "separate":
        out_train = os.path.join(seed_dir, "train_loss_A_vs_B.png")
        out_val   = os.path.join(seed_dir, "val_loss_A_vs_B.png")

        plot_two_curves(
            epochs, trainA, trainB,
            "Train loss - Subset A", "Train loss - Subset B",
            f"Train loss: Subset A vs Subset B (seed_{args.base_seed})",
            out_train
        )
        plot_two_curves(
            epochs, valA, valB,
            "Val loss - Subset A", "Val loss - Subset B",
            f"Validation loss: Subset A vs Subset B (seed_{args.base_seed})",
            out_val
        )

        print("Saved plots:")
        print(" -", out_train)
        print(" -", out_val)

    else:
        out_combined = os.path.join(seed_dir, "loss_A_vs_B_train_and_val.png")
        plot_four_curves(
            epochs, trainA, trainB, valA, valB,
            f"Train/Val loss: Subset A vs Subset B (seed_{args.base_seed})",
            out_combined
        )
        print("Saved plot:")
        print(" -", out_combined)


if __name__ == "__main__":
    main()
