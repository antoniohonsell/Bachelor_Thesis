import os
import re
import argparse
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import resnet18_arch  

"""
how to run:

python train_test_loss_comparison.py \
  --out_dir ./runs_resnet18_cifar10_two_subsets \
  --base_seed 7 \
  --batch_size 256 \
  --num_workers 4

"""


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
                return extract_series(obj[nested_key], candidate_keys)

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


def _epoch_from_filename(fname: str) -> Optional[int]:
    """
    Try to parse epoch number from checkpoint filename.
    Handles patterns like:
      epoch_1.pt, epoch-001.pth, ckpt_epoch10.pt, checkpoint_epoch_7.pt, model_ep_3.pt, ...
    """
    base = os.path.basename(fname)

    patterns = [
        r"(?:^|[_\-])epoch[_\-]?(\d+)(?:[_\-]|\.|$)",
        r"(?:^|[_\-])ep[_\-]?(\d+)(?:[_\-]|\.|$)",
        r"(?:^|[_\-])e[_\-]?(\d+)(?:[_\-]|\.|$)",
        r"(?:^|[_\-])checkpoint[_\-]?(\d+)(?:[_\-]|\.|$)",
        r"(?:^|[_\-])ckpt[_\-]?(\d+)(?:[_\-]|\.|$)",
    ]
    for p in patterns:
        m = re.search(p, base, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None

    # fallback: any number in filename (last resort)
    nums = re.findall(r"(\d+)", base)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


def find_epoch_checkpoints(folder: str) -> Dict[int, str]:
    """
    Return a dict: epoch -> checkpoint_path for all .pt/.pth files that appear to be epoch checkpoints.
    """
    ckpts: Dict[int, str] = {}
    if not os.path.isdir(folder):
        return ckpts

    for fn in os.listdir(folder):
        if not (fn.endswith(".pt") or fn.endswith(".pth")):
            continue
        path = os.path.join(folder, fn)
        if os.path.isdir(path):
            continue

        # Skip the history file itself
        if fn == "history.pt":
            continue

        ep = _epoch_from_filename(fn)
        if ep is None:
            continue

        # if duplicates, keep the newest by mtime
        if ep not in ckpts:
            ckpts[ep] = path
        else:
            if os.path.getmtime(path) > os.path.getmtime(ckpts[ep]):
                ckpts[ep] = path

    return ckpts


def load_state_dict_from_checkpoint(ckpt_obj) -> Dict[str, torch.Tensor]:
    """
    Robustly extract a model state_dict from common checkpoint formats.
    """
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        # If it *looks like* a state_dict already
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    # Sometimes checkpoints are saved directly as state_dict
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError(f"Unrecognized checkpoint type: {type(ckpt_obj)}")


@torch.no_grad()
def compute_test_loss_for_checkpoint(
    ckpt_path: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model = resnet18_arch.resnet_18_cifar()
    ckpt = safe_torch_load(ckpt_path)
    sd = load_state_dict_from_checkpoint(ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item())
        total_n += int(y.numel())

    return total_loss / max(total_n, 1)


def make_test_loader(batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                             (0.24703223, 0.24348513, 0.26158784)),
    ])
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=eval_transform
    )
    device_is_cuda = torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Root runs directory (contains seed_X folders).")
    parser.add_argument("--base_seed", type=int, required=True,
                        help="Seed folder to inspect, e.g. 7 for seed_7.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir = os.path.join(args.out_dir, f"seed_{args.base_seed}")
    subA_dir = os.path.join(seed_dir, "subset_A")
    subB_dir = os.path.join(seed_dir, "subset_B")

    hist_a_path = os.path.join(subA_dir, "history.pt")
    hist_b_path = os.path.join(subB_dir, "history.pt")

    if not os.path.exists(hist_a_path):
        raise FileNotFoundError(f"Missing: {hist_a_path}")
    if not os.path.exists(hist_b_path):
        raise FileNotFoundError(f"Missing: {hist_b_path}")

    A = safe_torch_load(hist_a_path)
    B = safe_torch_load(hist_b_path)

    # ---------- Train loss plot (from history) ----------
    train_keys = ["train_loss", "loss_train", "train/loss", "trainLoss", "train_loss_epoch"]
    trainA = extract_series(A, train_keys)
    trainB = extract_series(B, train_keys)

    n_train = max(len(trainA), len(trainB))
    epochs_train = list(range(1, n_train + 1))
    trainA = pad_to_len(trainA, n_train)
    trainB = pad_to_len(trainB, n_train)

    out_train = os.path.join(seed_dir, "train_loss_A_vs_B.png")
    plot_two_curves(
        epochs_train, trainA, trainB,
        "Train loss - Subset A", "Train loss - Subset B",
        f"Train loss: Subset A vs Subset B (seed_{args.base_seed})",
        out_train
    )

    # ---------- Test loss plot (computed from checkpoints) ----------
    test_loader = make_test_loader(args.batch_size, args.num_workers)

    ckptsA = find_epoch_checkpoints(subA_dir)
    ckptsB = find_epoch_checkpoints(subB_dir)

    if len(ckptsA) == 0:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {subA_dir}. "
            f"Your train_loop must save per-epoch checkpoint files with epoch numbers in the filename."
        )
    if len(ckptsB) == 0:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {subB_dir}. "
            f"Your train_loop must save per-epoch checkpoint files with epoch numbers in the filename."
        )

    max_epoch = max(max(ckptsA.keys()), max(ckptsB.keys()))
    epochs_test = list(range(1, max_epoch + 1))

    testA: List[float] = []
    testB: List[float] = []

    for ep in epochs_test:
        if ep in ckptsA:
            testA.append(compute_test_loss_for_checkpoint(ckptsA[ep], test_loader, device))
        else:
            testA.append(float("nan"))

        if ep in ckptsB:
            testB.append(compute_test_loss_for_checkpoint(ckptsB[ep], test_loader, device))
        else:
            testB.append(float("nan"))

        print(f"Epoch {ep:3d} | test loss A: {testA[-1]:.4f} | test loss B: {testB[-1]:.4f}")

    out_test = os.path.join(seed_dir, "test_loss_A_vs_B.png")
    plot_two_curves(
        epochs_test, testA, testB,
        "Test loss - Subset A", "Test loss - Subset B",
        f"Test loss: Subset A vs Subset B (seed_{args.base_seed})",
        out_test
    )

    print("\nSaved plots:")
    print(" -", out_train)
    print(" -", out_test)


if __name__ == "__main__":
    main()
