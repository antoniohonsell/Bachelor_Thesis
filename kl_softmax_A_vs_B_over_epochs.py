import os
import re
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import resnet18_arch
import utils

"""
how to run:

python kl_softmax_A_vs_B_over_epochs.py \
  --out_dir ./runs_resnet18_cifar10_two_subsets \
  --base_seed 7 \
  --direction "a||b" \
  --save_per_epoch \
  --plot


"""


# ---------- helpers: checkpoint discovery & loading ----------

EPOCH_PATTERNS = [
    re.compile(r"(?:epoch|ep|e)[_\-]?(\d+)", re.IGNORECASE),  # epoch_10, ep-10, e10
    re.compile(r"(\d+)(?:st|nd|rd|th)?(?:_epoch)?", re.IGNORECASE),  # fallback-ish
]


EPOCH_RE = re.compile(r"_epoch(\d+)\.(?:pt|pth)$", re.IGNORECASE)

def parse_epoch_from_filename(name: str):
    m = EPOCH_RE.search(name)
    return int(m.group(1)) if m else None



def find_epoch_checkpoints(dir_path: str):
    """
    Returns dict: epoch(int) -> filepath(str) for checkpoint-like files in dir_path.
    If multiple files map to same epoch, keep the most recently modified.
    """
    candidates = []
    for fn in os.listdir(dir_path):
        if not (fn.endswith(".pt") or fn.endswith(".pth")):
            continue
        if fn == "history.pt":
            continue
        epoch = parse_epoch_from_filename(fn)
        if epoch is None:
            continue
        fp = os.path.join(dir_path, fn)
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = 0.0
        candidates.append((epoch, mtime, fp))

    epoch2fp = {}
    for epoch, mtime, fp in sorted(candidates, key=lambda x: (x[0], x[1])):
        # Keep latest mtime per epoch
        if (epoch not in epoch2fp) or (mtime >= os.path.getmtime(epoch2fp[epoch])):
            epoch2fp[epoch] = fp
    return epoch2fp


def load_state_dict_flex(checkpoint_path: str):
    """
    Load a checkpoint and return a state_dict.
    Handles common formats:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - {"model": ...}
      - {"net": ...}
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        # could already be a state_dict
        # (heuristic: contains tensor values)
        tensor_vals = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_vals) > 0:
            return ckpt

    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


# ---------- helpers: KL computation ----------

@torch.no_grad()
def kl_per_sample_from_logits(logits_a, logits_b, direction="a||b"):
    """
    Compute per-sample KL divergence between softmax distributions.
    - direction="a||b" computes KL(pA || pB)
    - direction="b||a" computes KL(pB || pA)
    - direction="sym" computes 0.5*(KL(pA||pB)+KL(pB||pA))
    Returns: tensor [batch]
    """
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa = log_pa.exp()
    pb = log_pb.exp()

    if direction == "a||b":
        # KL(pA || pB) = sum pA * (log pA - log pB)
        # F.kl_div(input=log_pB, target=pA) gives KL(pA||pB)
        kl = F.kl_div(log_pb, pa, reduction="none").sum(dim=1)
        return kl

    if direction == "b||a":
        kl = F.kl_div(log_pa, pb, reduction="none").sum(dim=1)
        return kl

    if direction == "sym":
        kl_ab = F.kl_div(log_pb, pa, reduction="none").sum(dim=1)
        kl_ba = F.kl_div(log_pa, pb, reduction="none").sum(dim=1)
        return 0.5 * (kl_ab + kl_ba)

    raise ValueError("direction must be one of: a||b, b||a, sym")


@torch.no_grad()
def eval_epoch_kl(modelA, modelB, loader, device, direction="a||b"):
    """
    Returns: kl_all tensor [N_test] in test-set order.
    """
    modelA.eval()
    modelB.eval()

    kl_chunks = []
    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        logits_a = modelA(x)
        logits_b = modelB(x)
        kl = kl_per_sample_from_logits(logits_a, logits_b, direction=direction)
        kl_chunks.append(kl.detach().cpu())
    return torch.cat(kl_chunks, dim=0)


def summary_stats(x: torch.Tensor):
    """
    x: 1D tensor
    Returns dict of summary stats (floats).
    """
    x = x.float()
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "p95": float(x.kthvalue(int(0.95 * (x.numel() - 1)) + 1).values.item()),
        "max": float(x.max().item()),
        "min": float(x.min().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Root runs directory, e.g. ./runs_resnet18_cifar10_two_subsets")
    parser.add_argument("--base_seed", type=int, required=True,
                        help="Which seed folder, e.g. 7 for seed_7")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--direction", type=str, default="a||b", choices=["a||b", "b||a", "sym"],
                        help="Which KL to compute: KL(pA||pB), KL(pB||pA), or symmetric.")
    parser.add_argument("--save_per_epoch", action="store_true",
                        help="Save per-sample KL tensors for each epoch to disk (recommended).")
    parser.add_argument("--plot", action="store_true",
                        help="Save a plot of mean KL vs epoch.")
    args = parser.parse_args()

    device = utils.get_device()

    seed_dir = os.path.join(args.out_dir, f"seed_{args.base_seed}")
    dirA = os.path.join(seed_dir, "subset_A")
    dirB = os.path.join(seed_dir, "subset_B")

    if not os.path.isdir(dirA):
        raise FileNotFoundError(f"Missing directory: {dirA}")
    if not os.path.isdir(dirB):
        raise FileNotFoundError(f"Missing directory: {dirB}")

    ckptsA = find_epoch_checkpoints(dirA)
    ckptsB = find_epoch_checkpoints(dirB)

    common_epochs = sorted(set(ckptsA.keys()) & set(ckptsB.keys()))
    if len(common_epochs) == 0:
        raise RuntimeError(
            "No common epoch checkpoints found between subset_A and subset_B.\n"
            f"Found epochs A: {sorted(ckptsA.keys())}\n"
            f"Found epochs B: {sorted(ckptsB.keys())}\n"
            "If your checkpoints don't include epoch numbers in filenames, "
            "you'll need to adjust parse_epoch_from_filename()."
        )

    # Test set (fixed order)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                             (0.24703223, 0.24348513, 0.26158784)),
    ])
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Output dir for KL artifacts
    kl_dir = os.path.join(seed_dir, f"kl_{args.direction.replace('|','').replace('||','_')}")
    os.makedirs(kl_dir, exist_ok=True)

    # For plotting
    epochs_list = []
    mean_list = []
    median_list = []
    p95_list = []

    print(f"Computing KL divergence direction={args.direction} on CIFAR-10 test set")
    print(f"Common epochs: {common_epochs}")
    print("epoch\tmean\tmedian\tp95\tmax\tmin")

    for epoch in common_epochs:
        # Load models
        modelA = resnet18_arch.resnet_18_cifar()
        modelB = resnet18_arch.resnet_18_cifar()

        sdA = load_state_dict_flex(ckptsA[epoch])
        sdB = load_state_dict_flex(ckptsB[epoch])

        modelA.load_state_dict(sdA, strict=True)
        modelB.load_state_dict(sdB, strict=True)

        modelA.to(device)
        modelB.to(device)

        # Evaluate KL per sample
        kl_all = eval_epoch_kl(modelA, modelB, test_loader, device, direction=args.direction)
        stats = summary_stats(kl_all)

        print(f"{epoch}\t{stats['mean']:.6f}\t{stats['median']:.6f}\t{stats['p95']:.6f}\t{stats['max']:.6f}\t{stats['min']:.6f}")

        epochs_list.append(epoch)
        mean_list.append(stats["mean"])
        median_list.append(stats["median"])
        p95_list.append(stats["p95"])

        if args.save_per_epoch:
            out_path = os.path.join(kl_dir, f"kl_test_per_sample_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "direction": args.direction,
                    "kl_per_sample": kl_all,   # tensor [10000] in CIFAR-10 test order
                    "stats": stats,
                    "ckptA": ckptsA[epoch],
                    "ckptB": ckptsB[epoch],
                },
                out_path
            )

        # free GPU memory between epochs (useful if you have limited VRAM)
        del modelA, modelB
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(epochs_list, mean_list, label="mean KL")
        plt.plot(epochs_list, median_list, label="median KL")
        plt.plot(epochs_list, p95_list, label="p95 KL")
        plt.xlabel("Epoch")
        plt.ylabel("KL divergence (per sample)")
        plt.title(f"KL({args.direction}) between model A and B softmax on test set )")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(kl_dir, "kl_summary_vs_epoch.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"\nSaved plot: {plot_path}")

    print(f"\nArtifacts directory: {kl_dir}")
    if not args.save_per_epoch:
        print("Tip: add --save_per_epoch to save the per-sample KL vectors for downstream analysis.")


if __name__ == "__main__":
    main()
