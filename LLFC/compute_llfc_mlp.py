# LLFC/compute_llfc_mlp_mnist_fmnist.py
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- EDIT THESE IMPORTS IF YOUR PROJECT USES DIFFERENT MODULE PATHS ----
try:
    from architectures import build_model  # e.g. your build_model("mlp", ...)
except Exception:
    build_model = None

try:
    import utils  # for style and any shared helpers you have
except Exception:
    utils = None

from metrics_platonic import cosine_similarity_over_samples, best_scalar_coef

"""
HOW TO RUN:

# MNIST
python LLFC/compute_llfc_mlp_mnist_fmnist.py \
  --dataset mnist \
  --ckpt_a path/to/ckptA.pt \
  --ckpt_b path/to/ckptB.pt \
  --out runs/llfc_mlp_mnist \
  --flatten_input

python LLFC/plot_llfc_mlp_mnist_fmnist.py \
  --results runs/llfc_mlp_mnist/llfc_cos_mnist_mlp.pt \
  --heatmap

# FashionMNIST
python LLFC/compute_llfc_mlp_mnist_fmnist.py \
  --dataset fmnist \
  --ckpt_a path/to/ckptA.pt \
  --ckpt_b path/to/ckptB.pt \
  --out runs/llfc_mlp_fmnist \
  --flatten_input

python LLFC/plot_llfc_mlp_mnist_fmnist.py \
  --results runs/llfc_mlp_fmnist/llfc_cos_fmnist_mlp.pt \
  --heatmap
"""


def load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # common patterns: {"state_dict": ...} or raw state_dict
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        # might already be a state_dict
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def interpolate_state_dict(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor], lam: float) -> Dict[str, torch.Tensor]:
    out = {}
    for k in sd_a.keys():
        va = sd_a[k]
        vb = sd_b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb) and va.shape == vb.shape and va.dtype == vb.dtype:
            out[k] = lam * va + (1.0 - lam) * vb
        else:
            # buffers / non-float items: keep A by default
            out[k] = va
    return out


@dataclass
class HookCollector:
    layer_names: List[str]
    activations: Dict[str, torch.Tensor]
    handles: List[torch.utils.hooks.RemovableHandle]

    @staticmethod
    def for_mlp_linears(model: nn.Module, include_last: bool = True) -> "HookCollector":
        layer_names: List[str] = []
        handles: List[torch.utils.hooks.RemovableHandle] = []
        activations: Dict[str, torch.Tensor] = {}

        # collect linear layers (optionally exclude the last one)
        linear_items: List[Tuple[str, nn.Module]] = [
            (name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)
        ]
        if not include_last and len(linear_items) > 0:
            linear_items = linear_items[:-1]

        def make_hook(nm: str):
            def _hook(_module, _inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                activations[nm] = out.detach()
            return _hook

        for name, m in linear_items:
            layer_names.append(name)
            handles.append(m.register_forward_hook(make_hook(name)))

        return HookCollector(layer_names=layer_names, activations=activations, handles=handles)

    def clear(self) -> None:
        self.activations.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def make_loader(dataset_name: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset_name = dataset_name.lower()
    if dataset_name not in {"mnist", "fmnist"}:
        raise ValueError("dataset must be one of: mnist, fmnist")

    if dataset_name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        ds_cls = datasets.MNIST
    else:
        mean, std = (0.2860,), (0.3530,)
        ds_cls = datasets.FashionMNIST

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = ds_cls(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def forward_collect(model: nn.Module, hooks: HookCollector, x: torch.Tensor, flatten_input: bool) -> Dict[str, torch.Tensor]:
    hooks.clear()
    if flatten_input:
        x = x.view(x.size(0), -1)
    _ = model(x)
    # copy out references (already detached)
    return dict(hooks.activations)


@torch.no_grad()
def compute_llfc_cosine(
    model_a: nn.Module,
    model_b: nn.Module,
    model_lam: nn.Module,
    hooks_a: HookCollector,
    hooks_b: HookCollector,
    hooks_lam: HookCollector,
    loader: DataLoader,
    lambdas: List[float],
    device: str,
    flatten_input: bool,
    max_batches: int,
    eps: float,
) -> Dict[str, torch.Tensor]:
    n_layers = len(hooks_lam.layer_names)
    n_lams = len(lambdas)

    cos_mean = torch.zeros(n_layers, n_lams, dtype=torch.float64)
    cos_std = torch.zeros(n_layers, n_lams, dtype=torch.float64)
    coef_mean = torch.zeros(n_layers, n_lams, dtype=torch.float64)

    for j, lam in enumerate(lambdas):
        # accumulate per-layer over samples
        sum_cos = torch.zeros(n_layers, dtype=torch.float64)
        sum_cos2 = torch.zeros(n_layers, dtype=torch.float64)
        sum_coef = torch.zeros(n_layers, dtype=torch.float64)
        n_total = 0

        for bi, (x, _y) in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break

            x = x.to(device, non_blocking=True)

            feats_a = forward_collect(model_a, hooks_a, x, flatten_input=flatten_input)
            feats_b = forward_collect(model_b, hooks_b, x, flatten_input=flatten_input)
            feats_l = forward_collect(model_lam, hooks_lam, x, flatten_input=flatten_input)

            bs = x.size(0)
            n_total += bs

            for li, lname in enumerate(hooks_lam.layer_names):
                ha = feats_a[lname]
                hb = feats_b[lname]
                hl = feats_l[lname]

                hint = lam * ha + (1.0 - lam) * hb

                cos = cosine_similarity_over_samples(hl, hint, eps=eps).to(torch.float64)  # [B]
                coef = best_scalar_coef(hl, hint, eps=eps).to(torch.float64)  # [B]

                sum_cos[li] += cos.sum()
                sum_cos2[li] += (cos * cos).sum()
                sum_coef[li] += coef.mean() * bs

        mean = sum_cos / max(n_total, 1)
        var = (sum_cos2 / max(n_total, 1)) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))

        cos_mean[:, j] = mean
        cos_std[:, j] = std
        coef_mean[:, j] = sum_coef / max(n_total, 1)

    return {
        "cos_mean": cos_mean,
        "cos_std": cos_std,
        "coef_mean": coef_mean,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["mnist", "fmnist"], required=True)
    p.add_argument("--ckpt_a", type=str, required=True)
    p.add_argument("--ckpt_b", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--arch", type=str, default="mlp")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--lambdas", type=int, default=21, help="Number of lambda points in [0,1]")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test set; else limit batches")
    p.add_argument("--flatten_input", action="store_true", help="Flatten 28x28 -> 784 (recommended for MLP)")
    p.add_argument("--include_last_linear", action="store_true", help="Include last linear layer hooks")
    p.add_argument("--eps", type=float, default=1e-12)

    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if build_model is None:
        raise ImportError("Could not import build_model. Edit the import near the top of this file.")

    # Build models
    model_a = build_model(args.arch)
    model_b = build_model(args.arch)
    model_l = build_model(args.arch)

    sd_a = load_state_dict(args.ckpt_a)
    sd_b = load_state_dict(args.ckpt_b)

    model_a.load_state_dict(sd_a, strict=True)
    model_b.load_state_dict(sd_b, strict=True)

    device = args.device
    model_a.to(device).eval()
    model_b.to(device).eval()
    model_l.to(device).eval()

    loader = make_loader(args.dataset, args.batch_size, args.num_workers)

    # Hooks (MLP: linear layers)
    hooks_a = HookCollector.for_mlp_linears(model_a, include_last=args.include_last_linear)
    hooks_b = HookCollector.for_mlp_linears(model_b, include_last=args.include_last_linear)
    hooks_l = HookCollector.for_mlp_linears(model_l, include_last=args.include_last_linear)

    lambdas = torch.linspace(0.0, 1.0, steps=args.lambdas).tolist()

    # For each lambda, load interpolated weights into model_l
    # (and compute LLFC cosine similarity wrt linear interpolation of features)
    all_cos_mean = []
    all_cos_std = []
    all_coef_mean = []

    for lam in lambdas:
        sd_l = interpolate_state_dict(sd_a, sd_b, lam)
        model_l.load_state_dict(sd_l, strict=True)

    # Now compute across all lambdas (loop is inside compute; we just ensured model_l exists)
    # We reload inside compute_llfc_cosine per lambda by re-calling load_state_dict? No, we do it more cleanly:
    # We'll compute in a single call by reloading model_l each lambda inside a local loop.
    # To keep code simple, we do it here as a loop with a per-lambda compute over the loader.

    # Re-implement per-lambda loop with the same kernel to avoid extra complexity.
    n_layers = len(hooks_l.layer_names)
    n_lams = len(lambdas)
    cos_mean = torch.zeros(n_layers, n_lams, dtype=torch.float64)
    cos_std = torch.zeros(n_layers, n_lams, dtype=torch.float64)
    coef_mean = torch.zeros(n_layers, n_lams, dtype=torch.float64)

    for j, lam in enumerate(lambdas):
        sd_l = interpolate_state_dict(sd_a, sd_b, lam)
        model_l.load_state_dict(sd_l, strict=True)

        res = compute_llfc_cosine(
            model_a=model_a,
            model_b=model_b,
            model_lam=model_l,
            hooks_a=hooks_a,
            hooks_b=hooks_b,
            hooks_lam=hooks_l,
            loader=loader,
            lambdas=[lam],
            device=device,
            flatten_input=args.flatten_input,
            max_batches=args.max_batches,
            eps=args.eps,
        )
        cos_mean[:, j] = res["cos_mean"][:, 0]
        cos_std[:, j] = res["cos_std"][:, 0]
        coef_mean[:, j] = res["coef_mean"][:, 0]

        print(f"[LLFC] dataset={args.dataset} lambda={lam:.3f} done")

    hooks_a.remove()
    hooks_b.remove()
    hooks_l.remove()

    # Aggregate across layers (mean)
    cos_mean_layeravg = cos_mean.mean(dim=0)
    cos_std_layeravg = torch.sqrt(torch.clamp((cos_std ** 2).mean(dim=0), min=0.0))

    save_path = os.path.join(args.out, f"llfc_cos_{args.dataset}_{args.arch}.pt")
    torch.save(
        {
            "dataset": args.dataset,
            "arch": args.arch,
            "ckpt_a": args.ckpt_a,
            "ckpt_b": args.ckpt_b,
            "lambdas": torch.tensor(lambdas, dtype=torch.float64),
            "layers": hooks_l.layer_names,
            "cos_mean": cos_mean,
            "cos_std": cos_std,
            "coef_mean": coef_mean,
            "cos_mean_layeravg": cos_mean_layeravg,
            "cos_std_layeravg": cos_std_layeravg,
        },
        save_path,
    )
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()