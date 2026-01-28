# weight_matching_torch.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PermutationSpec:
    # perm_name -> list of (state_dict_key, axis_index)
    perm_to_axes: Dict[str, List[Tuple[str, int]]]
    # state_dict_key -> tuple[perm_name or None] of length = tensor.ndim
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]


def permutation_spec_from_axes_to_perm(
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]
) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for k, axis_perms in axes_to_perm.items():
        for axis, p in enumerate(axis_perms):
            if p is not None:
                perm_to_axes[p].append((k, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def resnet20_layernorm_permutation_spec() -> PermutationSpec:
    """
    Permutation spec for your CIFAR ResNet-20 with LayerNorm2d:
      - conv weights: (out_ch, in_ch, k, k)
      - linear weight: (out_features, in_features)
      - LayerNorm2d wraps nn.LayerNorm as `.ln` => params end with `.ln.weight` and `.ln.bias`
    """
    conv_w = lambda k, p_in, p_out: {k: (p_out, p_in, None, None)}
    ln_wb = lambda k, p: {f"{k}.ln.weight": (p,), f"{k}.ln.bias": (p,)}
    linear_wb = lambda k, p_in: {f"{k}.weight": (None, p_in), f"{k}.bias": (None,)}

    axes: Dict[str, Tuple[Optional[str], ...]] = {}

    # Stem
    axes.update(conv_w("conv1.weight", None, "P_bg0"))
    axes.update(ln_wb("norm1", "P_bg0"))

    # Helper for blocks
    def easyblock(layer: int, block: int, p: str) -> Dict[str, Tuple[Optional[str], ...]]:
        inner = f"P_layer{layer}_{block}_inner"
        prefix = f"layer{layer}.{block}"
        d = {}
        d.update(conv_w(f"{prefix}.conv1.weight", p, inner))
        d.update(ln_wb(f"{prefix}.norm1", inner))
        d.update(conv_w(f"{prefix}.conv2.weight", inner, p))
        d.update(ln_wb(f"{prefix}.norm2", p))
        # shortcut is Identity => no params
        return d

    def shortcutblock(layer: int, block: int, p_in: str, p_out: str) -> Dict[str, Tuple[Optional[str], ...]]:
        inner = f"P_layer{layer}_{block}_inner"
        prefix = f"layer{layer}.{block}"
        d = {}
        d.update(conv_w(f"{prefix}.conv1.weight", p_in, inner))
        d.update(ln_wb(f"{prefix}.norm1", inner))
        d.update(conv_w(f"{prefix}.conv2.weight", inner, p_out))
        d.update(ln_wb(f"{prefix}.norm2", p_out))

        # shortcut = nn.Sequential(Conv2d, LayerNorm2d)
        d.update(conv_w(f"{prefix}.shortcut.0.weight", p_in, p_out))
        d.update(ln_wb(f"{prefix}.shortcut.1", p_out))
        return d

    # layer1: 3 easy blocks at P_bg0
    axes.update(easyblock(1, 0, "P_bg0"))
    axes.update(easyblock(1, 1, "P_bg0"))
    axes.update(easyblock(1, 2, "P_bg0"))

    # layer2: first block changes channels: P_bg0 -> P_bg1, then easy blocks at P_bg1
    axes.update(shortcutblock(2, 0, "P_bg0", "P_bg1"))
    axes.update(easyblock(2, 1, "P_bg1"))
    axes.update(easyblock(2, 2, "P_bg1"))

    # layer3: first block changes channels: P_bg1 -> P_bg2, then easy blocks at P_bg2
    axes.update(shortcutblock(3, 0, "P_bg1", "P_bg2"))
    axes.update(easyblock(3, 1, "P_bg2"))
    axes.update(easyblock(3, 2, "P_bg2"))

    # classifier
    axes.update(linear_wb("linear", "P_bg2"))

    return permutation_spec_from_axes_to_perm(axes)


def _index_select(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    # index must be 1D long on same device
    return torch.index_select(x, dim, index)


def get_permuted_param(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    k: str,
    params: Dict[str, torch.Tensor],
    except_axis: Optional[int] = None,
) -> torch.Tensor:
    w = params[k]
    axis_perms = ps.axes_to_perm[k]
    for axis, p in enumerate(axis_perms):
        if axis == except_axis:
            continue
        if p is not None:
            w = _index_select(w, axis, perm[p])
    return w


def apply_permutation(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {}
    for k in params.keys():
        if k not in ps.axes_to_perm:
            # leave untouched if not part of the spec
            out[k] = params[k]
        else:
            out[k] = get_permuted_param(ps, perm, k, params)
    return out


def weight_matching(
    seed: int,
    ps: PermutationSpec,
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
    max_iter: int = 100,
    init_perm: Optional[Dict[str, torch.Tensor]] = None,
    silent: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Find a permutation of params_b to best match params_a (channel-wise) using Hungarian updates.
    """
    # sizes inferred from params_a shapes
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    if init_perm is None:
        perm = {p: torch.arange(n, dtype=torch.long) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.clone().to(dtype=torch.long) for p, v in init_perm.items()}

    perm_names = list(perm.keys())

    # Ensure CPU tensors for matching
    params_a_cpu = {k: v.detach().cpu() for k, v in params_a.items()}
    params_b_cpu = {k: v.detach().cpu() for k, v in params_b.items()}
    perm = {k: v.detach().cpu() for k, v in perm.items()}

    for it in range(max_iter):
        progress = False
        rng = np.random.default_rng(seed + it)
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]

            A = torch.zeros((n, n), dtype=torch.float64)
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a_cpu[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b_cpu, except_axis=axis)

                w_a = torch.movedim(w_a, axis, 0).reshape(n, -1).to(torch.float64)
                w_b = torch.movedim(w_b, axis, 0).reshape(n, -1).to(torch.float64)
                A += w_a @ w_b.T

            A_np = A.numpy()

            # SciPy maximize fallback if needed
            try:
                ri, ci = linear_sum_assignment(A_np, maximize=True)
            except TypeError:
                ri, ci = linear_sum_assignment(-A_np)
            assert np.all(ri == np.arange(len(ri)))

            oldL = float(A_np[np.arange(n), perm[p].numpy()].sum())
            newL = float(A_np[np.arange(n), ci].sum())

            if not silent:
                print(f"{it}/{p}: {newL - oldL:.6e}")

            if newL > oldL + 1e-12:
                progress = True
                perm[p] = torch.tensor(ci, dtype=torch.long)

        if not progress:
            break

    return perm
