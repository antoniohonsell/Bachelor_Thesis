import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import get_device


# -----------------------------
# Data + model (same as before)
# -----------------------------
def make_strict_dataset(n: int, seed: int = 0, device: str = "cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = (2.0 * torch.rand((n, 2), generator=g, device=device)) - 1.0
    y = ((x[:, 0] < 0.0) & (x[:, 1] > 0.0)).float().unsqueeze(1)
    return x, y


class MLP2x2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 2, bias=True)
        self.fc3 = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits


def set_manual_params(model: nn.Module, params: dict):
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name not in params:
                raise KeyError(f"Missing param: {name}")
            src = torch.tensor(params[name], dtype=tensor.dtype, device=tensor.device)
            if src.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {name}: got {src.shape}, expected {tensor.shape}")
            tensor.copy_(src)


@torch.no_grad()
def accuracy_on(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    probs = torch.sigmoid(model(x))
    preds = (probs >= 0.5).float()
    return (preds.eq(y)).float().mean().item()


def print_weights_and_biases(model: nn.Module, decimals: int = 6):
    with torch.no_grad():
        for name, p in model.named_parameters():
            arr = p.detach().cpu().numpy()
            if arr.ndim == 1:
                s = ", ".join([f"{v:.{decimals}f}" for v in arr.tolist()])
                print(f"{name} = [{s}]")
            else:
                print(f"{name} =")
                for row in arr:
                    s = ", ".join([f"{v:.{decimals}f}" for v in row.tolist()])
                    print(f"  [{s}]")


# -----------------------------------------
# Training with explicit SGD "randomness"
# -----------------------------------------
def train_sgd_seeded(
    *,
    # dataset controls (kept fixed across runs if you want)
    n_samples: int = 8192,
    data_seed: int = 0,
    val_frac: float = 0.2,
    device: str = "cpu",
    # optimization controls
    epochs: int = 300,
    lr: float = 0.1,
    batch_size: int = 256,
    sgd_seed: int = 0,          # controls init + minibatch order + any stochasticity
    # optional "strict" init
    strict_init_params: dict | None = None,
):
    # Fix dataset (same x,y every time for a given data_seed)
    x, y = make_strict_dataset(n_samples, seed=data_seed, device=device)

    # Train/val split (deterministic)
    n_val = int(math.floor(n_samples * val_frac))
    x_val, y_val = x[:n_val], y[:n_val]
    x_tr, y_tr = x[n_val:], y[n_val:]

    # Use a generator so DataLoader shuffling depends on sgd_seed (not global RNG)
    g = torch.Generator(device=device)
    g.manual_seed(sgd_seed)

    loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )

    # Seed global RNG for parameter init (and any other torch randomness)
    torch.manual_seed(sgd_seed)

    model = MLP2x2().to(device)
    if strict_init_params is not None:
        set_manual_params(model, strict_init_params)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "val_acc": []}

    for _ in range(epochs):
        model.train()
        total_loss, total = 0.0, 0

        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total += bs

        hist["train_loss"].append(total_loss / max(1, total))
        hist["val_acc"].append(accuracy_on(model, x_val, y_val))

    return model, hist


def tune_learning_rate_for_seed(
    lr_grid,
    *,
    epochs: int = 300,
    batch_size: int = 256,
    n_samples: int = 8192,
    data_seed: int = 0,
    val_frac: float = 0.2,
    sgd_seed: int = 0,
    device: str = "cpu",
    strict_init_params: dict | None = None,
    criterion: str = "best_val_acc",  # or "best_val_acc_peak"
):
    results = []
    best_lr, best_score, best_model = None, None, None

    for lr in lr_grid:
        model, hist = train_sgd_seeded(
            n_samples=n_samples,
            data_seed=data_seed,
            val_frac=val_frac,
            device=device,
            epochs=epochs,
            lr=float(lr),
            batch_size=batch_size,
            sgd_seed=sgd_seed,
            strict_init_params=strict_init_params,
        )

        final_val = hist["val_acc"][-1]
        peak_val = max(hist["val_acc"])
        final_loss = hist["train_loss"][-1]

        if criterion == "best_val_acc":
            score = final_val
        elif criterion == "best_val_acc_peak":
            score = peak_val
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        results.append(
            {
                "lr": float(lr),
                "final_val_acc": float(final_val),
                "peak_val_acc": float(peak_val),
                "final_train_loss": float(final_loss),
                "sgd_seed": int(sgd_seed),
                "data_seed": int(data_seed),
            }
        )

        print(
            f"[sgd_seed={sgd_seed}] lr={lr:.6g} | final_val_acc={final_val:.4f} "
            f"| peak_val_acc={peak_val:.4f} | final_loss={final_loss:.6f}"
        )

        if best_score is None or score > best_score:
            best_score = score
            best_lr = float(lr)
            best_model = model

    results_sorted = sorted(results, key=lambda d: d["final_val_acc"], reverse=True)
    return best_lr, results_sorted, best_model


# ----------------------------------------------------
# Two disjoint models (different SGD seeds) + LR tuning
# ----------------------------------------------------
def train_two_disjoint_models_with_lr_tuning(
    lr_grid,
    *,
    sgd_seed_a: int,
    sgd_seed_b: int,
    # keep dataset fixed for both models unless you want otherwise
    data_seed: int = 0,
    n_samples: int = 8192,
    val_frac: float = 0.2,
    # training
    epochs: int = 300,
    batch_size: int = 256,
    device: str = "cpu",
    strict_init_params: dict | None = None,
    criterion: str = "best_val_acc",
):
    best_lr_a, res_a, model_a = tune_learning_rate_for_seed(
        lr_grid,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        data_seed=data_seed,
        val_frac=val_frac,
        sgd_seed=sgd_seed_a,
        device=device,
        strict_init_params=strict_init_params,
        criterion=criterion,
    )

    best_lr_b, res_b, model_b = tune_learning_rate_for_seed(
        lr_grid,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        data_seed=data_seed,
        val_frac=val_frac,
        sgd_seed=sgd_seed_b,
        device=device,
        strict_init_params=strict_init_params,
        criterion=criterion,
    )

    summary = {
        "model_a": {"sgd_seed": sgd_seed_a, "best_lr": best_lr_a, "results": res_a},
        "model_b": {"sgd_seed": sgd_seed_b, "best_lr": best_lr_b, "results": res_b},
    }
    return (model_a, summary["model_a"]), (model_b, summary["model_b"]), summary


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    STRICT_INIT = None  

    lr_grid = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

    (model_a, info_a), (model_b, info_b), summary = train_two_disjoint_models_with_lr_tuning(
        lr_grid,
        sgd_seed_a=0,
        sgd_seed_b=1,   
        data_seed=0,      
        n_samples=8192,
        val_frac=0.2,
        epochs=500,
        batch_size=256,
        device="cpu",
        strict_init_params=STRICT_INIT,
        criterion="best_val_acc",
    )

    print("\n=== Best LRs ===")
    print(f"Model A (sgd_seed={info_a['sgd_seed']}): best_lr={info_a['best_lr']}")
    print(f"Model B (sgd_seed={info_b['sgd_seed']}): best_lr={info_b['best_lr']}")

    print("\n=== Model A parameters ===")
    print_weights_and_biases(model_a)

    print("\n=== Model B parameters ===")
    print_weights_and_biases(model_b)