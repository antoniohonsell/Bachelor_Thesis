import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

import resnet_arch
import train_loop
import utils


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (best-effort; some backends/ops may still be nondeterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Ensures dataloader workers are deterministically seeded
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--split_seed", type=int, default=50)   # fixed split across all runs
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)   # safer on macOS; increase on cluster
    parser.add_argument("--out_dir", type=str, default="./runs_resnet18_cifar10")
    args = parser.parse_args()

    device = utils.get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                             (0.24703223, 0.24348513, 0.26158784)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                             (0.24703223, 0.24348513, 0.26158784)),
    ])

    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    # Fixed train/val split for ALL seeds
    val_size = 5000
    train_size = len(ds) - val_size
    g_split = torch.Generator().manual_seed(args.split_seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g_split)

    # Save split indices once for reproducibility in later analysis
    split_path = os.path.join(args.out_dir, f"split_indices_seed{args.split_seed}.pt")
    if not os.path.exists(split_path):
        torch.save({"train_indices": train_ds.indices, "val_indices": val_ds.indices}, split_path)

    # Test loader never shuffled
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for seed in args.seeds:
        print(f"\n==============================\nRunning seed = {seed}\n==============================")
        set_seed(seed)

        run_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Per-seed generator controls shuffle order deterministically
        g_loader = torch.Generator().manual_seed(seed)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=g_loader,
            pin_memory=(device.type == "cuda"),
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = resnet_arch.resnet_18_cifar()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        history = train_loop.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            save_dir=run_dir,
            run_name=f"resnet18_cifar10_seed{seed}",
            save_every=10,
            save_last=True,
        )

        # Also save a lightweight summary file
        torch.save(
            {"seed": seed, "history": history},
            os.path.join(run_dir, "history.pt"),
        )

        # Optional: evaluate on test after training (you already created test_loader)
        # If you want, you can add a small test() helper similar to validate().


if __name__ == "__main__":
    main()
