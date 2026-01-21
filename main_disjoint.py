import os
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import resnet18_arch
import train_loop
import utils


"""
how to run :
python main_diff_split.py --seeds 7
"""


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


def stratified_train_val_split(targets, val_size: int, seed: int, num_classes: int = 10):
    """
    Deterministic stratified split: val gets ~val_size samples total, balanced across classes.
    Returns: (train_indices, val_indices) indices into the full train dataset.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    all_indices = np.arange(len(targets))
    per_class = {c: all_indices[targets == c].tolist() for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(per_class[c])

    # Allocate val counts per class
    base = val_size // num_classes
    rem = val_size % num_classes
    val_counts = {c: base + (1 if c < rem else 0) for c in range(num_classes)}

    val_indices = []
    train_indices = []
    for c in range(num_classes):
        k = val_counts[c]
        val_indices.extend(per_class[c][:k])
        train_indices.extend(per_class[c][k:])

    rng.shuffle(val_indices)
    rng.shuffle(train_indices)
    return train_indices, val_indices


def split_train_into_two_balanced_subsets(train_indices, targets, seed: int, num_classes: int = 10):
    """
    Given train_indices into the full dataset, return two disjoint balanced subsets A/B.
    Balanced means: each subset has the same number per class, and classes are equally represented.
    Uses all possible samples subject to exact balance constraints.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    train_indices = np.asarray(train_indices)
    per_class = {c: train_indices[targets[train_indices] == c].tolist() for c in range(num_classes)}
    for c in range(num_classes):
        rng.shuffle(per_class[c])

    # To be exactly balanced across classes and between subsets, use:
    # k = min class count in train, then make it even so it can split into 2 equal halves.
    min_count = min(len(per_class[c]) for c in range(num_classes))
    k = (min_count // 2) * 2  # largest even <= min_count
    half = k // 2

    subset_a = []
    subset_b = []
    for c in range(num_classes):
        cls = per_class[c][:k]          # truncate to common even count
        subset_a.extend(cls[:half])
        subset_b.extend(cls[half:])

    rng.shuffle(subset_a)
    rng.shuffle(subset_b)
    return subset_a, subset_b, {"per_class_used": half, "dropped_per_class": {c: len(per_class[c]) - k for c in range(num_classes)}}


def class_counts(indices, targets, num_classes: int = 10):
    targets = np.asarray(targets)
    ctr = Counter(targets[np.asarray(indices)].tolist())
    return [ctr.get(c, 0) for c in range(num_classes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])  # default one run; add more if you want
    parser.add_argument("--split_seed", type=int, default=50)         # fixed train/val split across all runs
    parser.add_argument("--subset_seed", type=int, default=None)      # fixed A/B membership; default = split_seed
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./runs_resnet18_cifar10_two_subsets")
    parser.add_argument("--val_size", type=int, default=5000)
    args = parser.parse_args()

    if args.subset_seed is None:
        args.subset_seed = args.split_seed

    device = utils.get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                             (0.24703223, 0.24348513, 0.26158784)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                             (0.24703223, 0.24348513, 0.26158784)),
    ])

    # Use separate dataset objects so validation has no augmentation
    train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    eval_full  = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=eval_transform)
    test_ds    = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_transform)

    targets = train_full.targets  # length 50k

    # Stratified train/val split (balanced per class)
    train_indices, val_indices = stratified_train_val_split(
        targets=targets,
        val_size=args.val_size,
        seed=args.split_seed,
        num_classes=10
    )

    # Two disjoint balanced subsets of the TRAIN indices
    subset_a_idx, subset_b_idx, subset_meta = split_train_into_two_balanced_subsets(
        train_indices=train_indices,
        targets=targets,
        seed=args.subset_seed,
        num_classes=10
    )

    # Save split indices for reproducibility
    split_path = os.path.join(
        args.out_dir,
        f"indices_splitseed{args.split_seed}_subsetseed{args.subset_seed}_val{args.val_size}.pt"
    )
    if not os.path.exists(split_path):
        torch.save(
            {
                "split_seed": args.split_seed,
                "subset_seed": args.subset_seed,
                "val_size": args.val_size,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "subset_a_indices": subset_a_idx,
                "subset_b_indices": subset_b_idx,
                "subset_meta": subset_meta,
            },
            split_path
        )

    # Build datasets
    subset_a = Subset(train_full, subset_a_idx)
    subset_b = Subset(train_full, subset_b_idx)
    val_ds   = Subset(eval_full, val_indices)

    # Sanity prints (class-balanced?)
    a_counts = class_counts(subset_a_idx, targets)
    b_counts = class_counts(subset_b_idx, targets)
    v_counts = class_counts(val_indices, targets)
    print("Subset A per-class counts:", a_counts)
    print("Subset B per-class counts:", b_counts)
    print("Val     per-class counts:", v_counts)
    print(f"Subset A size: {len(subset_a_idx)} | Subset B size: {len(subset_b_idx)} | Val size: {len(val_indices)}")

    # Test loader never shuffled
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Shared val loader (no shuffle)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for seed in args.seeds:
        print(f"\n==============================\nRunning base seed = {seed}\n==============================")

        run_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Train model on subset A
        modelA_dir = os.path.join(run_dir, "subset_A")
        os.makedirs(modelA_dir, exist_ok=True)

        # Use a deterministic but distinct seed stream per model
        seedA = seed * 1000 + 1
        set_seed(seedA)
        gA = torch.Generator().manual_seed(seedA)

        train_loader_A = torch.utils.data.DataLoader(
            subset_a,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=gA,
            pin_memory=(device.type == "cuda"),
        )

        modelA = resnet18_arch.resnet_18_cifar()
        criterionA = nn.CrossEntropyLoss()
        optimizerA = optim.SGD(modelA.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        historyA = train_loop.train(
            model=modelA,
            criterion=criterionA,
            optimizer=optimizerA,
            train_loader=train_loader_A,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            save_dir=modelA_dir,
            run_name=f"resnet18_cifar10_seed{seed}_subsetA",
            save_every=1,
            save_last=True,
        )
        torch.save({"base_seed": seed, "model_seed": seedA, "subset": "A", "history": historyA},
                   os.path.join(modelA_dir, "history.pt"))
        
        """

        # Train model on subset B
        modelB_dir = os.path.join(run_dir, "subset_B")
        os.makedirs(modelB_dir, exist_ok=True)

        seedB = seed * 1000 + 2
        set_seed(seedB)
        gB = torch.Generator().manual_seed(seedB)

        train_loader_B = torch.utils.data.DataLoader(
            subset_b,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=gB,
            pin_memory=(device.type == "cuda"),
        )

        modelB = resnet_arch.resnet_18_cifar()
        criterionB = nn.CrossEntropyLoss()
        optimizerB = optim.SGD(modelB.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        historyB = train_loop.train(
            model=modelB,
            criterion=criterionB,
            optimizer=optimizerB,
            train_loader=train_loader_B,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            save_dir=modelB_dir,
            run_name=f"resnet18_cifar10_seed{seed}_subsetB",
            save_every=1,
            save_last=True,
        )
        torch.save({"base_seed": seed, "model_seed": seedB, "subset": "B", "history": historyB},
                   os.path.join(modelB_dir, "history.pt"))

        # Optional: evaluate on test (you already have test_loader)
        # Add a test() helper similar to validate() if needed.
        """


if __name__ == "__main__":
    main()
