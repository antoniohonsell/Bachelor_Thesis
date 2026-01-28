# NON - DISJOINT
# CIFAR10
python cifar_resnet20_ln_weight_matching_interp.py \
  --dataset CIFAR10 \
  --ckpt-a ./runs_resnet20_CIFAR10/seed_0/resnet20_CIFAR10_seed0_best.pth \
  --ckpt-b ./runs_resnet20_CIFAR10/seed_1/resnet20_CIFAR10_seed1_best.pth \
  --width-multiplier 1 \
  --out-dir ./weight_matching_out/cifar10_resnet20_seed0_seed1


# CIFAR100
python cifar_resnet20_ln_weight_matching_interp.py \
  --dataset CIFAR100 \
  --ckpt-a ./runs_resnet20_CIFAR100/seed_0/resnet20_CIFAR100_seed0_best.pth \
  --ckpt-b ./runs_resnet20_CIFAR100/seed_1/resnet20_CIFAR100_seed1_best.pth \
  --width-multiplier 1 \
  --out-dir ./weight_matching_out/cifar100_resnet20_seed0_seed1

# DISJOINT SUBSET CIFAR10
python cifar_resnet20_ln_weight_matching_interp.py \
  --dataset CIFAR100 \
  --ckpt-a ./runs_resnet20_CIFAR100_disjoint/seed_0/subset_A/resnet20_CIFAR100_seed0_subsetA_best.pth \
  --ckpt-b ./runs_resnet20_CIFAR100_disjoint/seed_0/subset_B/resnet20_CIFAR100_seed0_subsetB_best.pth \
  --width-multiplier 1 \
  --out-dir ./weight_matching_out/cifar100_resnet20_seed0_subA_subB_disjoint
