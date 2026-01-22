# mutual kNN (CIFAR10 ResNet20)
python measure_alignment_platonic.py \
    --dataset CIFAR10 \
  --seed_a 0 --seed_b 1 \
  --runs_dir ./runs_resnet20_CIFAR10 \
  --which best \
  --split val \
  --metric mutual_knn \
  --topk 10 \
  --pairing diagonal


# NON-DISJOINT

# CKA (CIFAR10 ResNet20)
python measure_alignment_platonic.py \
    --dataset CIFAR10 \
  --seed_a 0 --seed_b 1 \
  --runs_dir ./runs_resnet20_CIFAR10 \
  --which best \
  --split val \
  --metric cka \
  --pairing diagonal

# CKA CIFAR100
python measure_alignment_platonic.py \
    --dataset CIFAR100 \
  --seed_a 0 --seed_b 1 \
  --runs_dir ./runs_resnet20_CIFAR100 \
  --which best \
  --split val \
  --metric cka \
  --pairing diagonal


# DISJOINT CIFAR-10

python measure_alignment_platonic.py \
  --dataset CIFAR10 \
  --runs_dir ./runs_resnet20_CIFAR10_disjoint \
  --disjoint \
  --seed_a 0 --subset_a A \
  --seed_b 0 --subset_b B \
  --which best \
  --split val \
  --metric cka \
  --pairing diagonal


# DISJOINT CIFAR-100

python measure_alignment_platonic.py \
  --dataset CIFAR100 \
  --runs_dir ./runs_resnet20_CIFAR100_disjoint \
  --disjoint \
  --seed_a 0 --subset_a A \
  --seed_b 0 --subset_b B \
  --which best \
  --split val \
  --metric cka \
  --pairing diagonal