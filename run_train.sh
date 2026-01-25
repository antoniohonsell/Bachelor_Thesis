# NON DISJOINT

# train ResNet20 on CIFAR-10 on my MAC M4
python main_non_disjoint.py --dataset CIFAR10 --model resnet20 --epochs 128 --batch_size 256 --lr 0.1 --seeds 0 1 --split_seed 50

# train ResNet20 on CIFAR-100 on my MAC M4
python main_non_disjoint.py --dataset CIFAR100 --model resnet20 --epochs 200 --batch_size 256 --lr 0.1 --seeds 0 1 --split_seed 50

# train ResNet20 on CIFAR-10 on cluster machine : NVIDIA H100
python main_non_disjoint.py --dataset CIFAR10 --model resnet20 --epochs 128 --batch_size 256 --lr 0.1 --num_workers 8 --seeds 0 1 --split_seed 50

# train ResNet20 on CIFAR-100 on cluster machine : NVIDIA H100
python main_non_disjoint.py --dataset CIFAR100 --model resnet20 --epochs 200 --batch_size 256 --lr 0.1 --num_workers 8 --seeds 1 --split_seed 50


# DISJOINT subset

# M4
python main_disjoint.py --dataset CIFAR10 --model resnet20 --epochs 128 --batch_size 256 --lr 0.1 --seeds 0 1 --split_seed 50

python main_disjoint.py --dataset CIFAR100 --model resnet20 --epochs 200 --batch_size 256 --lr 0.1 --seeds 0 1 --split_seed 50

#H100
python main_disjoint.py --dataset CIFAR10 --model resnet20 --epochs 128 --batch_size 256 --lr 0.1 --num_workers 8 --seeds 0 --split_seed 50

python main_disjoint.py --dataset CIFAR100 --model resnet20 --epochs 200 --batch_size 256 --lr 0.1 --num_workers 8 --seeds 0 1 --split_seed 50
