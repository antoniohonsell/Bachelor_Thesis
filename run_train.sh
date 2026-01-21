# train ResNet20 on CIFAR-10
python main_non_disjoint.py --dataset CIFAR10 --model resnet20 --epochs 128 --batch_size 256 --lr 0.1 --num_workers 8 --seeds 0 1 --split_seed 50

# train ResNet20 on CIFAR-100
python main_non_disjoint.py --dataset CIFAR100 --model resnet20 --epochs 200 --batch_size 256 --lr 0.1 --num_workers 8 --seeds 0 1 --split_seed 50"