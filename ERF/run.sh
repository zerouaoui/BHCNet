# Different ratio R
for((i=1;i<=3;i++)) do
    for((r=1;r<20;r++)) do
        rate=$(echo "scale=2; $r/20"  | bc)
        python3 train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --scheduler step --rate $rate
    done
done

# ERF(-2,2)
for((i=1;i<=3;i++)); do
    python3 train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --scheduler erf --alpha -2 --beta 2
    python3 train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8 --scheduler erf --alpha -2 --beta 2
done

# ERF(-3,3)
for((i=1;i<=3;i++)); do
    python3 train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --scheduler erf --alpha -3 --beta 3
    python3 train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8 --scheduler erf --alpha -3 --beta 3
done

# Cos
for((i=1;i<=3;i++)); do
    python3 train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --scheduler cos
    python3 train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8 --scheduler cos
done

# Exp
for((i=1;i<=3;i++)); do
    python3 train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --scheduler exp
    python3 train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8 --scheduler exp
done
