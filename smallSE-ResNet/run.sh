# SE-ResNet-50: resnet_50; is_se True; bottleneck
# SE-ResNet-66: resnet_50; is_se True; small_basic_block
SERESNET50=('resnet_50')
BLOCK=('bottleneck' 'small_basic_block')
for m in ${SERESNET50[*]}; do
    for b in ${BLOCK[*]}; do
        for d in ${DATASET[*]}; do
            for((i=1;i<=3;i++)); do
                python3 cifar.py --device 0 --allow_growth --resnet $m --dataset $d --block small_basic_block --is_se
            done
        done
    done
done

# SE-ResNet-18: resnet_18; is_se True; basic_block
# SE-ResNet-26: resnet_18; is_se True; bottleneck
# SE-ResNet-34: resnet_18; is_se True; small_basic_block
SERESNET50=('resnet_18')
BLOCK=('basic_block' 'bottleneck' 'small_basic_block')
for m in ${SERESNET50[*]}; do
    for b in ${BLOCK[*]}; do
        for d in ${DATASET[*]}; do
            for((i=1;i<=3;i++)); do
                python3 cifar.py --device 0 --allow_growth --resnet $m --dataset $d --block small_basic_block --is_se
            done
        done
    done
done


