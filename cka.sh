GPU=$1
echo "GPU=${GPU}"
shift

SEED_LIST=(2 23 233 13 131)
LAYER_LIST=('convs.1' 'convs.3' 'convs.6' 'convs.8' 'convs.11' 'convs.13' 'linear.1' 'linear.3')

# calc activation
for SEED in ${SEED_LIST[@]}
do
	python3 get_activate.py --config configs/smallscale/conv6/conv6_usc_unsigned.yml \
        --multigpu $GPU \
        --data dataset \
        --prune-rate 0.7 \
        --pretrained runs/conv6_usc_unsigned/seed_${SEED}/prune_rate=0.7/checkpoints/model_best.pth
done

for SEED in ${SEED_LIST[@]}
do
	python3 get_activate.py --config configs/smallscale/conv6/conv6_usc_unsigned.yml \
        --multigpu $GPU \
        --data dataset \
        --prune-rate 0.7 \
        --pretrained runs/conv6_usc_unsigned/seed_${SEED}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth
done


# calc CKA, subnet before and after weight train
for SEED in ${SEED_LIST[@]}
do
    for LAYER in ${LAYER_LIST[@]}
    do 
        python3 calc_CKA.py \
            -x runs/conv6_usc_unsigned/seed_${SEED}/prune_rate=0.7/checkpoints/model_best.pth_activations_${LAYER}.npy \
            -y runs/conv6_usc_unsigned/seed_${SEED}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth_activations_${LAYER}.npy
    done
done

# calc CKA, subnet before weight train
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
            python3 calc_CKA.py \
                -x runs/conv6_usc_unsigned/seed_${SEED_LIST[$i]}/prune_rate=0.7/checkpoints/model_best.pth_activations_${LAYER}.npy \
                -y runs/conv6_usc_unsigned/seed_${SEED_LIST[$j]}/prune_rate=0.7/checkpoints/model_best.pth_activations_${LAYER}.npy
        done
	done
done

# calc CKA, subnet after weight train
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
            python3 calc_CKA.py \
                -x runs/conv6_usc_unsigned/seed_${SEED_LIST[$i]}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth_activations_${LAYER}.npy \
                -y runs/conv6_usc_unsigned/seed_${SEED_LIST[$j]}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth_activations_${LAYER}.npy
        done
	done
done

