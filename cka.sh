# !!! Change: this is a new file

GPU=$1
echo "GPU=${GPU}"
shift

CONFIG=$1
echo "CONFIG=${CONFIG}"
shift

SEED_LIST=(2 23 233 13 131)
LAYER_LIST=('convs.1' 'convs.3' 'convs.6' 'convs.8' 'convs.11' 'convs.13' 'linear.1' 'linear.3')

# calc activation, subnet of weight untrained
for SEED in ${SEED_LIST[@]}
do
	python3 get_activate.py --config configs/smallscale/conv6/${CONFIG}.yml \
        --multigpu $GPU \
        --data dataset \
        --prune-rate 0.7 \
        --pretrained runs/${CONFIG}/seed_${SEED}/prune_rate=0.7/checkpoints/model_best.pth \
        --workers 0
done

# calc activation, subnet of weight trained
for SEED in ${SEED_LIST[@]}
do
	python3 get_activate.py --config configs/smallscale/conv6/${CONFIG}.yml \
        --multigpu $GPU \
        --data dataset \
        --prune-rate 0.7 \
        --pretrained runs/${CONFIG}/seed_${SEED}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth \
        --workers 0
done

# calc actiavtion, random subnet
for SEED in ${SEED_LIST[@]}
do
	python3 get_activate.py --config configs/smallscale/conv6/${CONFIG}.yml \
        --multigpu $GPU \
        --data dataset \
        --prune-rate 0.7 \
        --pretrained runs/${CONFIG}/seed_${SEED}/prune_rate=0.7/checkpoints/initial.state \
        --workers 0
done

# calc CKA, subnet before and after weight train
for SEED in ${SEED_LIST[@]}
do
    for LAYER in ${LAYER_LIST[@]}
    do 
        python3 calc_CKA.py --X_seed $SEED --Y_seed $SEED --Y_trained --layer $LAYER --config $CONFIG
    done
done

# calc CKA, subnet before weight train
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
            python3 calc_CKA.py --X_seed ${SEED_LIST[$i]} --Y_seed ${SEED_LIST[$j]} --layer $LAYER --config $CONFIG
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
            python3 calc_CKA.py --X_seed ${SEED_LIST[$i]} --Y_seed ${SEED_LIST[$j]} --X_trained --Y_trained --layer $LAYER --config $CONFIG
        done
	done
done

# calc CKA, random subnet
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
            python3 calc_CKA.py --X_seed ${SEED_LIST[$i]} --Y_seed ${SEED_LIST[$j]} --init-state --layer $LAYER --config $CONFIG
        done
	done
done
