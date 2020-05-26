GPU=$1
echo "GPU=${GPU}"
shift

CONFIG=$1
echo "CONFIG=${CONFIG}"
shift

SEED_LIST=(2 23 233 13 131)

# find subnet
# in runs/${CONFIG}/seed_${SEED}/prune_rate=0.7/checkpoints/model_best.pth
for SEED in ${SEED_LIST[@]}
do
	python3 main.py --config configs/smallscale/conv6/${CONFIG}.yml \
        --multigpu $GPU \
        --data dataset \
        --name seed_${SEED} \
        --prune-rate 0.7 \
        --seed ${SEED} \
        --workers 0
        
done

# train weight
# in runs/conv6_sgd_baseline/seed_${SEED}_train_weight_${CONFIG}/prune_rate=0.0/checkpoints/model_best.pth
for SEED in ${SEED_LIST[@]}
do
	python3 main.py --config configs/smallscale_baselines/dense/conv6/conv6_sgd_baseline.yml \
        --multigpu $GPU \
        --data dataset \
        --name seed_${SEED}_train_weight_${CONFIG} \
        --pretrained runs/${CONFIG}/seed_${SEED}/prune_rate=0.7/checkpoints/initial.state \
        --workers 0
done

# find subnet from weight trained network
# in runs/${CONFIG}/seed_${SEED}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth
for SEED in ${SEED_LIST[@]}
do
	python3 main.py --config configs/smallscale/conv6/${CONFIG}.yml \
        --multigpu $GPU \
        --data dataset \
        --name seed_${SEED}_weight_trained \
        --prune-rate 0.7 \
        --pretrained runs/conv6_sgd_baseline/seed_${SEED}_train_weight_${CONFIG}/prune_rate=0.0/checkpoints/model_best.pth \
        --workers 0
done
