GPU=$1
echo "GPU=${GPU}"
shift

SEED_LIST=(2 23 233 13 131)

# find subnet
# in runs/conv6_usc_unsigned/seed_${SEED}/prune_rate=0.7/checkpoints/model_best.pth
for SEED in ${SEED_LIST[@]}
do
	python3 main.py --config configs/smallscale/conv6/conv6_usc_unsigned.yml \
        --multigpu $GPU \
        --data dataset \
        --name seed_${SEED} \
        --prune-rate 0.7 \
        --seed ${SEED} \
        --workers 0
        
done

# train weight
# in runs/conv6_sgd_baseline/seed_${SEED}_train_weight/prune_rate=0.0/checkpoints/model_best.pth
for SEED in ${SEED_LIST[@]}
do
	python3 main.py --config configs/smallscale_baselines/dense/conv6/conv6_sgd_baseline.yml \
        --multigpu $GPU \
        --data dataset \
        --name seed_${SEED}_train_weight \
        --pretrained runs/conv6_usc_unsigned/seed_${SEED}/prune_rate=0.7/checkpoints/initial.state \
        --workers 0
done

# find subnet from weight trained network
# in runs/conv6_usc_unsigned/seed_${SEED}_weight_trained/prune_rate=0.7/checkpoints/model_best.pth
for SEED in ${SEED_LIST[@]}
do
	python3 main.py --config configs/smallscale/conv6/conv6_usc_unsigned.yml \
        --multigpu $GPU \
        --data dataset \
        --name seed_${SEED}_weight_trained \
        --prune-rate 0.7 \
        --pretrained runs/conv6_sgd_baseline/seed_${SEED}_train_weight/prune_rate=0.0/checkpoints/model_best.pth \
        --workers 0
done
