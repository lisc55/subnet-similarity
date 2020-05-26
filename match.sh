GPU=$1
echo "GPU=${GPU}"
shift

CONFIG=$1
echo "CONFIG=${CONFIG}"
shift

SEED_LIST=(2 23 233 13 131)
LAYER_LIST=('convs.1' 'convs.3' 'convs.6' 'convs.8' 'convs.11' 'convs.13' 'linear.1' 'linear.3')
# EPS_LIST=(0.15 0.20 0.25 0.30 0.50)

# calc match, subnet before and after weight train
for SEED in ${SEED_LIST[@]}
do
    for LAYER in ${LAYER_LIST[@]}
    do 
		# for EPS in ${EPS_LIST[@]}
		# do
        	python3 calc_match.py --X_seed $SEED --Y_seed $SEED --Y_trained --layer $LAYER --config $CONFIG
		# done
	done
done

# calc match, subnet before weight train
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
			# for EPS in ${EPS_LIST[@]}
			# do
            	python3 calc_match.py --X_seed ${SEED_LIST[$i]} --Y_seed ${SEED_LIST[$j]} --layer $LAYER --config $CONFIG
			# done
        done
	done
done

# calc match, subnet after weight train
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
			# for EPS in ${EPS_LIST[@]}
			# do
            	python3 calc_match.py --X_seed ${SEED_LIST[$i]} --Y_seed ${SEED_LIST[$j]} --X_trained --Y_trained --layer $LAYER --config $CONFIG
			# done
        done
	done
done

# calc match, random subnet
for ((i=0;i<${#SEED_LIST[@]};i++))
do
	for ((j=$i+1;j<${#SEED_LIST[@]};j++))
	do
        for LAYER in ${LAYER_LIST[@]}
        do
			# for EPS in ${EPS_LIST[@]}
			# do
            	python3 calc_match.py --X_seed ${SEED_LIST[$i]} --Y_seed ${SEED_LIST[$j]} --init-state --layer $LAYER --config $CONFIG
			# done
        done
	done
done
