import argparse
import os
import pathlib
import time

import numpy as np

from utils.match_utils import find_maximal_match, find_maximal_epsilon

epss = [0.15, 0.20, 0.25, 0.30, 0.50]

def calc_sim(mat0, mat1, epsilon, sample_ndim, sample_iter):
    mms_list = []

    for iter in range(sample_iter):
        sample_idx = np.random.choice(mat0.shape[1], sample_ndim, replace=False)
        X = mat0[:, sample_idx]
        Y = mat1[:, sample_idx]

        idx_X, idx_Y = find_maximal_match(X, Y, epsilon)
        print(idx_X, idx_Y)
        mms = float(len(idx_X) + len(idx_Y)) / (len(X) + len(Y))
        max_epsilon = find_maximal_epsilon(X, Y)

        mms_list.append(mms)
        print('Sampling iter {}: mms = {:.05f}%, max_epsilon = {:.3f}'.format(iter, 100 * mms, max_epsilon))

    return np.mean(mms_list)


def main():
    parser = argparse.ArgumentParser("calc max match similarity of two activation spaces")

    # parser.add_argument("-c", "--conv", help="calc for conv layers (default for fc layers)", action="store_true")
    # parser.add_argument("-x", "--X_path", help="the path to activations X")
    # parser.add_argument("-y", "--Y_path", help="the path to activations Y")
    parser.add_argument("--X_seed")
    parser.add_argument("--Y_seed")
    parser.add_argument("--X_trained", action="store_true")
    parser.add_argument("--Y_trained", action="store_true")
    parser.add_argument("--init-state", action="store_true")
    parser.add_argument("--layer")
    parser.add_argument("--config")
    # parser.add_argument("--eps", type=float)
    parser.add_argument('--ndim', dest='sample_ndim', type=int, default=10000, help="only for feature maps")
    parser.add_argument('--iter', dest='sample_iter', type=int, default=16, help="number of samples")
    args = parser.parse_args()
    print(args)
    sample_ndim = args.sample_ndim
    sample_iter = args.sample_iter

    X_path = args.X_seed + ("_weight_trained" if args.X_trained else "")
    Y_path = args.Y_seed + ("_weight_trained" if args.Y_trained else "")
    state = "initial.state" if args.init_state else "model_best.pth"
    X_path = f"runs/{args.config}/seed_" + X_path + f"/prune_rate=0.7/checkpoints/{state}_activations_{args.layer}.npy"
    Y_path = f"runs/{args.config}/seed_" + Y_path + f"/prune_rate=0.7/checkpoints/{state}_activations_{args.layer}.npy"
    print(X_path, Y_path, sep='\n')
    
    X = np.load(X_path)
    Y = np.load(Y_path)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    if X.shape[2] == 1 and X.shape[3] == 1 and Y.shape[2] == 1 and Y.shape[3] == 1:
        f_X = X.reshape((X.shape[0], X.shape[1]))
        f_X = f_X.transpose()
        f_Y = Y.reshape((Y.shape[0], Y.shape[1]))
        f_Y = f_Y.transpose()
        
    else:
        num_datapoints, channels, h, w = X.shape
        f_X = X.transpose((1,0,2,3)).reshape((channels, num_datapoints*h*w))

        num_datapoints, channels, h, w = Y.shape
        f_Y = Y.transpose((1,0,2,3)).reshape((channels, num_datapoints*h*w))


    print(f_X.shape, f_Y.shape)

    for eps in epss:
        sim = calc_sim(f_X, f_Y, eps, sample_ndim, sample_iter)

        print(f"==> mms = {sim:.05f}")

        write_result_to_csv(
            layer=args.layer,
            sim=sim,
            eps=eps,
            init_state=args.init_state,
            X_trained=args.X_trained,
            Y_trained=args.Y_trained,
            X_seed=args.X_seed,
            Y_seed=args.Y_seed,
            config=args.config
        )

def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "match_results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "layer, "
            "similarity, "
            "eps, "
            "init_state, "
            "X_trained, "
            "Y_trained, "
            "X_seed, "
            "Y_seed, "
            "config\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{layer}, "
                "{sim:.05f}, "
                "{eps}, "
                "{init_state}, "
                "{X_trained}, "
                "{Y_trained}, "
                "{X_seed}, "
                "{Y_seed}, "
                "{config}\n"
            ).format(now=now, **kwargs)
        )

if __name__ == "__main__":
    main()
