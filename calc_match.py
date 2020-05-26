import argparse
import os
import pathlib
import time

import numpy as np

# _DEBUG = False
_ZERO = 1e-16

def dist2plane(x, Y):
    """
    Calucating the distance of a vector to a plane.
    Assume that the norm of x is large enough
    :param X: [M,]
    :param Y: [N2, D]
    :return: (scalar)
    """
    assert x.ndim == 1
    x_norm = np.linalg.norm(x)
    x = x.reshape(-1, 1)
    Y_t = np.transpose(Y)
    if x_norm < _ZERO:
        return 0.
    solution = np.linalg.lstsq(Y_t, x, rcond=None)
    dist = np.linalg.norm(np.dot(Y_t, solution[0]) - x)
    return dist / x_norm


def dists2plane(X, Y):
    """
    Calucating the distances of a group of vectors to a plane
    Assume norm is large enough
    :param X: [N1, D]
    :param Y: [N2, D]
    :return: [N1,]
    """
    Y_t = np.transpose(Y)
    X_t = np.transpose(X)
    solution = np.linalg.lstsq(Y_t, X_t, rcond=None)
    dist = np.linalg.norm(np.dot(Y_t, solution[0]) - X_t, axis=0)
    norm = np.linalg.norm(X_t, axis=0)
    return dist / norm


def remove_zeros(X):
    """
    Remove zero-norm vectors
    Args:
            X: [N, D]

    Returns:
            non-zero vectors: [N',]
            non-zero indices: [N',]

    """
    assert X.ndim == 2, "Only support 2-D X"
    norm_X = np.linalg.norm(X, axis=1)
    non_zero = np.where(norm_X > _ZERO)[0]
    return X[non_zero], non_zero


def find_maximal_match(X, Y, eps, has_purge=False):
    """
    Find maximal match set between X and Y
    Args:
            X: [N1, D]
            Y: [N2, D]
            eps: scalar
            has_purge: whether X and Y have removed zero vectors

    Returns:
            idx_X: X's match set indices
            idx_Y: Y's match set indices
    """
    assert X.ndim == 2 and Y.ndim == 2, 'Check dimensions of X and Y'
    # if _DEBUG: print('eps={:.4f}'.format(eps))

    if not has_purge:
        X, non_zero_X = remove_zeros(X)
        Y, non_zero_Y = remove_zeros(Y)

    idx_X = np.arange(X.shape[0])
    idx_Y = np.arange(Y.shape[0])

    if len(idx_X) == 0 or len(idx_Y) == 0:
        return idx_X[[]], idx_Y[[]]

    flag = True
    while flag:
        flag = False

        # tic = time.time()
        dist_X = dists2plane(X[idx_X], Y[idx_Y])
        # toc = time.time()
        # print(toc-tic)
        remain_idx_X = idx_X[dist_X <= eps]

        if len(remain_idx_X) < len(idx_X):
            flag = True

        idx_X = remain_idx_X
        if len(idx_X) == 0:
            idx_Y = idx_Y[[]]
            break

        # tic = time.time()
        dist_Y = dists2plane(Y[idx_Y], X[idx_X])
        # toc = time.time()
        # print(toc-tic)
        remain_idx_Y = idx_Y[dist_Y <= eps]

        if len(remain_idx_Y) < len(idx_Y):
            flag = True

        idx_Y = remain_idx_Y
        if len(idx_Y) == 0:
            idx_X = idx_X[[]]
            break

        # if _DEBUG: print('|X|={:d}, |Y|={:d}'.format(len(idx_X), len(idx_Y)))

    if not has_purge:
        idx_X = non_zero_X[idx_X]
        idx_Y = non_zero_Y[idx_Y]

    return idx_X, idx_Y

def calc_sim(X, Y, eps):
    idx_X, idx_Y = find_maximal_match(X, Y, eps)
    return (idx_X.shape[0] + idx_Y.shape[0]) / (X.shape[0] + Y.shape[0])


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
    parser.add_argument("--eps", type=float)
    args = parser.parse_args()
    print(args)

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
    sim = calc_sim(f_X, f_Y, args.eps)

    print(f"{sim:.05f}")

    write_result_to_csv(
        layer=args.layer,
        sim=sim,
        eps=args.eps,
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
