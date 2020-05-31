# !!! Change: this is a new file

import torch
import sys
import numpy as np
import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument("--X_seed", help="2, 13, 23, 131, 233")
# #parser.add_argument("--Y_seed")
# parser.add_argument("--X_trained", action="store_true")
# #parser.add_argument("--Y_trained", action="store_true")
# # parser.add_argument("--init_state", action="store_true")
# parser.add_argument("--config")
# args = parser.parse_args()

def get_topk_tensor(scores, k):
    '''
    score: 4 dim numpy array
    '''
    scores_size = scores.shape
    scores_flatten = scores.flatten()
    scores_topk = np.argpartition(scores_flatten, k)
    # print(scores_topk)
    ones_topk = np.zeros_like(scores)
    for i in range(k):
        idx = scores_topk[i]
        id4 = idx % scores_size[3]
        idx = idx // scores_size[3]
        id3 = idx % scores_size[2]
        idx = idx // scores_size[2]
        id2 = idx % scores_size[1]
        idx = idx // scores_size[1]
        id1 = idx % scores_size[0]
        idx = idx // scores_size[0]
        ones_topk[id1][id2][id3][id4] = 1
    return ones_topk

def get_data(X_seed, X_trained, config, init_state):
    X_filename = "seed_" + X_seed + ("_weight_trained/" if X_trained else "/")
    X_path = "./runs/conv6_usc_unsigned/" if config == "conv6_usc_unsigned" else "./runs_ukn/"
    X_path = X_path + X_filename + "prune_rate=0.7/checkpoints/"
    filename = "model_best.pth" if init_state else 'initial.state'
    output_name = 'm.txt' if init_state else 'i.txt'
    model = torch.load(X_path + filename, map_location=torch.device('cpu'))
    d = model['state_dict']

    k = 0.7
    scores_0_topk = get_topk_tensor(d['module.convs.0.scores'].numpy(), int(k * d['module.convs.0.scores'].flatten().numpy().shape[0]))
    scores_2_topk = get_topk_tensor(d['module.convs.2.scores'].numpy(), int(k * d['module.convs.2.scores'].flatten().numpy().shape[0]))
    scores_5_topk = get_topk_tensor(d['module.convs.5.scores'].numpy(), int(k * d['module.convs.5.scores'].flatten().numpy().shape[0]))
    scores_7_topk = get_topk_tensor(d['module.convs.7.scores'].numpy(), int(k * d['module.convs.7.scores'].flatten().numpy().shape[0]))
    scores_10_topk = get_topk_tensor(d['module.convs.10.scores'].numpy(), int(k * d['module.convs.10.scores'].flatten().numpy().shape[0]))
    scores_12_topk = get_topk_tensor(d['module.convs.12.scores'].numpy(), int(k * d['module.convs.12.scores'].flatten().numpy().shape[0]))
    scores_l0_topk = get_topk_tensor(d['module.linear.0.scores'].numpy(), int(k * d['module.linear.0.scores'].flatten().numpy().shape[0]))
    scores_l2_topk = get_topk_tensor(d['module.linear.2.scores'].numpy(), int(k * d['module.linear.2.scores'].flatten().numpy().shape[0]))
    scores_l4_topk = get_topk_tensor(d['module.linear.4.scores'].numpy(), int(k * d['module.linear.4.scores'].flatten().numpy().shape[0]))


    saved_stdout = sys.stdout
    with open(X_path + output_name, 'w') as file:
        sys.stdout = file
        print(scores_0_topk.shape[0])
        print(scores_0_topk.shape[1])
        print(scores_0_topk.shape[2])
        print(scores_0_topk.shape[3])
        s0_f = scores_0_topk.flatten()
        for i in range(s0_f.shape[0]):
            print(s0_f[i], end=' ')
        print('')

        print(scores_2_topk.shape[0])
        print(scores_2_topk.shape[1])
        print(scores_2_topk.shape[2])
        print(scores_2_topk.shape[3])
        s2_f = scores_2_topk.flatten()
        for i in range(s2_f.shape[0]):
            print(s2_f[i], end=' ')
        print('')

        print(scores_5_topk.shape[0])
        print(scores_5_topk.shape[1])
        print(scores_5_topk.shape[2])
        print(scores_5_topk.shape[3])
        s5_f = scores_5_topk.flatten()
        for i in range(s5_f.shape[0]):
            print(s5_f[i], end=' ')
        print('')

        print(scores_7_topk.shape[0])
        print(scores_7_topk.shape[1])
        print(scores_7_topk.shape[2])
        print(scores_7_topk.shape[3])
        s7_f = scores_7_topk.flatten()
        for i in range(s7_f.shape[0]):
            print(s7_f[i], end=' ')
        print('')

        print(scores_10_topk.shape[0])
        print(scores_10_topk.shape[1])
        print(scores_10_topk.shape[2])
        print(scores_10_topk.shape[3])
        s10_f = scores_10_topk.flatten()
        for i in range(s10_f.shape[0]):
            print(s10_f[i], end=' ')
        print('')

        print(scores_12_topk.shape[0])
        print(scores_12_topk.shape[1])
        print(scores_12_topk.shape[2])
        print(scores_12_topk.shape[3])
        s12_f = scores_12_topk.flatten()
        for i in range(s12_f.shape[0]):
            print(s12_f[i], end=' ')
        print('')

        print(scores_l0_topk.shape[0])
        print(scores_l0_topk.shape[1])
        print(scores_l0_topk.shape[2])
        print(scores_l0_topk.shape[3])
        sl0_f = scores_l0_topk.flatten()
        for i in range(sl0_f.shape[0]):
            print(sl0_f[i], end=' ')
        print('')

        print(scores_l2_topk.shape[0])
        print(scores_l2_topk.shape[1])
        print(scores_l2_topk.shape[2])
        print(scores_l2_topk.shape[3])
        sl2_f = scores_l2_topk.flatten()
        for i in range(sl2_f.shape[0]):
            print(sl2_f[i], end=' ')
        print('')

        print(scores_l4_topk.shape[0])
        print(scores_l4_topk.shape[1])
        print(scores_l4_topk.shape[2])
        print(scores_l4_topk.shape[3])
        sl4_f = scores_l4_topk.flatten()
        for i in range(sl4_f.shape[0]):
            print(sl4_f[i], end=' ')
        print('')
        sys.stdout = saved_stdout
    print(X_path + filename)
    # print(model.keys())
    # print(model['state_dict'].keys())
    # print(model['state_dict']['module.convs.10.scores'].size())
    # print(model['state_dict']['module.convs.12.scores'].size())
    # print(model['state_dict']['module.linear.0.scores'].size())
    # print(model['state_dict']['module.linear.2.scores'].size())
    # print(model['state_dict']['module.linear.4.scores'].size())
seed_list = ['2', '13', '23', '131', '233']
is_trained_list = [True, False]
init_state_list = [True, False]
config_list = ['conv6_usc_unsigned', 'conv6_ukn_unsigned']

for config in config_list:
    for is_trained in is_trained_list:
        for seed in seed_list:
            for init_state in init_state_list:
                get_data(seed, is_trained, config, init_state)