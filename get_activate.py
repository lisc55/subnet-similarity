import os
import pathlib
import random
import time
import pprint

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy
from utils.feature_extractor import FeatureExtractor

from args import args
import importlib

import data
import models


def main():
    print(args)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    if args.pretrained:
        pretrained(args, model)

    data = get_dataset(args)
    output_path = args.pretrained + "_activations"

     # setup feature extractor
    feature_extractor = FeatureExtractor(model)

    target_layers = feature_extractor.parse_default_layers()
    target_types = feature_extractor.parse_type("relu")

    feature_extractor.append_target_layers(target_layers, target_types)

    print(feature_extractor.module_dict)
    print(feature_extractor.target_outputs.keys())

    predicate(data.val_loader, feature_extractor, output_path)

def predicate(data_loader, feature_extractor, output_path=None):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    model = feature_extractor.model
    outputs_dict = dict()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        toc = time.time()
        for batch_ind, (input, _) in enumerate(data_loader):
            input = input.cuda(non_blocking=True)

            # forward to get intermediate outputs
            _ = model(input)

            # synchronize so that everything is calculated
            torch.cuda.synchronize()

            # print(feature_extractor.target_outputs)
            for target_layer, target_output in feature_extractor.target_outputs.items():
                if target_layer in outputs_dict:
                    outputs_dict[target_layer].append(target_output.data.numpy())
                else:
                    outputs_dict[target_layer] = [target_output.data.numpy()]

            # measure elapsed time
            batch_time.update(time.time() - toc)
            toc = time.time()

            if batch_ind % 10 == 0:
                print('Predicate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_ind, len(data_loader), batch_time=batch_time))

    if output_path is not None:
        def _squeeze_dict(d):
            for key, val in d.items():
                d[key] = np.concatenate(val, 0)
            return d

        outputs_dict = _squeeze_dict(outputs_dict)
        np.savez_compressed(output_path, **outputs_dict)


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


if __name__ == "__main__":
    main()
