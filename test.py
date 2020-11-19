#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning

# Test on different datasets/backbones given a pre-trained checkpoint.
# This is generally used after the linear-evaluation phase.
# Example command:
#
# python test.py --dataset="cifar10" --backbone="conv4" --seed=3 --data_size=128 --gpu=0 --checkpoint="./checkpoint/relationnet/cifar10/relationnet_cifar10_conv4_seed_3_epoch_100_linear_evaluation.tar"

import os
import argparse

parser = argparse.ArgumentParser(description='Test script: loads and test a pre-trained model (e.g. after linear evaluation)')
parser.add_argument('--seed', default=-1, type=int, help='Seed for Numpy and pyTorch. Default: -1 (None)')
parser.add_argument('--dataset', default='cifar10', help='Dataset: cifar10|100, supercifar100, tiny, slim, stl10')
parser.add_argument('--backbone', default='conv4', help='Backbone: conv4, resnet|8|32|34|56')
parser.add_argument('--data_size', default=256, type=int, help='Total number of epochs.')
parser.add_argument('--checkpoint', default='./', help='Address of the checkpoint file')
parser.add_argument('--gpu', default=0, type=int, help='GPU id in case of multiple GPUs')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import numpy as np
import random

if(args.seed>=0):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    print("[INFO] Setting SEED: " + str(args.seed))   
else:
    print("[INFO] Setting SEED: None")

if(torch.cuda.is_available() == False): print('[WARNING] CUDA is not available.')

print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device type: " + str(device))

from datamanager import DataManager
manager = DataManager(args.seed)
num_classes = manager.get_num_classes(args.dataset)

if(args.backbone=="conv4"):
    from backbones.conv4 import Conv4
    feature_extractor = Conv4(flatten=True)
elif(args.backbone=="resnet8"):
    from backbones.resnet_small import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, [1, 1, 1], channels=[16, 32, 64], flatten=True)
elif(args.backbone=="resnet32"):
    from backbones.resnet_small import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, [5, 5, 5], channels=[16, 32, 64], flatten=True)
elif(args.backbone=="resnet56"):
    from backbones.resnet_small import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, [9, 9, 9], channels=[16, 32, 64], flatten=True)
elif(args.backbone=="resnet34"):
    from backbones.resnet_large import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, layers=[3, 4, 6, 3],zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None)
else:
    raise RuntimeError("[ERROR] the backbone " + str(args.backbone) +  " is not supported.")

print("[INFO]", str(str(args.backbone)), "loaded in memory.")
print("[INFO] Feature size:", str(feature_extractor.feature_size))

test_loader = manager.get_test_loader(args.dataset, args.data_size)


def main():
    from methods.standard import StandardModel
    model = StandardModel(feature_extractor, num_classes)
    print("[INFO] Loading checkpoint...")
    model.load(args.checkpoint)
    print("[INFO] Loading checkpoint: done!")

    model.to(device)

    print("Testing...")
    loss_test, accuracy_test = model.test(test_loader)
    print("===========================")
    print("Seed:", str(args.seed))
    print("Test loss:", str(loss_test) + "%")
    print("Test accuracy:", str(accuracy_test) + "%")
    print("===========================")
         
if __name__== "__main__": main()
