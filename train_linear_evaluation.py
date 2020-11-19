#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning

# Manage the linear evaluation phase on different datasets/backbones/methods.
# This script should be used after the self-supervised training phase to evaluate the methods.
# Example command:
#
# python train_linear_evaluation.py --dataset="cifar10" --method="relationnet" --backbone="conv4" --seed=3 --data_size=128 --gpu=0 --epochs=100 --checkpoint="./checkpoint/relationnet/cifar10/relationnet_cifar10_conv4_seed_3_epoch_200.tar"

import os
import argparse

parser = argparse.ArgumentParser(description="Linear evaluation script")
parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs")
parser.add_argument("--dataset", default="cifar10", help="Dataset: cifar10|100, supercifar100, tiny, slim, stl10")
parser.add_argument("--backbone", default="conv4", help="Backbone: conv4, resnet|8|32|34|56")
parser.add_argument("--method", default="standard", help="Method name (used just as string in the checkpoint name)")
parser.add_argument("--data_size", default=128, type=int, help="Size of the mini-batch")
parser.add_argument("--checkpoint", default="./", help="Address of the checkpoint file")
parser.add_argument("--finetune", default="False", type=str, help="Finetune the backbone during training (default: False)")
parser.add_argument("--num_workers", default=8, type=int, help="Number of torchvision workers used to load data (default: 8)")
parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
parser.add_argument("--gpu", default=0, type=int, help="GPU id in case of multiple GPUs")
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

if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

if(args.finetune=="True" or args.finetune=="true"):
    print("[INFO] Finetune set to True, the backbone will be finetuned.")
print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device type:", str(device))

from datamanager import DataManager
manager = DataManager(args.seed)
num_classes = manager.get_num_classes(args.dataset)
train_transform = manager.get_train_transforms("lineval", args.dataset)
train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                        data_type="single",
                                        data_size=args.data_size,
                                        train_transform=train_transform,
                                        repeat_augmentations=None,
                                        num_workers=args.num_workers, 
                                        drop_last=False)

test_loader = manager.get_test_loader(args.dataset, args.data_size)

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

def main():
    print("[INFO] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint)
    feature_extractor.load_state_dict(checkpoint["backbone"])
    from methods.standard import StandardModel
    model = StandardModel(feature_extractor, num_classes)
    model.to(device)
    if not os.path.exists("./checkpoint/"+str(args.method)+"/"+str(args.dataset)):
        os.makedirs("./checkpoint/"+str(args.method)+"/"+str(args.dataset))

    if(args.id!=""):    
        header = str(args.method)+ "_" + str(args.id) + "_" + str(args.dataset) + "_" + str(args.backbone) + "_seed_" + str(args.seed)
    else:
        header = str(args.method) + "_" + str(args.dataset) + "_" + str(args.backbone) + "_seed_" + str(args.seed)
 
    for epoch in range(0, args.epochs):
        if(args.finetune=="True" or args.finetune=="true"):
            loss_train, accuracy_train = model.finetune(epoch, train_loader)
        else:
            loss_train, accuracy_train = model.linear_evaluation(epoch, train_loader)

    checkpoint_path = "./checkpoint/"+str(args.method)+"/"+str(args.dataset)+"/"+header+"_epoch_"+ str(epoch+1)+"_linear_evaluation.tar"
    print("[INFO] Saving in:", checkpoint_path)
    model.save(checkpoint_path)
    loss_test, accuracy_test = model.test(test_loader)
    print("Test accuracy: " + str(accuracy_test) + "%")
          
if __name__== "__main__": main()
