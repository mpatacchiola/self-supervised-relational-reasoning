#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of the paper:
# "A Simple Framework for Contrastive Learning of Visual Representations", Chen et al. (2020)
# Paper: https://arxiv.org/abs/2002.05709
# Code (adapted from):
# https://github.com/pietz/simclr
# https://github.com/google-research/simclr

import math
import time
import collections

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils import AverageMeter
               
class Model(torch.nn.Module):
    def __init__(self, feature_extractor):
        super(Model, self).__init__()

        self.net = nn.Sequential(collections.OrderedDict([
          ("feature_extractor", feature_extractor)
        ]))

        self.head = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(feature_extractor.feature_size, 256)),
          ("bn1",      nn.BatchNorm1d(256)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(256, 64)),
        ]))

        self.optimizer = Adam([{"params": self.net.parameters(), "lr": 0.001},
                               {"params": self.head.parameters(), "lr": 0.001}])

    def return_loss_fn(self, x, t=0.5, eps=1e-8):
        # Taken from: https://github.com/pietz/simclr/blob/master/SimCLR.ipynb
        # Estimate cosine similarity
        n = torch.norm(x, p=2, dim=1, keepdim=True)
        x = (x @ x.t()) / (n * n.t()).clamp(min=eps)
        x = torch.exp(x / t)
        # Put positive pairs on the diagonal
        idx = torch.arange(x.size()[0])
        idx[::2] += 1
        idx[1::2] -= 1
        x = x[idx]
        # subtract the similarity of 1 from the numerator
        x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
        # NOTE: some implementation have used the loss `torch.mean(-torch.log(x))`,
        # but in preliminary experiments we saw that `-torch.log(x.mean())` is slightly
        # more effective (e.g. 77% vs 76% on CIFAR-10).
        return -torch.log(x.mean())

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train()
        self.head.train()
        loss_meter = AverageMeter()
        statistics_dict = {}
        for i, (data, data_augmented, _) in enumerate(train_loader):
            data = torch.stack(data_augmented, dim=1)
            d = data.size()
            train_x = data.view(d[0]*2, d[2], d[3], d[4]).cuda()
            
            self.optimizer.zero_grad()                     
            features = self.net(train_x)
            tot_pairs = int(features.shape[0]*features.shape[0])
            embeddings = self.head(features)
            loss = self.return_loss_fn(embeddings)
            loss_meter.update(loss.item(), features.shape[0])
            loss.backward()
            self.optimizer.step()
            if(i==0):
                statistics_dict["batch_size"] = data.shape[0]
                statistics_dict["tot_pairs"] = tot_pairs

        elapsed_time = time.time() - start_time 
        print("Epoch [" + str(epoch) + "]"
               + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
               + " loss: " + str(loss_meter.avg)
               + "; batch-size: " + str(statistics_dict["batch_size"])
               + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))
                             
        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        feature_extractor_state_dict = self.net.feature_extractor.state_dict()
        head_state_dict = self.head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "head": head_state_dict,
                    "optimizer": optimizer_state_dict}, 
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.head.load_state_dict(checkpoint["head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
