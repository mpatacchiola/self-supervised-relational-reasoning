#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of the paper:
# "Unsupervised representation learning by predicting image rotations.", Gidaris et al. (2018)
# Paper: https://arxiv.org/abs/1803.07728

import math
import time

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils import AverageMeter

def _data_augmentation(images):
    output_list = list()
    target_list = list()
    
    output_list.append(images)
    target_list.append(torch.from_numpy(np.full(images.shape[0], 0, dtype=np.int64)))

    output_list.append(torch.rot90(images, 1, dims=(2,3)))
    target_list.append(torch.from_numpy(np.full(images.shape[0], 1, dtype=np.int64)))

    output_list.append(torch.rot90(images, 2, dims=(2,3)))
    target_list.append(torch.from_numpy(np.full(images.shape[0], 2, dtype=np.int64)))

    output_list.append(torch.rot90(images, 3, dims=(2,3)))
    target_list.append(torch.from_numpy(np.full(images.shape[0], 3, dtype=np.int64)))
    return torch.cat(output_list, 0), torch.cat(target_list, 0)

class Model(torch.nn.Module):
    def __init__(self, feature_extractor):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(feature_extractor.feature_size, 4) #the 4 rotations
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam([{"params": self.feature_extractor.parameters(), "lr": 0.001},
                               {"params": self.classifier.parameters(), "lr": 0.001}])

    def forward(self, x, detach=False):    
        if(detach): out = self.feature_extractor(x).detach()
        else: out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.feature_extractor.train()
        self.classifier.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for i, (data, _) in enumerate(train_loader):
            data, target = _data_augmentation(data)
            if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]" 
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg) 
              + "; acc.: " + str(accuracy_meter.avg) )
        return loss_meter.avg, accuracy_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        state_dict = self.classifier.state_dict()
        feature_extractor_state_dict = self.feature_extractor.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"classifier": state_dict, 
                    "backbone": feature_extractor_state_dict,
                    "optimizer": optimizer_state_dict}, 
                    file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
