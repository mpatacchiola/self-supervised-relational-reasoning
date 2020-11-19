#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of the paper:
# "Learning deep representations by mutual information estimation and maximization", Hjelm et al. (2018)
# Paper: https://arxiv.org/abs/1808.06670
# Code (adapted from): 
# https://github.com/DuaneNielsen/DeepInfomaxPytorch
# https://github.com/rdevon/DIM

import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from utils import AverageMeter

class Encoder(nn.Module):
    """The encoder class.

    Takes a feature extractor and returns the representaion
    produced by it (y), and additionally the feature-maps of the
    very first layer (M).
    """
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        # Set y_size and M_channels based on the type of backbone
        self.y_size = feature_extractor.feature_size
        if(self.feature_extractor.name=="conv4"): self.M_channels=8
        elif(self.feature_extractor.name=="resnet"): self.M_channels=feature_extractor.channels[0]
        elif(self.feature_extractor.name=="resnet_large"): self.M_channels=feature_extractor.channels[0]
        else: raise ValueError("[ERROR][DeepInfoMax] The network type " + str(self.feature_extractor.name) + " is not supported!")
        print("[INFO][DeepInfoMax] y-size: " + str(self.y_size) + "; M-channels: " + str(self.M_channels))

    def forward_resnet_large(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        M = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(M)

        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)

        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x) #not used in a backbone-style net

        return x, M

    def forward_resnet(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        M = self.feature_extractor.relu(x)
        x = self.feature_extractor.layer1(M)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        if(self.feature_extractor.flatten):
            x = self.feature_extractor.avgpool(x)
            x = x.view(x.size(0), -1)
        return x, M

    def forward_conv4(self, x):
        x = self.feature_extractor.layer1.conv(x)
        x = self.feature_extractor.layer1.bn(x)
        M = self.feature_extractor.layer1.relu(x)
        x = self.feature_extractor.layer1.avgpool(M)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)
        if(self.feature_extractor.is_flatten): x = self.feature_extractor.flatten(x)
        return x, M

    def forward(self, x):
        if(self.feature_extractor.name=="conv4"):
            return self.forward_conv4(x)
        elif(self.feature_extractor.name=="resnet"):
            return self.forward_resnet(x)
        elif(self.feature_extractor.name=="resnet_large"):
            return self.forward_resnet_large(x)
        else:
            raise ValueError("[ERROR][DeepInfoMax] The network type " + str(self.feature_extractor.name) + " is not supported!")
       
class GlobalDiscriminator(nn.Module):
    def __init__(self, y_size, M_channels):
        super().__init__()
        self.c0 = nn.Conv2d(M_channels, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d(16) #adaptive downsize to [b, c, 16, 16]
        self.l0 = nn.Linear(32*16*16+y_size, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = self.avgpool(h)
        h = h.view(M.shape[0], -1) #flattening
        h = torch.cat((y, h), dim=1) #[128, 64] cat [128, 128] = [128, 192]
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    """The local discriminator class.

    A network that analyses the relation between the
    output of the encoder y, and the feature map M.
    It is called "local" because it compares y with
    each one of the features in M. So if M is a [64, 6, 6]
    feature map, and y is a [32] vector, the comparison is
    done concatenating y along each one of the 6x6 features
    in M: 
    (i) [32] -> [64, 1, 1]; (ii) [32] -> [64, 1, 2]
    ... (xxxvi) [32] -> [64, 6, 6]. 
    This can be efficiently done expanding y to have same 
    dimensionality as M such that:
    [32] torch.expand -> [32, 6, 6]
    and then concatenate on the channel dimension:
    [32, 6, 6] torch.cat(axis=0) -> [64, 6, 6] = [96, 6, 6]
    The tensor is then feed to the local discriminator.
    """
    def __init__(self, y_size, M_channels):
        super().__init__()
        # conv4 has y_size=64 and M_channels=8
        # resnet32 has y_size=64 and M_channels=16
        # resnet34 has y_size=512 and M_channels=64
        self.c0 = nn.Conv2d(y_size+M_channels, 256, kernel_size=1)
        self.c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.c2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    """The prior discriminator class.

    This discriminate between a vector drawn from random uniform,
    and the vector y obtained as output of the encoder.
    It enforces y to be close to a uniform distribution.
    """
    def __init__(self, y_size):
        super().__init__()
        self.l0 = nn.Linear(y_size, 512)
        self.l1 = nn.Linear(512, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, y_size, M_channels, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        # Generate the networks only if needed (hyperparam != 0)
        if(alpha!=0.0): self.global_d = GlobalDiscriminator(y_size, M_channels)
        if(beta!=0.0): self.local_d = LocalDiscriminator(y_size, M_channels)
        if(gamma!=0.0):self.prior_d = PriorDiscriminator(y_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        if(self.beta!=0.0):
            # Expanding the vector y to have same dimensionality as M.
            # Note that expansion is done only on the height and width not on channels.
            y_expanded = y.unsqueeze(-1).unsqueeze(-1)
            y_expanded = y_expanded.expand(-1, -1, M.shape[2], M.shape[3])
            # Concat y_expanded and M, in order to perform local discrimination.
            # Note that channels can be different between y_expanded and M, this
            # does not matter since concatenation is along channel dimension.
            y_M = torch.cat((M, y_expanded), dim=1)
            y_M_prime = torch.cat((M_prime, y_expanded), dim=1)
            # Forward through local discriminator
            Ej = -F.softplus(-self.local_d(y_M)).mean()
            Em = F.softplus(self.local_d(y_M_prime)).mean()
            LOCAL = (Em - Ej) * self.beta
        else:
            LOCAL = 0.0

        if(self.alpha!=0.0):
            Ej = -F.softplus(-self.global_d(y, M)).mean()
            Em = F.softplus(self.global_d(y, M_prime)).mean()
            GLOBAL = (Em - Ej) * self.alpha
        else:
            GLOBAL = 0.0

        if(self.gamma!=0.0):
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0.0

        return LOCAL + GLOBAL + PRIOR


class DIM(nn.Module):
    def __init__(self, feature_extractor, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.encoder = Encoder(feature_extractor)
        y_size = self.encoder.y_size
        M_channels = self.encoder.M_channels
        self.loss_fn = DeepInfoMaxLoss(y_size, M_channels, alpha, beta, gamma)
        if torch.cuda.is_available(): 
             self.encoder = self.encoder.cuda()
             self.loss_fn = self.loss_fn.cuda()
        # Training with a lower learning rate of 1e-4 as reported
        # by the authors in the paper. Higher values seem to lead to a NaN loss.
        self.optimizer = Adam([{"params": self.encoder.parameters(), "lr": 1e-4},
                               {"params": self.loss_fn.parameters(), "lr": 1e-4}])

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.encoder.train()
        self.loss_fn.train()
        loss_meter = AverageMeter()
        for i, (data, _) in enumerate(train_loader):
            if torch.cuda.is_available(): data = data.cuda()
            
            self.optimizer.zero_grad()
            y, M = self.encoder(data)
 
            # This correspons to "roll" the mini-batch along the first
            #dimension of one shift. This is used to create pairs for comparison.
            M_prime = torch.cat([M[1:], M[0].unsqueeze(0)], dim=0)
            
            loss = self.loss_fn(y, M, M_prime)
            loss_meter.update(loss.item(), data.shape[0])
            loss.backward()
            self.optimizer.step()

        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]" 
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg))
        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        feature_extractor_state_dict = self.encoder.feature_extractor.state_dict()
        loss_fn_state_dict = self.loss_fn.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "loss_fn": loss_fn_state_dict,
                    "optimizer": optimizer_state_dict}, 
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.encoder.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.loss_fn.load_state_dict(checkpoint["loss_fn"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
