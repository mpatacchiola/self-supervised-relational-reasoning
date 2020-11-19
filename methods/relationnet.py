#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of our Relational Reasoning method as described in the paper.
# This code use a Focal Loss but also a standard BCE loss can be used.
# An essential version of this code has also been provided in the repository.

import time
import math
import collections
import numpy as np

import torch
from torch import nn
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter

class FocalLoss(torch.nn.Module):
  """Sigmoid focal cross entropy loss.
  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, gamma=2.0, alpha=0.25):
    """Constructor.
    Args:
      gamma: exponent of the modulating factor (1 - p_t)^gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives,
           with alpha in [0, 1] for class 1 and 1-alpha for class 0. 
           In practice alpha may be set by inverse class frequency,
           so that for a low number of positives, its weight is high.
    """
    super(FocalLoss, self).__init__()
    self._alpha = alpha
    self._gamma = gamma
    self.BCEWithLogits = nn.BCEWithLogitsLoss(reduction="none")

  def forward(self, prediction_tensor, target_tensor):
    """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets.
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    per_entry_cross_ent = self.BCEWithLogits(prediction_tensor, target_tensor)
    prediction_probabilities = torch.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) + #positives probs
          ((1 - target_tensor) * (1 - prediction_probabilities))) #negatives probs
    modulating_factor = 1.0
    if self._gamma:
        modulating_factor = torch.pow(1.0 - p_t, self._gamma) #the lowest the probability the highest the weight
    alpha_weight_factor = 1.0
    if self._alpha is not None:
        alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
    return torch.mean(focal_cross_entropy_loss)
                    
class Model(torch.nn.Module):
    def __init__(self, feature_extractor, device="cuda", aggregation="cat"):
        super(Model, self).__init__()
        self.device = device
        self.net = nn.Sequential(collections.OrderedDict([
          ("feature_extractor", feature_extractor),
        ]))

        self.aggregation=aggregation
        if(self.aggregation=="cat"): resizer=2
        elif(self.aggregation=="sum"): resizer=1
        elif(self.aggregation=="mean"): resizer=1
        elif(self.aggregation=="max"): resizer=1
        else: RuntimeError("[ERROR] aggregation type " + str(self.aggregation) +  " not supported, must be: cat, sum, mean.")

        self.relation_module = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(feature_extractor.feature_size*resizer, 256)),
          ("bn1",      nn.BatchNorm1d(256)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(256, 1)),
        ]))

        self.optimizer = Adam([{"params": self.net.parameters(), "lr": 0.001},
                               {"params": self.relation_module.parameters(), "lr": 0.001}])

        self.fl = FocalLoss(gamma=2.0, alpha=0.5) #Using reccommended value for gamma: 2.0
        #self.bce = nn.BCEWithLogitsLoss() # Standard BCE loss can also be used


    def aggregate(self, features, tot_augmentations, type="cat"):
        """Aggregation function.
        Args:
          features: The features returned by the backbone, it is a tensor
            of shape [batch_size*K, feature_size].
            num_classes] representing the predicted logits for each class
          tot_augmentations: The total number of augmentations, corresponds
            to the parameter K in the paper.
        Returns:
          relation_pairs: a tensor with the aggregated pairs that can be
            given as input to the relation head. 
          target: the values (zeros and ones) for each pair, that
            represents the target used to train the relation head.
          tot_positive: Counter for the total number of positives.
          tot_negative: Counter for the total number of negatives.
        """
        relation_pairs_list = list()
        target_list = list()
        size = int(features.shape[0] / tot_augmentations)
        tot_positive = 0.0
        tot_negative = 0.0
        shifts_counter=1
        for index_1 in range(0, size*tot_augmentations, size):
            for index_2 in range(index_1+size, size*tot_augmentations, size):
                if(type=="cat"): 
                    positive_pair = torch.cat([features[index_1:index_1+size], features[index_2:index_2+size]], 1)
                    negative_pair = torch.cat([features[index_1:index_1+size], 
                                               torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)], 1)
                elif(type=="sum"): 
                    positive_pair = features[index_1:index_1+size] + features[index_2:index_2+size]
                    negative_pair = features[index_1:index_1+size] + torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)
                elif(type=="mean"): 
                    positive_pair = (features[index_1:index_1+size] + features[index_2:index_2+size]) / 2.0
                    negative_pair = (features[index_1:index_1+size] + torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)) / 2.0
                elif(type=="max"):
                    positive_pair, _ = torch.max(torch.stack([features[index_1:index_1+size], features[index_2:index_2+size]], 2), 2)
                    negative_pair, _ = torch.max(torch.stack([features[index_1:index_1+size], 
                                                              torch.roll(features[index_2:index_2+size], shifts=shifts_counter, dims=0)], 2), 2)
                relation_pairs_list.append(positive_pair)
                relation_pairs_list.append(negative_pair)
                target_list.append(torch.ones(size, dtype=torch.float32))
                target_list.append(torch.zeros(size, dtype=torch.float32))
                tot_positive += size 
                tot_negative += size
                shifts_counter+=1
                if(shifts_counter>=size): shifts_counter=1 # reset to avoid neutralizing the roll
        relation_pairs = torch.cat(relation_pairs_list, 0)
        target = torch.cat(target_list, 0)
        return relation_pairs, target, tot_positive, tot_negative


    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train()
        self.relation_module.train()
        accuracy_pos_list = list()
        accuracy_neg_list = list()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        statistics_dict = {}
        for i, (data, data_augmented, _) in enumerate(train_loader):
            batch_size = data.shape[0]
            tot_augmentations = len(data_augmented)
            train_x = torch.cat(data_augmented, 0).to(self.device)

            self.optimizer.zero_grad()
            # forward pass in the backbone                  
            features = self.net(train_x)
            # aggregation over the representations returned by the backbone
            relation_pairs, train_y, tot_positive, tot_negative = self.aggregate(features, tot_augmentations, type=self.aggregation)
            train_y = train_y.to(self.device)
            tot_pairs = int(relation_pairs.shape[0])
            # forward of the pairs through the relation head
            predictions = self.relation_module(relation_pairs).squeeze()
            # estimate the focal loss (also standard BCE can be used here)
            loss = self.fl(predictions, train_y)
            loss_meter.update(loss.item(), len(train_y))
            # backward step and weights update
            loss.backward()
            self.optimizer.step()
            
            best_guess = torch.round(torch.sigmoid(predictions))
            correct = best_guess.eq(train_y.view_as(best_guess))
            correct_positive = correct[0:int(len(correct)/2)].cpu().sum()
            correct_negative = correct[int(len(correct)/2):].cpu().sum()
            correct = correct.cpu().sum()
            accuracy = (100.0 * correct / float(len(train_y))) 
            accuracy_meter.update(accuracy.item(), len(train_y))
            accuracy_pos_list.append((100.0 * correct_positive / float(len(train_y)/2)).item())
            accuracy_neg_list.append((100.0 * correct_negative / float(len(train_y)/2)).item())
            if(i==0): 
                statistics_dict["batch_size"] = batch_size
                statistics_dict["tot_pairs"] = tot_pairs
                statistics_dict["tot_positive"] = int(tot_positive)
                statistics_dict["tot_negative"] = int(tot_negative)
        elapsed_time = time.time() - start_time
        
        # Here we are printing a rich set of information to monitor training.
        # The accuracy over both positive and negative pairs is printed separately.
        # The batch-size and total number of pairs is printed for debugging.
        print("Epoch [" + str(epoch) + "]"
                + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                + " loss: " + str(loss_meter.avg)
                + "; acc: " + str(accuracy_meter.avg) + "%"
                + "; acc+: " + str(round(np.mean(accuracy_pos_list), 2)) + "%"
                + "; acc-: " + str(round(np.mean(accuracy_neg_list), 2)) + "%"
                + "; batch-size: " + str(statistics_dict["batch_size"])
                + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))
        return loss_meter.avg, accuracy_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        feature_extractor_state_dict = self.net.feature_extractor.state_dict()
        relation_state_dict = self.relation_module.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "relation": relation_state_dict,
                    "optimizer": optimizer_state_dict}, 
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.relation_module.load_state_dict(checkpoint["relation"])
        if("optimizer" in checkpoint): 
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("[INFO][RelationNet] Loaded optimizer state-dict")
