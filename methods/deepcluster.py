#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of the paper:
# Deep Clustering for Unsupervised Learning of Visual Features; Caron et al. (2018)
# Paper: https://arxiv.org/pdf/1807.05520.pdf
# Code (adapted from): https://github.com/facebookresearch/deepcluster
# Requirements: sklearn (https://scikit-learn.org)

import math
import time
import collections

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional
import numpy as np
from PIL import Image
from utils import AverageMeter

class Model(torch.nn.Module):
    def __init__(self, feature_extractor, batch_size, num_clusters, train_transform):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.feature_extractor = feature_extractor
        self.train_transform = train_transform
        feature_size = feature_extractor.feature_size
        self.classifier = nn.Sequential(collections.OrderedDict([
            ("linear1",  nn.Linear(feature_size, 256)),
            ("bn1",      nn.BatchNorm1d(256)),
            ("relu",     nn.LeakyReLU()),
            ("linear2",  nn.Linear(256, num_clusters)),
        ]))
        self.optimizer = Adam([{"params": self.feature_extractor.parameters(), "lr": 0.001},
                               {"params": self.classifier.parameters(), "lr": 0.001}])

    def forward(self, x, detach=False):    
        if(detach): out = self.feature_extractor(x).detach()
        else: out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

    def preprocess_features(self, npdata): # pca=256):
        #from scipy.cluster.vq import whiten
        from sklearn.decomposition import PCA
        # npdata (np.array N * ndim): features to preprocess
        # pca (int): dim of output
        _, ndim = npdata.shape
        npdata =  npdata.astype("float32")
        # Apply PCA-whitening with scipy
        # npdata = whiten(npdata)
        # Sklearn PCA with withening
        if(npdata.shape[1]>=512): scale=0.5
        else: scale=1.0
        npdata = PCA(n_components=int(npdata.shape[1]*scale), whiten=True).fit_transform(npdata)
        # L2 normalization
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]
        # Return: np.array of dim N * pca (data PCA-reduced whitened and L2-normalized)
        return npdata

    def run_kmeans(self, x, num_clusters):
        from sklearn.cluster import KMeans
        local_seed = np.random.randint(time.time())
        kmeans = KMeans(n_clusters=num_clusters, random_state=local_seed, init="k-means++", n_jobs=16, max_iter=30, n_init=5).fit(x)
        return torch.tensor(kmeans.labels_, dtype=torch.int64)

    def compute_features(self, data_loader, model, batch_size):
        model.eval()
        features_list = list()
        with torch.no_grad():
            # discard the label information in the dataloader
            for i, (input_tensor, _) in enumerate(data_loader):
                input_tensor = input_tensor.cuda()
                aux = model(input_tensor).data.cpu().numpy()
                features_list.append(aux)
        features = np.concatenate(features_list, 0)
        return features

    def train(self, epoch, train_loader):
        start_time = time.time()       
        # At each epoch reset the last layer, since the labels are going to change
        self.classifier.linear2.weight.data.normal_(0, 0.01) #reset the weights
        self.classifier.linear2.bias.data.zero_() #reset the bias

        # The input is not a loader but a set for DeepCluster
        train_set = train_loader
        # Define a loader (no data augmentations)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=False, 
                                                   num_workers=8, pin_memory=True, drop_last=False)

        # Extract all the features of the dataset using the backbone
        features = self.compute_features(data_loader=train_loader, model=self.feature_extractor, batch_size=self.batch_size)
        # PCA normalized and withening
        features_normalized = self.preprocess_features(features)
        # Cluster the data
        pseudolabels = self.run_kmeans(features_normalized, self.num_clusters)

        # Each cluster may have a different amount of points,
        #therefore it is necessary to weight each class in the Cross Entropy loss
        weight = torch.zeros(self.num_clusters, dtype=torch.float32)
        for value in pseudolabels: weight[value] += 1.0
        weight = 1.0 - (weight / len(pseudolabels)) #inverse frequency
        ce = torch.nn.CrossEntropyLoss(weight.cuda()).cuda()

        # Another way to do the same thing
        #ce = torch.nn.CrossEntropyLoss().cuda()        
        #indices_per_class_list = [[] for _ in range(self.num_clusters)]
        #for idx, value in enumerate(pseudolabels): indices_per_class_list[value].append(idx)
        #train_sampler = UnbalancedSampler(N=len(pseudolabels), indices_per_class=indices_per_class_list)

        # Assign the pseudolabels obtained with Kmeans to the train dataset
        train_set = ReassignedDataset(train_set, pseudolabels, transform=self.train_transform)

        # Re-define the train dataloader with shuffling enabled
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True, drop_last=False)

        # Enable training and define optimizer
        self.feature_extractor.train()
        self.classifier.train()

        # Training loop of one epoch, using the pseudolabels
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for i, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available(): data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = ce(output, target)
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
                + "; acc: " + str(accuracy_meter.avg) + "%")
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

class ReassignedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pseudolabels, transform=None):
        self.dataset = dataset
        self.pseudolabels = pseudolabels
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        #img = img.permute(1,2,0).data.cpu().numpy()
        img = torchvision.transforms.functional.to_pil_image(img)
        pseudolabel = self.pseudolabels[index]
        #img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.pseudolabels)

class UnbalancedSampler(torch.utils.data.sampler.Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
          N (int): dataset size.
          indices_per_class: dict of key (target), 
            and value (list of data with this target)
    """

    def __init__(self, N, indices_per_class):
        self.N = N
        self.indices_per_class = indices_per_class
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.indices_per_class)):
            if len(self.indices_per_class[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.indices_per_class)):
            # skip empty clusters
            if len(self.indices_per_class[i]) == 0:
                continue
            indexes = np.random.choice(
                self.indices_per_class[i],
                size_per_pseudolabel,
                replace=(len(self.indices_per_class[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype("int"))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res
        
    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)      
