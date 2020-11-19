#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning

# Data manager that returns transformations and samplers for each method/dataset.
# If a new method is included it should be added to the "DataManager" class.
# The dataset classes with prefix "Multi" are overriding the original dataset class
# to allow multi-sampling of more images in parallel (required by our method).

import os
import sys
import random
import pickle

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class MultiSTL10(dset.STL10):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pic = Image.fromarray(np.transpose(img, (1, 2, 0)))

        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class MultiCIFAR10(dset.CIFAR10):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations
            
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        pic = Image.fromarray(img)
            
        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)
        else:
            img_list = None
           
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class MultiCIFAR100(dset.CIFAR100):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations
            
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        pic = Image.fromarray(img)
            
        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)
        else:
            img_list = None
           
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class SuperCIFAR100(dset.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train,
                 transform, target_transform,
                 download)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, "rb")
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding="latin1")
                self.train_data.append(entry["data"])
                self.train_labels += entry["coarse_labels"]
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, "rb")
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding="latin1")
            self.test_data = entry["data"]
            self.test_labels = entry["coarse_labels"]
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class TinyImageFolder(dset.ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)


class MultiTinyImageFolder(dset.ImageFolder):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations
            
    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(pic, np.uint8, copy=False))
            
        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)
        else:
            img_list = None
           
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class SlimImageFolder(dset.ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)


class MultiSlimImageFolder(dset.ImageFolder):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations
            
    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(pic, np.uint8, copy=False))
            
        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)
        else:
            img_list = None
           
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class DataManager():
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _check(self, dataset):
        datasets_list = ["cifar10", "stl10", "cifar100", 
                         "supercifar100", "tiny", "slim"]
        if(dataset not in datasets_list):
            raise Exception("[ERROR] The dataset " + str(dataset) + " is not supported!")            
        if(dataset=="slim"):
            if(os.path.isdir("./data/SlimageNet64/train")==False):
                raise Exception("[ERROR] The train data of SlimageNet64 has not been found in ./data/SlimageNet64/train \n"
                            + "1) Download the dataset from: https://zenodo.org/record/3672132 \n"
                            + "2) Uncompress the dataset in ./data/SlimageNet64  \n"
                            + "3) Place training images in /train and test images in /test")
        elif(dataset=="tiny"):
            if(os.path.isdir("./data/tiny-imagenet-200/train")==False):                            
                raise Exception("[ERROR] The train data of TinyImagenet has not been found in ./data/tiny-imagenet-200/train \n"
                            + "1) Download the dataset \n"
                            + "2) Uncompress the dataset in ./data/tiny-imagenet-200  \n"
                            + "3) Place training images in /train and test images in /test")                         

    def get_num_classes(self, dataset):
        self._check(dataset)
        if(dataset=="cifar10"): return 10
        elif(dataset=="stl10"): return 10
        elif(dataset=="supercifar100"): return 20
        elif(dataset=="cifar100"): return 100
        elif(dataset=="tiny"): return 200
        elif(dataset=="slim"): return 1000

    def get_train_transforms(self, method, dataset):
        """Returns the training torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
          dataset: The name of the dataset.
        Returns:
          train_transform: An object of type torchvision.transforms.
        """
        self._check(dataset)
        if(dataset=="cifar10"): 
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            side = 32; padding = 4; cutout=0.25
        elif(dataset=="stl10"): 
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            side = 96; padding = 12; cutout=0.111
        elif(dataset=="cifar100" or dataset=="supercifar100"): 
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            side = 32; padding = 4; cutout=0.0625
        elif(dataset=="tiny"):
            #Image-Net --> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            side = 64; padding = 8
        elif(dataset=="slim"):
            #Image-Net --> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            side = 64; padding = 8

        if(method=="relationnet" or method=="simclr"):
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            #rnd_rot = transforms.RandomRotation(10., resample=2),
            train_transform = transforms.Compose([rnd_resizedcrop, rnd_hflip,
                                                  rnd_color_jitter, rnd_gray, transforms.ToTensor(), normalize])
        elif(method=="deepinfomax"):
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        elif(method=="standard" or method=="rotationnet" or method=="deepcluster"):
            train_transform = transforms.Compose([transforms.RandomCrop(side, padding=padding), 
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(), normalize])
        elif(method=="finetune"):
            rnd_affine = transforms.RandomApply([transforms.RandomAffine(18, scale=(0.9, 1.1),
                                                                         translate=(0.1, 0.1), shear=10,
                                                                         resample=Image.BILINEAR, fillcolor=0)], p=0.5)
            train_transform = transforms.Compose([#transforms.RandomCrop(side, padding=padding),
                                                  transforms.RandomHorizontalFlip(),
                                                  rnd_affine,
                                                  transforms.ToTensor(), normalize,
                                                  #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))]) #pytorch default
                                                  transforms.RandomErasing(p=0.5, scale=(cutout, cutout), ratio=(1.0, 1.0))])
        elif(method=="lineval"):
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise Exception("[ERROR] The method " + str(method) + " is not supported!")

        return train_transform

    def get_train_loader(self, dataset, data_type, data_size, train_transform, repeat_augmentations, num_workers=8, drop_last=False):
        """Returns the training loader for each dataset/method.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          method: The name of the method.
          dataset: The name of the dataset.
          data_type: The type of data multi (multiple images in parallel),
            single (one image at the time), unsupervised (used in STL10 to load 
            the unlabeled data split).
          data_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          repeat_augmentations: repeat the augmentations on the same image
            for the specified number of times (needed by RelationNet and SimCLR).
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.            
        Returns:
          train_loader: The loader that can be used a training time.
          train_set: The train set (used in DeepCluster)
        """
        self._check(dataset)
        from torch.utils.data.dataset import Subset
        if(data_type=="multi"):
        #Used for: Relational reasoning, SimCLR
            if(dataset=="cifar10"): 
                train_set = MultiCIFAR10(repeat_augmentations, root="data", train=True, transform=train_transform, download=True)
            elif(dataset=="stl10"):
                train_set = MultiSTL10(repeat_augmentations, root="data", split="unlabeled", transform=train_transform, download=True)
            elif(dataset=="cifar100"): 
                train_set = MultiCIFAR100(repeat_augmentations, root="data", train=True, transform=train_transform, download=True)
            elif(dataset=="tiny"): 
                train_set = MultiTinyImageFolder(repeat_augmentations, root="./data/tiny-imagenet-200/train", transform=train_transform)
            elif(dataset=="slim"): 
                train_set = MultiSlimImageFolder(repeat_augmentations, root="./data/SlimageNet64/train", transform=train_transform)
        elif(data_type=="single"):
        #Used for: deepinfomax, rotationnet, standard, lineval, finetune, deepcluster
            if(dataset=="cifar10"): 
                train_set = dset.CIFAR10("data", train=True, transform=train_transform, download=True)
            elif(dataset=="stl10"):
                train_set = dset.STL10(root="data", split="train", transform=train_transform, download=True)
            elif(dataset=="supercifar100"):
                train_set = SuperCIFAR100("data", train=True, transform=train_transform, download=True)
            elif(dataset=="cifar100"):
                train_set = dset.CIFAR100("data", train=True, transform=train_transform, download=True)
            elif(dataset=="tiny"): 
                train_set = TinyImageFolder(root="./data/tiny-imagenet-200/train", transform=train_transform)
            elif(dataset=="slim"): 
                train_set = SlimImageFolder(root="./data/SlimageNet64/train", transform=train_transform)
        elif(data_type=="unsupervised"):
            if(dataset=="stl10"):
                train_set = dset.STL10(root="data", split="unlabeled", transform=train_transform, download=True)
        else:
            raise Exception("[ERROR] The type " + str(data_type) + " is not supported!")

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=data_size, shuffle=True, 
                                                   num_workers=num_workers, pin_memory=True, drop_last=drop_last)        
        return train_loader, train_set


    def get_test_loader(self, dataset, data_size, num_workers=8):
        self._check(dataset)        
        if(dataset=="cifar10"): 
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.CIFAR10("data", train=False, transform=test_transform, download=True)
        elif(dataset=="stl10"): 
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.STL10("data", split="test", transform=test_transform, download=True)
        elif(dataset=="supercifar100"): 
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = SuperCIFAR100("data", train=False, transform=test_transform, download=True)
        elif(dataset=="cifar100"): 
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.CIFAR100("data", train=False, transform=test_transform, download=True)
        elif(dataset=="tiny"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = TinyImageFolder(root="./data/tiny-imagenet-200/val", transform=test_transform)
        elif(dataset=="slim"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = SlimImageFolder(root="./data/SlimageNet64/test", transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=data_size, shuffle=False, 
                                                   num_workers=num_workers, pin_memory=True)
        return test_loader
