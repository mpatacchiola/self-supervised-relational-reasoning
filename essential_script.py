#MIT License
#
#Copyright (c) 2020 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#This is an essential implementation of the paper:
#
#"Self-Supervised Relational Reasoning for Representation Learning"
#Patacchiola M. and Strokey A., Advances in Neural Information 
#Processing Systems (NeurIPS 2020, Spotlight).
#
#The code runs on CPU (porting on GPU is trivial in PyTorch) with the
#hyper-parameters set in the main function. The script will download
#the CIFAR-10 dataset if not present and start training for 200 epochs.
#This can take a few hours, depending on the availabel hardware.
#At the end of the procedure the trained backbone is stored, and can be
#used for downstream tasks (e.g. classification, image retrieval).
#The code can be easily customized on other datasets and backbones.

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import math

class MultiCIFAR10(torchvision.datasets.CIFAR10):
  """Override torchvision CIFAR10 for multi-image management.
  Similar class can be defined for other datasets (e.g. CIFAR100).
  Given K total augmentations, it returns a list of lenght K with
  different augmentations of the input mini-batch.
  """
  def __init__(self, K, **kwds):
    super().__init__(**kwds)
    self.K = K # tot number of augmentations
            
  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    pic = Image.fromarray(img)            
    img_list = list()
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
    else:
        img_list = img
    return img_list, target


class Conv4(torch.nn.Module):
    """A simple 4 layers CNN.
    Used as backbone.    
    """
    def __init__(self):
        super(Conv4, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = torch.nn.Sequential(
          torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(8),
          torch.nn.ReLU(),
          torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
          torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(16),
          torch.nn.ReLU(),
          torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
          torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),
          torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = torch.nn.Sequential(
          torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          torch.nn.AdaptiveAvgPool2d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.flatten(h)
        return h


class RelationalReasoning(torch.nn.Module):
  """Self-Supervised Relational Reasoning.
  Essential implementation of the method, which uses
  the 'cat' aggregation function (the most effective),
  and can be used with any backbone.
  """
  def __init__(self, backbone, feature_size=64):
    super(RelationalReasoning, self).__init__()
    self.backbone = backbone
    self.relation_head = torch.nn.Sequential(
                             torch.nn.Linear(feature_size*2, 256),
                             torch.nn.BatchNorm1d(256),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(256, 1))

  def aggregate(self, features, K):
    relation_pairs_list = list()
    targets_list = list()
    size = int(features.shape[0] / K)
    shifts_counter=1
    for index_1 in range(0, size*K, size):
      for index_2 in range(index_1+size, size*K, size):
        # Using the 'cat' aggregation function by default
        pos_pair = torch.cat([features[index_1:index_1+size], 
                              features[index_2:index_2+size]], 1)
        # Shuffle without collisions by rolling the mini-batch (negatives)
        neg_pair = torch.cat([
                     features[index_1:index_1+size], 
                     torch.roll(features[index_2:index_2+size], 
                     shifts=shifts_counter, dims=0)], 1)
        relation_pairs_list.append(pos_pair)
        relation_pairs_list.append(neg_pair)
        targets_list.append(torch.ones(size, dtype=torch.float32))
        targets_list.append(torch.zeros(size, dtype=torch.float32))
        shifts_counter+=1
        if(shifts_counter>=size): 
            shifts_counter=1 # avoid identity pairs
    relation_pairs = torch.cat(relation_pairs_list, 0)
    targets = torch.cat(targets_list, 0)
    return relation_pairs, targets

  def train(self, tot_epochs, train_loader):
    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
                  {'params': self.relation_head.parameters()}])                               
    BCE = torch.nn.BCEWithLogitsLoss()
    self.backbone.train()
    self.relation_head.train()
    for epoch in range(tot_epochs):
      # the real target is discarded (unsupervised)
      for i, (data_augmented, _) in enumerate(train_loader):
        K = len(data_augmented) # tot augmentations
        x = torch.cat(data_augmented, 0)
        optimizer.zero_grad()              
        # forward pass (backbone)
        features = self.backbone(x) 
        # aggregation function
        relation_pairs, targets = self.aggregate(features, K)
        # forward pass (relation head)
        score = self.relation_head(relation_pairs).squeeze()        
        # cross-entropy loss and backward
        loss = BCE(score, targets)
        loss.backward()
        optimizer.step()            
        # estimate the accuracy
        predicted = torch.round(torch.sigmoid(score))
        correct = predicted.eq(targets.view_as(predicted)).sum()
        accuracy = (100.0 * correct / float(len(targets)))
        
        if(i%100==0):
          print('Epoch [{}][{}/{}] loss: {:.5f}; accuracy: {:.2f}%' \
            .format(epoch+1, i+1, len(train_loader)+1, 
                    loss.item(), accuracy.item()))

def main():

  # Hyper-parameters of the simulation
  K = 4 # tot augmentations, in the paper K=32 for CIFAR10/100
  batch_size = 64 # 64 has been used in the paper
  tot_epochs = 1 # 200 has been used in the paper
  feature_size = 64 # number of units for the Conv4 backbone
  # Those are the transformations used in the paper
  normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], 
                                   std=[0.247, 0.243, 0.262]) # CIFAR10
  color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, 
                                        saturation=0.8, hue=0.2)
  rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
  rnd_gray = transforms.RandomGrayscale(p=0.2)
  rnd_rcrop = transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0), 
                                           interpolation=2)
  rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
  train_transform = transforms.Compose([rnd_rcrop, rnd_hflip,
                                        rnd_color_jitter, rnd_gray, 
                                        transforms.ToTensor(), normalize])
                                      
  backbone = Conv4() # simple CNN with 64 linear output units
  model = RelationalReasoning(backbone, feature_size)    
  train_set = MultiCIFAR10(K=K, root='data', train=True, 
                           transform=train_transform, 
                           download=True)
  train_loader = torch.utils.data.DataLoader(train_set, 
                                             batch_size=batch_size, 
                                             shuffle=True)                                                   
  model.train(tot_epochs=tot_epochs, train_loader=train_loader)
  torch.save(model.backbone.state_dict(), './backbone.tar')
  
if __name__ == "__main__":
    main()
