""" Implements an augmented SVHN Net with STNs to see the accuracy etc """ 

import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import STNModule

import numpy as np

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, config, kernel_size, stride, use_maxpool=True):
#         super(ConvBlock, self).__init__()
#         self._in_channels = in_channels 
#         self.ch_config = config 
#         self._ksz = kernel_size 
#         self._stride = stride
#         self.layers = self._make_layers()
    
#     def _make_layers(self):
#         layers = [] 
#         layers.append(nn.Sequential(
#                  nn.Conv2d(self._in_channels, self.ch_config[0], kernel_size=self._ksz, stride=self.stride, padding=1,bias=False),
#                 nn.BatchNx = F.max_pool2d(x, 2)ch_config[0])))
#         for i in range(1,x = F.max_pool2d(x, 2)h_config)):
#             layers.appendx = F.max_pool2d(x, 2)ial(
#                 nn.Conv2dx = F.max_pool2d(x, 2)nfig[i-1], self.ch_config[i], kernel_size=self._ksz, stride=self.stride, padding=1, bias=False),
#                 nn.BatchNx = F.max_pool2d(x, 2)ch_config[i])
#             ))
#             # layers.appex = F.max_pool2d(x, 2)d(self.ch_config[i-1], self.ch_config[i], kernel_size=self._ksz, stride=self.stride, padding=1, bias=False))
#             # layers.appex = F.max_pool2d(x, 2)Norm2d(self.ch_config[i]))
#         return layers
    
#     def forward(self, x):x = F.max_pool2d(x, 2)
#         for i in range(len(self.layers)):
#             x = F.relu(self.layers[i](x))
#             if i%2 == 0:
#                 x = F.maxpool2d(x, 2)
#             else:
#                 x = F.maxpool2d(x, 1)
#         return x




    
    

class BaseSVHNet(nn.Module):
    """
    Base SVHN Net to be trained
    """
    def __init__(self, in_channels, kernel_size, num_classes=10, use_dropout=False):
        super(BaseSVHNet, self).__init__()
        self._in_ch = in_channels 
        self._ksize = kernel_size 
        self.ncls = num_classes 
        self.dropout = use_dropout 
        self.drop_prob = 0.5
        self.stride = 1 

        self.conv1 = nn.Conv2d(self._in_ch, 32, kernel_size=self._ksize, stride=self.stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(32*4*4, 1024)
        self.fc2 = nn.Linear(1024, self.ncls)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        print(x.size())

        x = x.view(-1, 32*4*4)
        if self.dropout:
            x = F.dropout(self.fc1(x),p=0.5)
        else:
            x = self.fc1(x)
        x = self.fc2(x)
        return x
       
        

class STNSVHNet(nn.Module):
     def __init__(self, spatial_dim,in_channels, stn_kernel_size, kernel_size, num_classes=10, use_dropout=False):
        super(STNSVHNet, self).__init__()
        self._in_ch = in_channels 
        self._ksize = kernel_size 
        self._sksize = stn_kernel_size
        self.ncls = num_classes 
        self.dropout = use_dropout 
        self.drop_prob = 0.5
        self.stride = 1 
        self.spatial_dim = spatial_dim

        self.stnmod = STNModule.SpatialTransformer(self._in_ch, self.spatial_dim, self._sksize)
        self.conv1 = nn.Conv2d(self._in_ch, 32, kernel_size=self._ksize, stride=self.stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(128*4*4, 3092)
        self.fc2 = nn.Linear(3092, self.ncls)

        

     def forward(self, x):
        rois, affine_grid = self.stnmod(x)
        out = F.relu(self.conv1(rois))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = out.view(-1, 128*4*4)
        if self.dropout:
           out = F.dropout(self.fc1(out), p=0.5)
        else:
            out = self.fc1(out)
        out = self.fc2(out)
        return out

    





def loss_fn(outputs, labels):
    """
    Computes the loss between the predictions
    """
    return nn.CrossEntropyLoss()(outputs, labels)



def accuracy(output, labels):
    corr_output = np.argmax(output, axis=1)
    acc = np.sum(corr_output==labels)/float(labels.size)
    return acc 


metrics = {
    "accuracy":accuracy
}
    

        

