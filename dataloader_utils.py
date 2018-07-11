"""Utils for loading the dataset in pytorch.Currently supported - MNIST, CIFAR-10"""

import os 
import numpy as np 
import torch 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
from torch.utils.data.sampler import SubsetRandomSampler



def load_dataset(mode, name, dataset_params):
    train_loader, val_loader = None, None 
    train_transform = None 

    if dataset_params.aug == "on":
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))     
            ])
    else:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    
    if name == 'mnist':
        root = './data_mnist'
        train_set = dsets.MNIST(root=root, train=True, download=True, transform=train_transform)
        val_set = dsets.MNIST(root=root, train=False, download=True, transform=val_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=dataset_params.batch_size, shuffle=True, num_workers=dataset_params.num_workers, pin_memory=dataset_params.cuda)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=dataset_params.batch_size, shuffle=True, num_workers=dataset_params.num_workers, pin_memory=dataset_params.cuda)
    else:
        root = './data_cifar10'
        if dataset_params.aug == "on":
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
             train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        
        train_set = dsets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        val_set =  dsets.CIFAR10(root=root, train=False, download=True, transform=val_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=dataset_params.batch_size, shuffle=True, num_workers=dataset_params.num_workers, pin_memory=dataset_params.cuda)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=dataset_params.batch_size, shuffle=True, num_workers=dataset_params.num_workers, pin_memory=dataset_params.cuda)
    
    ch_dataset = None 
    if mode=='train':
        ch_dataset = train_loader
    else:
        ch_dataset = val_loader 
    
    return ch_dataset



def load_subsampled_dataset(mode, name, dataset_params):
    """
    Currently only for CIFAR10 dataset
    """
    if name=='mnist':
        raise ValueError("MNIST is not supported for random subset sampling")

    root = './data_cifar10'
    if dataset_params["aug"] == "on":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
        
    train_set = dsets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    val_set =  dsets.CIFAR10(root=root, train=False, download=True, transform=val_transform)

    len_train = len(train_set)
    indices = list(range(len_train))
    split = int(np.floor(dataset_params["subset_percent"]*len_train))
    np.random.seed(912) # Heh 
    np.random.shuffle(indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=dataset_params["batch_size"], shuffle=True, sampler=SubsetRandomSampler(indices[:split]), num_workers=dataset_params["num_workers"],pin_memory=dataset_params["cuda"])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=dataset_params["batch_size"], shuffle=False, num_workers=dataset_params["num_workers"],pin_memory=dataset_params["cuda"])

    ch_dataset = None 
    if mode == "train":
        ch_dataset = train_loader 
    else:
        ch_dataset = val_loader
    return ch_dataset 






        