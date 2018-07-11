""" A collection of functions for visualization and loading sample data for testing """
import torch
import torchvision.transforms as transforms 
import numpy as np 
import os 
from PIL import Image 

from array import array
import struct 



def load_mnist_images(fname):
    with open(fname, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))

        img = array("B", f.read())
    
        return img, rows, cols, size 



def get_torch_image(img, rows, cols, size, idx):
    img  = np.array(array.tolist(img), dtype=np.uint8)
    img = img.reshape(10000, rows, cols)
    timg = img[idx,:,:]
    return timg


