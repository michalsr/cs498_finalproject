from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle
from scipy.io import loadmat
import json
import random
import torch.utils.data as data
from torchvision.transforms import ToPILImage
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.datasets
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from torchvision.transforms import ToPILImage
from torchvision import transforms
from PIL import Image
from skimage import io, transform
import shutil

def get_transform(dataset_name,data_type):
    if dataset_name == 'CIFAR10TRAIN':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif data_type != 'test':
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                         std=[0.229, 0.224,0.225])
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,

        ])
    elif data_type == 'test':
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])
        transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),normalize, ])

    return  transform

class Airplanes(data.dataset):
    def __init__(self,data_type,data_path):
        super(Airplanes,self).__init__()
        self.data_path = data_path
        self.type = data_type
        self.transforms = self.get_transforms('Airplanes',self.type)
        self.data, self.labels = self.load_images(self.type)

    def load_images(self,data_type):
        if data_type == 'train':
            label_file = 'datasets/airplane/images_variant_train.txt'
        elif data_type == 'val':
            label_file = 'datasets/airplane/images_variant_val.txt'
        else:
            label_file = 'datasets/airplane/images_variant_test.txt'
        data = []
        labels = []
        label_info = open(label_file)
        for line in label_info:
            line = line.split(' ')
            img_num, variant = line[0],line[1]
            data.append(img_num)
            labels.append(variant)
        return data, labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img, target = self.data[idx], self.label[idx]
        img = Image.open(self.data_path+'/'+self.data_type+'/'+img+'.jpg')

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)

        return img, target













