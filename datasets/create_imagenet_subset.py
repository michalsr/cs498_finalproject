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


def get_list_of_classes():
    "choose 50 out of 200 classes from ImageNet"
    class_list = os.listdir('/data/common/ILSVRC2012/ILSVRC2012_img_train/')
    final_classes = np.random.choice(class_list,50)
    return final_classes

def get_images_from_classes(class_list):
    "get 100 images from each of the 50 classes."
    train_list = []
    for c in class_list:
        possible_imgs = os.listdir('/data/common/ILSVRC2012/ILSVRC2012_img_train/'+c+'/')
        final_imgs = np.random.choice(possible_imgs,100)
        for f in final_imgs:
            prefix = '/data/common/ILSVRC2012/ILSVRC2012_img_train/'+c+'/'
            train_list.append(prefix+f)
    image_net_file = '/home/michal5/cs498_finalproject/ '+ 'image_net_train_list.json'
    with open(image_net_file, 'w+') as image_files:
        json.dump(train_list, image_files)


if __name__ == '__main__':
    classes = get_list_of_classes()
    get_images_from_classes(classes)

