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

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_labels(labels):
    label_dict = {}
    num = 0
    final_labels = []
    for i in labels:
        if i not in label_dict:
           label_dict[i] = num
           final_labels.append(num)
           num += 1
        else:
          final_labels.append(label_dict[i])
    return final_labels
def get_transform(dataset_name,data_type):
    if dataset_name == 'CIFAR10':
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

class Airplanes():
    def __init__(self,data_type,data_path):
        super(Airplanes,self).__init__()
        self.data_path = data_path
        self.type = data_type
        self.transform = get_transform('Airplanes',self.type)
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
        final_labels = convert_labels(labels)
        labels = np.asarray(final_labels,dtype=np.int)
        return data, labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.open(self.data_path+'/'+img+'.jpg')

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        #target = torch.from_numpy(target)
        return img, target


class CIFAR10TrainVal():

    def __init__(self,data_type):
        super(CIFAR10TrainVal, self).__init__()
        self.type = data_type
        self.transform = get_transform('CIFAR10',self.type)
        self.train_data, self.train_labels,self.val_data,self.val_labels = self.load_images()




    def __getitem__(self, index):
        if self.type == 'val':
            img,target = self.val_data[index], self.val_labels[index]
        else:


            img, target = self.train_data[index], self.train_labels[index]



        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)



        return img, target

    def __len__(self):
        if self.type == 'val':
            return len(self.val_labels)
        else:
            return len(self.train_labels)

    def load_images(self):
        'Process training data and save as images. Overwrite for each new dataset'
        # for cifar10
        #np.random.seed(self.seed)
        subfolders = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        val_dict = {}
        data = []
        labels = []
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        for folder in subfolders:
            dictionary = unpickle('/data/common/' + 'cifar-10-batches-py' + '/' + folder)
            images = dictionary[b'data']
            label_entry = dictionary[b'labels']

            data.append(images)
            labels+=label_entry
        data = np.concatenate(data)
        data = data.reshape((50000, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))
        labels = np.asarray(labels)
        for i, label in enumerate(labels):
            if random.random()>0.8:
                val_data.append(data[i])
                val_labels.append(label)
                if label not in val_dict:
                    val_dict[int(label)] = []
                val_dict[int(label)].append(int(label))

            else:
                train_data.append(data[i])
                train_labels.append(label)
        file_name = '/shared/rsaas/michal5/classes/498_dl/cs498_finalproject/' +'cifar10_val_data'+'.txt'
        with open(file_name, 'w+') as image_val:
            json.dump(val_dict, image_val)
        return train_data,train_labels,val_data,val_labels

class CIFAR10Test():

    def __init__(self, data_type):
        super(CIFAR10Test, self).__init__()
        self.type = data_type
        self.transform = get_transform('CIFAR10','test')
        self.data, self.labels = self.load_images()

    def __getitem__(self, index):


        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):

        return len(self.labels)

    def load_images(self):
        'Process training data and save as images. Overwrite for each new dataset'
        # for cifar10
        data = []
        labels = []
        test_data = unpickle('/data/common/cifar-10-batches-py/test_batch')
        test_images = test_data[b'data']
        test_labels = test_data[b'labels']
        test_images = test_images.reshape((len(test_data[b'data']), 3, 32, 32))
        test_images = test_images.transpose((0,2,3,1))
        for i,image in enumerate(test_images):
            data.append(image)
            labels.append(test_labels[i])
        return data, labels

class PACS():
    def __init__(self, data_type):
        super(PACS, self).__init__()
        self.type = data_type
        self.transform = get_transform('PACS', self.type)
        self.data, self.labels = self.load_images(self.type)

    def load_images(self, data_type):
        if data_type == 'train':
            label_file = ['datasets/pacs_text_files/art_painting_train_kfold.txt','datasets/pacs_text_files/cartoon_train_kfold.txt', 'datasets/pacs_text_files/sketch_train_kfold.txt']
        elif data_type == 'val':
            label_file = ['datasets/pacs_text_files/art_painting_crossval_kfold.txt','datasets/pacs_text_files/cartoon_crossval_kfold.txt', 'datasets/pacs_text_files/sketch_crossval_kfold.txt']
        else:
            label_file = ['datasets/pacs_text_files/art_painting_test_kfold.txt','datasets/pacs_text_files/cartoon_test_kfold.txt', 'datasets/pacs_text_files/sketch_test_kfold.txt']
        data = []
        labels = []
        for files in label_file:
            label_info = open(files)
            for line in label_info:
                line = line.split(' ')
                img_num = line[0]
                data.append(img_num)
                more_info = img_num.split('/')
                labels.append(more_info[1])
                #print(labels)
        final_labels = convert_labels(labels)
        labels = np.asarray(final_labels, dtype=np.int)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.open('/data/common/pacs/'+img)

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        # target = torch.from_numpy(target)
        return img, target










