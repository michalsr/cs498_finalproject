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
def get_transform(dataset_name,data_type,noise=False):
    if dataset_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        if noise:
            #add color jitter and random affine
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomAffine(90,shear=(0,30)),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:

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
        if noise:

            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomAffine(90,shear=(0,30)),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,

            ])
        else:
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
    def __init__(self,data_type,data_path,noise=False):
        super(Airplanes,self).__init__()
        self.data_path = data_path
        self.type = data_type
        self.noise = noise
        self.transform = get_transform('Airplanes',self.type,self.noise)
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
            data.append(self.data_path+'/'+img_num+'.jpg')
            labels.append(variant)
        final_labels = convert_labels(labels)
        labels = np.asarray(final_labels,dtype=np.int)
        return data, labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.open(img)

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        #target = torch.from_numpy(target)
        return img, target


class CIFAR10TrainVal():

    def __init__(self,data_type,noise=False,process_once=False):
        super(CIFAR10TrainVal, self).__init__()
        self.type = data_type
        self.noise= noise
        self.transform = get_transform('CIFAR10',self.type,self.noise)
        self.train_data, self.train_labels,self.val_data,self.val_labels = self.load_images(process_once)




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

    def load_images(self,process_once=True):
        'Process training data and save as images. Overwrite for each new dataset'
        # for cifar10
        #np.random.seed(self.seed)
        if os.path.exists('/home/michal5/cs498_finalproject/cifar10_val_data.txt'):
            process_once = False
        subfolders = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        val_dict = []
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
        if process_once:
            with open( '/home/michal5/cs498_finalproject/cifar10_val_data.txt') as f:
                val_values = json.load(f)


        for i, label in enumerate(labels):
            if not process_once:
                if random.random()>0.8:
                    val_data.append(data[i])
                    val_labels.append(label)
                    val_dict.append(i)


                else:
                    train_data.append(data[i])
                    train_labels.append(label)
            else:
                if i in val_values:
                    val_data.append(data[i])
                    val_labels.append(label)
                else:
                    train_data.append(data[i])
                    train_data.append(label)
        if not process_once:
            file_name = '/shared/rsaas/michal5/classes/498_dl/cs498_finalproject/' +'cifar10_val_data'+'.txt'
            with open(file_name, 'w+') as image_val:
                json.dump(val_dict, image_val)
        train_labels = np.asarray(train_labels, dtype=np.int)
        val_labels = np.asarray(val_labels,dtype=np.int)
        return train_data,train_labels,val_data,val_labels

class CIFAR10Test():

    def __init__(self, data_type,noise=False):
        super(CIFAR10Test, self).__init__()
        self.type = data_type
        self.noise=noise
        self.transform = get_transform('CIFAR10','test',self.noise)
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
        labels = np.asarray(labels, dtype=np.int)
        return data, labels

class PACS():
    def __init__(self, data_type,noise=False):
        super(PACS, self).__init__()
        self.type = data_type
        self.noise = noise
        self.transform = get_transform('PACS', self.type,self.noise)

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
                data.append('/data/common/pacs/'+img_num)
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
        img = Image.open(img)

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        # target = torch.from_numpy(target)
        return img, target



class ImageNetSubset():
    def __init__(self,data_type, second_class,noise=False):
        super(ImageNetSubset, self).__init__()
        self.type = data_type
        self.noise = noise
        self.transform = get_transform('PACS', self.type,self.noise)
        self.second_class = second_class
        self.data, self.labels = self.load_images()
    def load_images(self):
        data = []
        labels = []
        with open('/home/michal5/cs498_finalproject/image_net_labelled_'+self.second_class+'.txt') as i:
            image_labels = json.load(i)
        with open('/home/michal5/cs498_finalproject/image_net_train_list.json') as j:
            image_locations = json.load(j)
        for image,label in zip(image_locations,image_labels):
            data.append(image)
            labels.append(label)
        labels = np.asarray(labels, dtype=np.int)
        return data,labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img,target = self.data[idx],self.labels[idx]
        img = Image.open(img)
        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        return img,target
class ImageNetPair():
    def __init__(self,data_type,second_class,airplane_path=None,noise=False):
        super(ImageNetPair,self).__init__()
        self.type = data_type
        self.noise = noise
        self.second_class = second_class

        self.airplane_path = airplane_path
        if self.second_class == 'cifar10':
            self.transform = get_transform("CIFAR10",self.type)
        else:
            self.transform = get_transform('ImageNet',self.type)
        self.data, self.labels = self.load_images()
    def load_images(self):

        if self.type == 'test':
            cifar_10_class = CIFAR10Test('test')
        else:
            cifar_10_class = CIFAR10TrainVal(self.type)
        second_class_dict = {'airplane':Airplanes(self.type,self.airplane_path),'cifar10':cifar_10_class,'pacs':PACS(self.type)}
        second_class = second_class_dict[self.second_class]

        imagenet = ImageNetSubset(self.type,self.second_class)
        imagenet_data, imagenet_labels = imagenet.data, imagenet.labels
        if self.type == 'train':
            if self.second_class == 'cifar10':

                second_class_data, second_class_labels = second_class.train_data, second_class.train_labels
            else:
                second_class_data, second_class_labels = second_class.data, second_class.labels
            second_class_data.extend(imagenet_data)

            data = second_class_data

            second_class_labels = second_class_labels.tolist()
            imagenet_labels = imagenet_labels.tolist()
            second_class_labels.extend(imagenet_labels)
            labels = second_class_labels
        elif self.type == 'val':
            if self.second_class == 'cifar10':
                data = second_class.val_data
                labels = second_class.val_labels
            else:
                data = second_class.data
                labels = second_class.labels

        else:
            data = second_class.data
            labels = second_class.labels
        labels = np.asarray(labels, dtype=np.int)
        return data, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        if isinstance(img,np.ndarray):
            img = ToPILImage()(img)
        else:
            img = Image.open(img)
        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        return img, target

class ImageNetUnlabelled():

    def __init__(self):
        super(ImageNetUnlabelled, self).__init__()
        self.transform = get_transform('PACS','test')
        self.data, self.labels = self.load_images()

    def load_images(self):
        data = []
        labels = []
        with open('/home/michal5/cs498_finalproject/image_net_train_list.json') as i:
            image_values = json.load(i)
        for k in image_values:
            data.append(k)
            labels.append(0)
        labels = np.asarray(labels, dtype=np.int)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.open(img)
        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)
        return img, target