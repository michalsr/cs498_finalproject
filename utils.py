# based on https://github.com/Bjarten/early-stopping-pytorch
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np
from tqdm import tqdm
import json
from datasets.make_datasets import  *
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False


    def __call__(self, val_accuracy):



        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy> self.best_score:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0

def accuracy(output,target):
    """Calculates model accuracy

    Arguments:
        mdl {nn.model} -- nn model
        X {torch.Tensor} -- input data
        Y {torch.Tensor} -- labels/target values

    Returns:
        [torch.Tensor] -- accuracy
    """
    _, preds = torch.max(output, 1)
    n = preds.size(0)  # index 0 for extracting the # of elements
    #print(preds, 'preds')
    #print(target.data,'target')
    train_acc = torch.sum(preds == target.data)
    return train_acc.item()/n
def get_optimizer(config,model,is_student=False,finetune=False):
    if is_student:
        lr = config.student_lr
    elif finetune:
        lr = config.finetune_lr
    else:
        lr = config.teacher_lr
    if config.optimizer == 'sgd':
        optimizer_model = torch.optim.SGD(model.parameters(), lr,
                                          momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer_model = torch.optim.Adam(model.parameters(),lr)
    return optimizer_model

def get_datasets(config,noise=False,joint=False,return_dataset=False):
    if not joint:
        if config.labelled_dataset == 'airplane':
            train_data =Airplanes('train',config.airplane_path,noise)

            val_data = Airplanes('val',config.airplane_path,noise)

            test_data = Airplanes('test',config.airplane_path,noise)

        elif config.labelled_dataset == 'cifar10':
            train_data = CIFAR10TrainVal('train',noise)
            val_data = CIFAR10TrainVal('val',noise)
            test_data = CIFAR10Test('test',noise)
        elif config.labelled_dataset == 'pacs':
            train_data = PACS('train',noise)
            val_data = PACS('val',noise)
            test_data = PACS('test',noise)

    else:
        train_data = ImageNetPair('train',config.labelled_dataset,config.airplane_path,noise)

        val_data = ImageNetPair('val',config.labelled_dataset,config.airplane_path,noise)
        test_data =ImageNetPair('test',config.labelled_dataset,config.airplane_path,noise)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    if return_dataset:
        return train_loader, val_loader, test_loader,train_data,val_data,test_data
def create_data_loader(dataset,config,is_train):
    return torch.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=is_train)