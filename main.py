
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
from models import load_models
from student_teacher import *
from utils import *
import hydra
from omegaconf import DictConfig, OmegaConf
def load_previous_weights(old_model,new_model):
    old_state_dict = old_model.state_dict()
    new_model_state_dict= new_model.state_dict()
    for k in old_state_dict:
        if k in new_model_state_dict and 'fc' not in k:
            new_model_state_dict[k] = old_state_dict[k]
    new_model.load_state_dict(new_model_state_dict)
    return new_model
@hydra.main(config_name='conf/conf_student_teacher.yaml')
def main(config):
    use_cuda = True
    num_classes = config.num_classes
    device = torch.device("cuda" if use_cuda else "cpu")
    log_dir = config.save_dir.format(**config)
    writer = SummaryWriter(log_dir=log_dir)
    # train resnet 18 on labelled set with augment
    print('Training teacher on labelled dataset')
    labelled_train_loader,labelled_val_loader, labelled_test_loader = get_datasets(config,noise=True)
    teacher_1_model = load_models.load_model('resnet18',config.num_classes,config.pretrained)
    optimizer = get_optimizer(config,teacher_1_model,is_student=False)
    if not os.path.exists(config.save_dir.format(**config)+'/model/'+'resnet18_teacher.pt'):
        train_teacher_labelled(teacher_1_model, labelled_train_loader, labelled_val_loader, labelled_test_loader,
                               optimizer, config, device, writer, 'resnet18_teacher')
    else:
        print('Loading teacher model')
        checkpoint = torch.load(config.save_dir.format(**config)+'/model/'+'resnet18_teacher.pt')
        teacher_1_model.load_state_dict(checkpoint['state_dict'])



    #
    # # predict labels using resnet18 from labelled set (make sure label numbers take into account the target dataset)
    num_ftrs = teacher_1_model.fc.in_features
    teacher_1_model.fc = torch.nn.Linear(num_ftrs,50)
    print('Teacher predicting labels')
    predict_labels(teacher_1_model,num_classes,device,config)

    # create joint dataset with labelled image net and target data set and augment
    student_1_model = load_models.load_model('resnet34',config.num_classes+50,config.pretrained)
    student_1_model = load_previous_weights(teacher_1_model,student_1_model)
    joint_train_loader, joint_val_loader, joint_test_loader = get_datasets(config,noise=True, joint=True)
    # train resnet 34 on labelled image net and target data set
    optimizer = get_optimizer(config,student_1_model,is_student=True)
    print('Training student 1 on joint dataset')


    if not os.path.exists(config.save_dir.format(**config) + '/model/' + 'resnet34_student.pt'):
        train_student_joint(student_1_model, joint_train_loader, joint_val_loader, joint_test_loader, optimizer, config,
                            device, writer, 'resnet34_student')
    else:
        print('Loading student 1 model')
        checkpoint = torch.load(config.save_dir.format(**config) + '/model/' + 'resnet34_student.pt')
        student_1_model.load_state_dict(checkpoint['state_dict'])
    # resnet 34 predicts labels on image net
    num_ftrs = student_1_model.fc.in_features
    student_1_model.fc = torch.nn.Linear(num_ftrs,50)
    print('Student 1 predicting labels')
    predict_labels(student_1_model,num_classes,device,config)

    # create joint dataset with new labels  (can add function to joint dataset to easily update labels)
    joint_train_loader, joint_val_loader, joint_test_loader = get_datasets(config,noise=True,joint=True)
    student_2_model = load_models.load_model('resnet50',config.num_classes+50,config.pretrained)
    if config.pretrained == 'moco':
        num_features = student_2_model.fc.in_features
        student_2_model.fc = nn.Linear(num_features,config.num_classes)
        finetune(student_2_model, labelled_train_loader, labelled_val_loader, labelled_test_loader, optimizer,
                 config,
                 device, writer, 'finetune teacher')
    num_features = student_2_model.fc.in_features
    student_2_model.fc = nn.Linear(num_features,config.num_classes+59)
    optimizer = get_optimizer(config,student_2_model,is_student=True)
    print('Training student 2 on joint dataset')
    if not os.path.exists(config.save_dir.format(**config)+'/model/'+'resnet50_student_1.pt'):
        train_student_joint(student_2_model,joint_train_loader,joint_val_loader,joint_test_loader,optimizer,config,device,writer,'resnet50_student_1')
    else:
        print('Loading student 2 model')
        checkpoint = torch.load(config.save_dir.format(**config)+'/model/'+'resnet50_student_1.pt')
        student_2_model.load_state_dict(checkpoint['state_dict'])
    num_ftrs = student_2_model.fc.in_features
    student_2_model.fc = torch.nn.Linear(num_ftrs, config.num_classes)
    labelled_train_loader, labelled_val_loader, labelled_test_loader = get_datasets(config, noise=False)
    optimizer = get_optimizer(config, student_2_model, is_student=True, finetune=True)
    print('Finetuning student 2')
    if not os.path.exists(config.save_dir.format(**config)+'/model/'+'resnet50_final.pt'):
        finetune(student_2_model, labelled_train_loader, labelled_val_loader, labelled_test_loader, optimizer, config,
                 device, writer, 'resnet50_final')
    else:
        checkpoint = torch.load(config.save_dir.format(**config)+'/model/'+'resnet50_final.pt')
        student_2_model.load_state_dict(checkpoint['state_dict'])





if __name__ == '__main__':
    main()








