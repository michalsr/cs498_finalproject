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
from utils import EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf

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

def get_optimizer(config,model):
    if config.optimizer == 'sgd':
        optimizer_model = torch.optim.SGD(model.parameters(), config.lr,
                                          momentum=config.momentum, weight_decay=config.weight_decay)
    return optimizer_model


def test(model, test_loader, device, writer, config, epoch,test=False):
    if test:
        eval_type = 'test'
    else:
        eval_type = 'val'

    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar(eval_type+ ' loss', test_loss, epoch)
    writer.add_scalar( eval_type+ ' accuracy', accuracy, epoch)

    tqdm.write('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(eval_type,
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    fname = os.path.join(config.save_dir.format(**config), eval_type+'_results/' + 'epoch_' + str(epoch) + '.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        json.dump({eval_type+'_loss': test_loss, eval_type+'_accuracy': accuracy}, f)
        tqdm.write(f'Saved {eval_type} results to {fname}')
    return accuracy
def train(train_loader,model,optimizer,epoch,device,writer,config):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    train_accuracy = []

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        #print(inputs)
        #print(targets)
        #model.cuda()
        model.train()
        # inputs.cuda()
        # targets.cuda()
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        cross_entropy = nn.CrossEntropyLoss()
        cost = cross_entropy(outputs, targets)
        prec_train = accuracy(outputs, targets)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        train_loss += cost.item()
        train_accuracy.append(prec_train)



        if (batch_idx + 1) % 50 == 0:
            tqdm.write('Epoch: [%d/%d]\t'
                       'Iters: [%d/%d]\t'
                       'Loss: %.4f\t'

                       'Prec@1 %.2f\t'
                       % (
                           (epoch + 1), config.epochs, batch_idx + 1, len(train_loader.dataset) / config.batch_size,
                           (train_loss / (batch_idx + 1)),
                           prec_train))
            writer.add_scalar('Train Loss', train_loss / (batch_idx + 1), (batch_idx + 1) * (epoch + 1))
            writer.add_scalar('Train Accuracy', np.mean(train_accuracy), (batch_idx + 1) * (epoch + 1))
        if (batch_idx+1) %50 == 0:

            model_checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            model_save_dir = config.save_dir.format(**config) + '/model/'
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            torch.save(model_checkpoint, model_save_dir + 'acc:' + str(np.mean(train_accuracy)) + '.pt')


def get_datasets(config):
    if config.dataset_name == 'airplane':
        train_data =Airplanes('train',config.data_path)

        val_data = Airplanes('val',config.data_path)

        test_data = Airplanes('test',config.data_path)

    elif config.dataset_name == 'cifar10':
        train_data = CIFAR10TrainVal('train')
        val_data = CIFAR10TrainVal('val')
        test_data = CIFAR10Test('test')
    elif config.dataset_name == 'pacs':
        train_data = PACS('train')
        val_data = PACS('val')
        test_data = PACS('test')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


@hydra.main(config_name='conf/conf.yaml')
def main(config):
    use_cuda = True
    dataset_classes = config.num_classes
    model, _ = load_models.load_model(config.model_type,dataset_classes,config.pretrained)
    device = torch.device("cuda" if use_cuda else "cpu")
    log_dir = config.save_dir.format(**config)
    writer = SummaryWriter(log_dir=log_dir)
    train_loader, val_loader, test_loader = get_datasets(config)

    #optim = get_optimizer(config,model)
    optim = torch.optim.SGD(model.parameters(),config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,5)
    early_stopping = EarlyStopping()
    model.cuda()
    initial = test(model,test_loader,device,writer,config,1,True)
    # / shared / rsaas / michal5 / classes / 498
    # _dl / cs498_finalproject / outputs / dataset_name = airplane / pretrained = True, self_train = False / opt = sgd, bs = 64, lr = 0.01 / model / acc:0.71875.pt
    #model = load_model('/shared/rsaas/michal5/classes/498_dl/cs498_finalproject/outputs/dataset_name=airplane/pretrained=True,self_train=True/opt=sgd,bs=64,lr=0.01'+'/model/'+'acc:0.7185.pt',model)
    epoch_reached = 0
    #model = load_model(config.save_dir.format(**config)+'/model/'+'acc:0.71875.pt',model)
    #test_acc = test(model,test_loader,device,writer,config,1,True)

    if config.train:
        for epoch in range(config.epochs):
            train(train_loader,model,optim,epoch,device,writer,config)

            val_acc = test(model, test_loader, device, writer, config, epoch, test=False)
            early_stopping(val_acc)
            if early_stopping.early_stop:
                epoch_reached = epoch
                break
            scheduler.step()
    test_acc = test(model, test_loader, device, writer, config, epoch_reached, test=True)
    fname = os.path.join(config.save_dir.format(**config), 'best_accuracy' + '.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        json.dump({'test_accuracy': test_acc}, f)
        tqdm.write(f'Saved accuracy results to {fname}')
    print('test accuracy:', test_acc)

def load_model(model_path,model):
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['state_dict'])

    return model


if __name__ == '__main__':
    main()

