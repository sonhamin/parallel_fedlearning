import matplotlib

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from utils.customloader import CustomDataset, DatasetSplit
from utils.separate_into_classes import separate_into_classes
from utils.arguments import Args
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.smooth_crossentropy import SmoothCrossEntropyLoss
from utils.dataloader import get_dataloader, set_seed
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import copy
from sklearn.model_selection import train_test_split

loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
total_acc = []
args = Args()    

def global_train_epoch(epoch, batch_loss, net_glob, global_train_loader, sloss, optimizer):    
    net_glob.train()
    for batch_idx,(data, target) in enumerate(global_train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = net_glob(data)
        loss = sloss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(global_train_loader.dataset),
                       100. * batch_idx / len(global_train_loader), loss.item()))
        batch_loss.append(loss.item())
    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain loss:', loss_avg)
    
    return net_glob, loss_avg


def test_model(net_glob, test_loader, sloss):
    net_glob.eval()
    test_loss = 0
    correct = 0
    l = len(test_loader)
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_glob(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)

    #After fedlearning
    #print('After Federated Learning')
    print('\nTest set: Average loss: {:.5f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    
    

def train_global_model(net_glob, args, global_train_loader, test_loader, sloss, optimizer):
    
    
    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):

        batch_loss = []

        net_glob, loss_avg = global_train_epoch(epoch, batch_loss, net_glob, global_train_loader, sloss, optimizer)
        list_loss.append(loss_avg)

        test_model(net_glob, test_loader, sloss)


    net_glob.eval()
    
    path_checkpoint='./test'
    torch.save(net_glob.state_dict(), path_checkpoint)
    return




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
