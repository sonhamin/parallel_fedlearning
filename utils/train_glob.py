import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.customloader import CustomDataset, DatasetSplit
from models.Update import LocalUpdate
from models.Fed import FedAvg
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import copy


def test_model(net_glob, test_loader, sloss, args):
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

    test_loss /= len(test_loader)

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




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
