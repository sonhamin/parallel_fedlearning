#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
#from sklearn import metrics

from os import fork
from threading import Lock



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, user_num, loss_func=nn.CrossEntropyLoss(), dataset=None, idxs=None):
        self.args = args
        self.user_num = user_num
        self.loss_func = loss_func
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net2 = net.to(self.args.device)
        net2.train()
        
        optimizer = torch.optim.SGD(net2.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        loss22 = self.loss_func
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net2.zero_grad()
                log_probs = net2(images)
                loss = loss22(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Local user {} -- Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.user_num, iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

