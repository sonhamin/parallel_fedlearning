#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from models.Update import LocalUpdate
import torch
import time    
    
def multi_train_local_dif(q_l, q_w, arguemnts, idx, loss, data_loader, distribution,  model):
    
    print("training user: " + str(idx))
    
    local = LocalUpdate(args=arguemnts, user_num=idx, loss_func=loss, dataset=data_loader.dataset, idxs=distribution[idx])
    w, loss = local.train(net=model.to(arguemnts.device))
    q_l.put(loss)
    q_w.put(w)
    time.sleep(5)
    