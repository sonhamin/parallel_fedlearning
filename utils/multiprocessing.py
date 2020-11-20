#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from models.Update import LocalUpdate
import torch
import time
def work(idx, output):
    print("user: "+str(idx))
    output.put("asdfasdfasdf") 
    #return
    
    
    
def multi_train_local_dif(q_l, q_w, arguemnts, idx, loss, local_train_loader, non_iid,  model):
    
    #print("asdf11")
    
    print("training user: " + str(idx))
    #print("ye22: " + str(non_iid[idx]))
    
    local = LocalUpdate(args=arguemnts, user_num=idx, loss_func=loss, dataset=local_train_loader.dataset, idxs=non_iid[idx])
    
    #print("asdf22")
    
    w, loss = local.train(net=model.to(arguemnts.device))
    
    #print("asdf33")
    
    #lock.acquire()
    q_l.put(loss)
    q_w.put(w)
    #lock.release()
    
    time.sleep(5)
    
    
    #return w, loss
    #w_locals.append(copy.deepcopy(w))
    #loss_locals.append(copy.deepcopy(loss))    
    