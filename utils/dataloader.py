from utils.customloader import CustomDataset, DatasetSplit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random

def get_dataloader(data='mnist', test_size=0.5, num_workers=0, batch_size=32, seed=42):
    
    if (data == 'mnist'):
        transform = transforms.Compose([ transforms.ToTensor(),  transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=transform)



        
    total_data_x, total_data_y = dataset_train.data, dataset_train.targets

    total_data_x = np.expand_dims(total_data_x, -1)
    total_data_y = np.expand_dims(total_data_y, -1)


    global_x, local_x, global_y, local_y = train_test_split(total_data_x, total_data_y, 
                                                            test_size=test_size, 
                                                            random_state=seed, 
                                                            stratify=total_data_y)
    
    global_train_set = CustomDataset(global_x, global_y, transform=transform)
    local_train_set = CustomDataset(local_x, local_y, transform=transform)
    
    
    global_train_loader = DataLoader(global_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    local_train_loader = DataLoader(global_train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return global_train_loader, local_train_loader, test_loader
    