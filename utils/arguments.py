import torch
class Args:
    #federated arugments
    epochs=5
    num_users=10
    frac=0.1
    local_ep=1
    local_bs=10
    bs=128
    lr=0.01
    momentum=0.5
    split='user'
    
    
    #model arguments
    model='mnist'
    kernel_num=9
    kernel_sizes='3,4,5'
    norm='batch_norm'
    num_filters=32
    max_pool=True
    
    #other arguments
    #data='mnist'
    #iid='store_true'
    num_channels=1
    num_classes=10
    #stopping_rounds=10
    verbose='store_true'
    seed=1
    #all_clients='store_true'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    