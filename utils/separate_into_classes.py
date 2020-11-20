import numpy as np
import copy
def separate_into_classes(valid_y, args):
    unique, counts = np.unique(valid_y, return_counts=True)
    print(unique)
    print(counts)
    sorted_y = copy.deepcopy(valid_y)
    sorted_index_y = np.argsort(np.squeeze(sorted_y))

    valid_dist=[]

    for i in range(args.num_classes):
        print(i)
        valid_dist.append(sorted_index_y[sum(counts[:i]):sum(counts[:i+1])])

    
    return valid_dist