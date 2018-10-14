import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data

class AdvCifar10(data.Dataset):
    """API to load in dataset (already in format saved as numpy array) into training queue"""
    
    """
    Args:
        root_data = location of numpy array for images
        root_lbls = location of numpy array for labels
    """
    def __init__(self, root_data, root_lbls):
        self.data = np.load(root_data)
        self.labels = np.load(root_lbls)
        
    def __getitem__(self, index):
        img_, target = self.data[index], self.labels[index]
        img_ = torch.Tensor(img_)
        return img_, target
    
    def __len__(self):
        return len(self.data)