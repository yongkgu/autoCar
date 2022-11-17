"""
    Load Datasets
"""
import PIL
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


import numpy as np
import os


class KittiDataset(Dataset):
    def __init__(self, train=True):
        self.common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/"
        if train:
            self.common_path += "training/velodyne"
        else:
            self.common_path += "testing/velodyne"
        
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass        


if __name__=='__main__':
    print("[ data_loader.py ]")

