"""
    Load Datasets
    - X
        1. Point Cloud
        2. Image
    - Y
        1. VOC Label (2D)
        2. 3D Label
"""
import open3d as o3d

import PIL
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import os


class KittiDataset(Dataset):
    def __init__(self, train=True):
        common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/"
        if train:
            self.pcd_path = common_path + "train/object/"
            self.img_path = common_path + "train/image/"
            # self.label_path = self.common_path + ""
        else:
            self.pcd_path = common_path + "test/object"
            self.img_path = common_path + "test/image/"
        
        self.pcd_list = [file for file in os.listdir(self.pcd_path) if file.endswith(".pcd")]
        self.img_list = os.listdir(self.img_path)
        print(self.pcd_path)
    def __len__(self):
        return len(self.pcd_list)
    def __getitem__(self, index):
        img = np.asarray(Image.open(self.img_path + self.img_list[index].lstrip("._")).convert("L"))
        point_clouds = np.asarray(o3d.io.read_point_cloud(self.pcd_path + self.pcd_list[index]))
        return img, point_clouds
              


if __name__=='__main__':
    kitti = KittiDataset(train=True)
    print(len(kitti))
    
    kitti_test = KittiDataset(train=False)
    print(len(kitti_test))
