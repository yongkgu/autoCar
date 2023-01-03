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

def load_sample_pcd(idx=None):
    if not idx:
        print("Please input a number of point cloud file !!")

    path = "/Volumes/CrucialX8/BOAZ/DATA/"+str(idx)+".pcd"
    
    point_clouds = o3d.io.read_point_cloud(path)
    point_clouds_np = np.asarray(point_clouds.points)
    print(point_clouds_np)
    print(len(point_clouds_np))
    o3d.visualization.draw_geometries([point_clouds])
                                    # zoom=0.3412,
                                    # front=[0.4257, -0.2125, -0.8795],
                                    # lookat=[2.6172, 2.0475, 1.532],
                                    # up=[-0.0694, -0.9768, 0.2024])
    return point_clouds

class ElevatorDataset(Dataset):
    def __init__(self):
        super().__init__()
    def __len__():
        return
    def __getitem__(self, index):





        return 


if __name__=="__main__":
    load_sample_pcd(90)
