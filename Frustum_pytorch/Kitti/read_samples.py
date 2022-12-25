import open3d as o3d
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 

def load_sample_pcd(train=True, idx=None):
    common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/"
    if train:
        common_path += "training/velodyne/"
    else:
        common_path += "testing/velodyne/"
    
    if not idx:
        print("Please input a number of point cloud file !!")
    
    pcd = str(idx)
    front = ""
    for i in range(6-len(pcd)):
        front+="0"
    
    file_path = common_path + front + pcd + ".pcd"
    
    point_clouds = o3d.io.read_point_cloud(file_path)
    point_clouds_np = np.asarray(point_clouds.points)
    print(point_clouds_np)
    print(len(point_clouds_np))
    o3d.visualization.draw_geometries([point_clouds])
                                    # zoom=0.3412,
                                    # front=[0.4257, -0.2125, -0.8795],
                                    # lookat=[2.6172, 2.0475, 1.532],
                                    # up=[-0.0694, -0.9768, 0.2024])
    return point_clouds

def load_sample_depth(train=True, file_idx=None, img_num=2, idx=None):
    common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_depth_velodyne/"
    
    if train:
        common_path += "train/"
    else:
        common_path += "val/"

    common_path += (os.listdir(common_path)[file_idx]+"/proj_depth/velodyne_raw/image_0"+str(img_num)+"/")

    depth = str(idx)
    front = ""
    for i in range(10-len(depth)):
        front+="0"
    file_path = common_path + front + depth + ".png"


    try:
        img = Image.open(file_path).convert("L")
        img_mat = np.asarray(img)
        print(img_mat)
        print(img_mat.shape)
        plt.imshow(img)
        plt.show()
    except:
        print("No file")

def load_sample_image(train=True, idx=None):
    common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_image_3/"
    if train:
        common_path += "training/image_3/"
    else:
        common_path += "testing/image_3/"
    
    if not idx:
        print("Please input a number of point cloud file !!")

    file = str(idx)
    front = ""
    for i in range(6-len(file)):
        front+="0"
    
    file_path = common_path + front + file + ".png"

    try:
        img = Image.open(file_path).convert("L")
        img_mat = np.asarray(img)
        print(img_mat)
        print(img_mat.shape)
        plt.imshow(img)
        plt.show()
    except:
        print("No file")
    
    return img

def load_sample_label():
    pass

    

    # print(file_path)



if __name__=='__main__':
    # PCD File
    # point_cloud = load_sample_pcd(train=True, idx=2)

    # IMAGE File
    img = load_sample_image(train=True, idx=0)


    # Depth File
    # load_sample_depth(train=True, file_idx=1, img_num=2, idx=5)