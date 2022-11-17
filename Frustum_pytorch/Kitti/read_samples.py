import open3d as o3d
import numpy as np

def load_sample_pcd(train=True, num=None):
    common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/"
    if train:
        common_path += "training/velodyne/"
    else:
        common_path += "testing/velodyne/"
    
    if not num:
        print("Please input a number of point cloud file !!")
    
    pcd = str(num)
    front = ""
    for i in range(6-len(pcd)):
        front+="0"
    
    file_path = common_path + front + pcd + ".pcd"
    
    point_clouds = o3d.io.read_point_cloud(file_path)
    point_clouds_np = np.asarray(point_clouds.points)
    print(point_clouds_np)
    print(len(point_clouds_np))
    o3d.visualization.draw_geometries([point_clouds],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])





if __name__=='__main__':
    # point_clouds = o3d.io.read_point_cloud("/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/velodyne/000004.pcd")
    # point_clouds_np = np.asarray(point_clouds.points)
    # print(point_clouds_np)
    # print(len(point_clouds_np))
    # o3d.visualization.draw_geometries([point_clouds],
    #                                 zoom=0.3412,
    #                                 front=[0.4257, -0.2125, -0.8795],
    #                                 lookat=[2.6172, 2.0475, 1.532],
    #                                 up=[-0.0694, -0.9768, 0.2024])

    load_sample_pcd(train=True, num=300)