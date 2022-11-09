import open3d as o3d
import numpy as np

# "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/velodyne/000001.pcd"
# ply_point_cloud = o3d.data.PLYPointCloud()
point_clouds = o3d.io.read_point_cloud("/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/velodyne/000001.pcd")
# print(pc)
point_clouds_np = np.asarray(point_clouds.points)
print(point_clouds_np)
print(len(point_clouds_np))
o3d.visualization.draw_geometries([point_clouds],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

