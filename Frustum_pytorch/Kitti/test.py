# sample_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/velodyne/000000.bin"
# save_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/000000.las"

import pygame as pg
from OpenGL.GL import *
import open3d as o3d
import numpy as np

class App:

    def __init__(self):
        # initialize python
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()

        # initialize opengl
        glClearColor(0.1, 0.2, 0.2, 1) # R, G, B, alpha
        self.mainLoop()
    
    def mainLoop(self):

        running = True
        while(running):
            # check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                
            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT)
            pg.display.flip()

            # timing
            self.clock.tick(60)
        self.quit()
    
    def quit(self):
        pg.quit()

def load_pointcloud():
    pcd = o3d.io.read_point_cloud("../pointclouds/0004.ply")
    print(pcd)
    print("Pointcloud Center: " + str(pcd.get_center()))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors
if __name__=="__main__":
    myApp = App()