import struct
import sys
from struct import unpack
import os

def readBinFile(fname):
    x = []
    y = []
    z = []
    try:
        with open(fname, 'rb') as fp:
            while True:
                val1 = unpack('<1f', fp.read(4))
                val2 = unpack('<1f', fp.read(4))
                val3 = unpack('<1f', fp.read(4))
                val4 = unpack('<1f', fp.read(4))
                x.append(val1[0])
                y.append(val2[0])
                z.append(val3[0])
                # print(val1[0], val2[0], val3[0])
    except Exception as e:
        print(e)
        pass
    finally:
        fp.close()  # 파일 닫기
    return x,y,z

def writePCDFile(fname,x,y,z):
    numPoints= len(x)
    with open(fname, 'w') as fp:
        fp.write("VERSION 0.7\n")
        fp.write("FIELDS x y z\n")
        fp.write("SIZE 4 4 4\n")
        fp.write("TYPE F F F\n")
        fp.write("WIDTH "+str(numPoints)+"\n")
        fp.write("HEIGHT 1\n")
        fp.write("POINTS "+str(numPoints)+"\n")
        fp.write("DATA ascii\n")
        for index in range(numPoints):
            txtLine = "{} {} {}\n".format(x[index],y[index],z[index] )
            fp.write(txtLine)
        pass
def translate_bin2pcd(train=True):
    common_path = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/"
    if train:
        common_path += "training/velodyne/"
    else:
        common_path += "testing/velodyne/"

    for bin_file in os.listdir(common_path):
        bin_path = common_path+bin_file
        pcd_path = bin_path.replace(".bin", ".pcd")
        print(bin_path)
        print(pcd_path)
        x, y, z = readBinFile(bin_path)
        writePCDFile(pcd_path, x, y, z)




if __name__ == "__main__":
    # fname1 ="/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/velodyne/000001.bin"
    # fname2 ="/Volumes/Crucial X8/BOAZ/KITTI_dataset/data_object_velodyne/training/velodyne/000001.pcd"
    # x,y,z =readBinFile(fname1)
    # writePCDFile(fname2, x, y, z)
    
    translate_bin2pcd(train=True)
    translate_bin2pcd(train=False)