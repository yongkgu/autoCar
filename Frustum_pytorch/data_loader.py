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



# label_classes = {
#     (0, 0, 0):255, (111, 74, 0):255, (81, 0, 81):255, (128, 64, 128):0, (244, 35, 232):1, 
#     (250, 170, 160):255, (230, 150, 140):255, (70, 70, 70):2, (102, 102, 156):3, (190, 153, 153):3, 
#     (180, 165, 180):255, (150, 100, 100):255, (150, 120, 90):255, (153, 153, 153):5, (250, 170, 30):6, 
#     (220, 220, 0):7, (107, 142, 35):8, (152, 251, 152):9, (70, 130, 180):10, (220, 20, 60):11, 
#     (255, 0, 0):12, (0, 0, 142):13, (0, 0, 70):14, (0, 60, 100):15, (0, 0, 90):255, 
#     (0, 0, 110):255, (0, 80, 100):16, (0, 0, 230):17, (119, 11, 32):18
# }

# baseline = 0.54
# width_to_focal = dict()
# width_to_focal[1242] = 721.5377
# width_to_focal[1241] = 718.856
# width_to_focal[1224] = 707.0493
# width_to_focal[1238] = 718.3351
# width_to_focal[1226] = 711.3722

# class KITTI_Dataloader(Dataset):
#     __left = []
#     __disp = []
#     __seg = []
#     def __init__(self):
#         self.img_root = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/image/train/left/"
#         self.mask_root = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/seg/train/left/"
#         self.disp_root = "/Volumes/Crucial X8/BOAZ/KITTI_dataset/disp/train/left/"

#         for line in os.listdir(self.mask_root):
#             self.__left.append(self.img_root + line)
#             self.__disp.append(self.disp_root + line)
#             self.__seg.append(self.mask_root + line)
#     def get_transform(method=Image.BICUBIC, normalize=True):
#         transform_list = []
#         osize = [192, 480]
#         transform_list.append(transforms.Resize(osize, method))
#         transform_list += [transforms.ToTensor()]

#         if normalize:
#             transform_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
#         return transforms.Compose(transform_list)
#     def load_image2(self, image):
#         w, h = image.size
#         imx_t = (np.asarray(image))/256
#         imx_t = baseline * width_to_focal[w] / imx_t
#         nmimg = Image.fromarray(imx_t)
#         nmimg = nmimg.resize((480, 192))
#         imx_t = (np.asarray(nmimg))
#         return imx_t
#     def __len__(self):
#         return len(self.__left)
#     def __getitem__(self, index):
#         img1 = Image.open(self.__left[index]).convert('RGB')
#         img2 = Image.open(self.__disp[index])
#         img3 = Image.open(self.__seg[index]).convert('RGB')

#         transform = self.get_transform()
#         img1 = transform(img1)

#         img2 = self.load_image2(img2)
#         img2 = torch.Tensor(img2)

#         img3 = img3.resize((480, 192), PIL.Image.NEARSET)
#         img3 = np.array(img3)

#         one_hot_pred_ls = np.zeros((192, 480))

#         for j in range(192):
#             for k in range(480):
#                 one_hot_pred_ls[j][k] = label_classes[tuple(img3[j][k])]
        
#         one_hot_pred_ls = torch.from_numpy(one_hot_pred_ls).long()

#         input_dict = {'left_img':img1, 'disp_img':img2, 'mask':one_hot_pred_ls}
#         return input_dict


if __name__=='__main__':
    print("[ data_loader.py ]")

    train_dataset = Kitti()

    print(train_dataset)