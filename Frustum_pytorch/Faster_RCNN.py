from __future__ import annotations
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms, datasets, models

import os
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

"""
To do
- 강아지와 오랑우탄 인형 Detection
"""
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
# Faster R-CNN with FPN
def get_model_object_detection(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def generate_box(obj):
    
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

adjust_label = 0

def generate_label(obj):

    if obj.find('name').text == "botong":

        return 1 + adjust_label

    elif obj.find('name').text == "orangutan":

        return 2 + adjust_label

    return 0 + adjust_label

def generate_target(file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return target
def plot_image_from_output(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 2 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)

    # plt.show()
# Sample Dataset
class BotongDataset(Dataset):
    def __init__(self, train=True, transforms=None):
        # self.datapath = "/Users/hyojin/Downloads/2D Detector for Faster R-CNN.v2i.voc/"
        self.datapath = "/Users/hyojin/Boaz_project/2D Detector for Faster R-CNN.v3i.voc/"
        if train:
            self.datapath += "train/"
        else:
            self.datapath += "test/"

        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(self.datapath + "imgs/")))
        self.labels = list(sorted(os.listdir(self.datapath + "labels/")))
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_path = self.labels[index]

        img = Image.open(self.datapath + "imgs/" + img_path).convert("RGB")
        label = generate_target(self.datapath + "labels/" + label_path)

        if self.transforms:
            img = self.transforms(img)

        return img, label

def collate_fn(batch):
    return tuple(zip(*batch))

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds


if __name__ == '__main__':
    # train on the GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Botong, Orangutan
    num_classes = 3

    # Model(Faster R-CNN)
    model = get_model_object_detection(num_classes)
    model.to(device)

    # # construct an optimizer & optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                                 momentum=0.9, weight_decay=0.0005)

    # num_epochs = 10

    data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
        transforms.ToTensor() # ToTensor : numpy 이미지에서 torch 이미지로 변경
    ])

    # dataset = BotongDataset(train=True, transforms=data_transform)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    dataset_test = BotongDataset(train=False, transforms=data_transform)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, collate_fn=collate_fn)

    # model.train()
    # for epoch in range(num_epochs):
    #     # train for one epoch, printing every 10 iterations
    #     print(">> Epoch", epoch)
    #     i = 0
    #     epoch_loss = 0
    #     for imgs, annotations in tqdm(data_loader):
    #         i += 1
            
    #         imgs = list(img.to(device) for img in imgs)
    #         annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

    #         loss_dict = model(imgs, annotations)
    #         losses = sum(loss for loss in loss_dict.values())        

    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step() 
    #         epoch_loss += losses
    #     print(f'epoch : {epoch+1}, Loss : {epoch_loss}')

    # torch.save(model.state_dict(),f'model_{num_epochs}.pt')
    # model.load_state_dict(torch.load(f'model_{num_epochs}.pt'))

    model.load_state_dict(torch.load("/Users/hyojin/Boaz_project/model_10.pt"))
    
    """
        Test로 평가
    """
    model.eval()
    with torch.no_grad():
        for imgs, annotations in data_loader_test:
            imgs = list(img.to(device) for img in imgs)

            pred = make_prediction(model, imgs, 0.5)
            print(pred)

            for idx in range(len(imgs)):
                plot_image_from_output(imgs[idx], annotations[idx])
                plt.title("Ground Truth")
                plt.show()

                plot_image_from_output(imgs[idx], pred[idx])
                plt.title("Prediction")
                plt.show()

    # _idx = 1
    # print("Target : ", annotations[_idx]['labels'])
    # plot_image_from_output(imgs[_idx], annotations[_idx])
    # print("Prediction : ", pred[_idx]['labels'])
    # plot_image_from_output(imgs[_idx], pred[_idx])
    # print("That's it!")