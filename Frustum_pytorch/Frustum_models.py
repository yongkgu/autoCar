"""
    Frustum Pointnets Model
"""
from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 2D Detection Module
def get_model_object_detection(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
# PointNet
class PointNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Inputs : B x n x 4 (x, y, z, intensity)
        # Outputs : B x n x 1 (Object Prob.)

        self.mlp1 = nn.Linear(4, 64)
        self.mlp2 = nn.Linear(64, 64)

        self.mlp3 = nn.Linear(64, 64)
        self.mlp4 = nn.Linear(64, 128)
        self.mlp5 = nn.Linear(128, 1024)

        self.maxpool = nn.MaxPool1d(kernel_size=1024)

        self.mlp6 = nn.Linear(1088+config["num_classes"], 512)
        self.mlp7 = nn.Linear(512, 256)
        self.mlp8 = nn.Linear(256, 128)
        self.mlp9 = nn.Linear(128, 128)
        self.mlp10 = nn.Linear(128, 2)

    def forward(self, point_cloud):
        outputs1 = self.mlp1(point_cloud)
        outputs1 = self.mlp2(outputs1)

        outputs2 = self.mlp3(outputs1)
        outputs2 = self.mlp4(outputs2)
        outputs2 = self.mlp5(outputs2)

        outputs2 = torch.cat(outputs2, outputs1, dim=-1)

        outputs2 = self.mlp6(outputs2)
        outputs2 = self.mlp7(outputs2)
        outputs2 = self.mlp8(outputs2)
        outputs2 = self.mlp9(outputs2)
        outputs2 = self.mlp10(outputs2)

        return outputs2        
# T-Net
class TNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Inputs : B x m x 3
        # Outputs : B x 3 (Residual Centers; x, y, z)

        self.mlp1 = nn.Linear(3, 128)
        self.mlp2 = nn.Linear(128, 256)
        self.mlp3 = nn.Linear(256, 512)
        
        self.maxpool = nn.MaxPool1d(kernel_size=512)

        self.fc1 = nn.Linear(512+config["num_classes"], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, point_cloud, input_labels):
        outputs = self.mlp1(point_cloud)
        outputs = self.mlp2(outputs)
        outputs = self.mlp3(outputs)

        outputs = self.maxpool(outputs)
        outputs = torch.cat(outputs, input_labels, dim=3)

        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)

        return outputs
# Amodal 3D Box Estimator
class Box_estimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Inputs : B x m x 3
        # Outputs : B x (3+4NS+2NH) (Box Parameters)

        self.mlp1 = nn.Linear(3, 128)
        self.mlp2 = nn.Linear(128, 128)
        self.mlp3 = nn.Linear(128, 256)
        self.mlp4 = nn.Linear(256, 512)

        self.maxpool = nn.MaxPool1d(kernel_size=512)

        self.fc1 = nn.Linear(512+config["num_classes"], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, config["num_box_parameters"])

    def forward(self, point_cloud, input_labels):
        outputs = self.mlp1(point_cloud)
        outputs = self.mlp2(outputs)
        outputs = self.mlp3(outputs)
        outputs = self.mlp4(outputs)

        outputs = torch.cat(outputs, input_labels, dim=-1)

        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)

        return outputs
# Final Model
class FrustumPointNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.Detector = get_model_object_detection(config["num_classes"])
        self.PointNet = PointNet(config)
        self.TNet = TNet(config)
        self.Box_estimator = Box_estimator(config)

    def forward(self, images, point_clouds):
        pass


def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load Model
    model = FrustumPointNet(config).to(device)
    # Load Dataset

    # Loss Func. & Optimizer
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train Model
    model.train()
    for epoch in range(config["epochs"]):
        costs = []


def test(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Model
    model = FrustumPointNet(config).to(device)
    # Load Dataset



    # Test Model
    model.eval()
    with torch.no_grad():
        pass





if __name__=='__main__':
    print("[ Frustum_models.py ]")

    config = {
        "mode" : "train",
        "batch_size" : 8,
        "epochs" : 200,
        "learning_rate" : 1e-4,
        "early_stopping" : 20,
        "optimizer" : "adam",
        "num_classes" : 2,
        "num_box_parameters" : 10
    }

    # if config["mode"]=="train":
    #     train(config)
    # else:
    #     test(config)


    TNet_model = TNet(config)

    input3 = torch.rand((16, 60, 3))
    input_labels = torch.rand((16, 2))

    TNet_model.train()

    print(TNet_model(input3, input_labels))
    # print(Box_estimator_model(input3, input_labels))