"""
    Pointnets Model
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def mlpblock(in_channels, out_channels, act_f=True):
    # MLP Block with Batch Normalization and ReLU
    # : 데이터가 n x dim 형태로 들어오기 때문에 Linear가 아닌 Conv1d 사용
    layers = [
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
    ]
    if act_f:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
def fcblock(in_channels, out_channels, dropout_rate=None):
    # Linear FC Block with DropOut, Batch Normalization, and ReLU
    layers = [
        nn.Linear(in_channels, out_channels),
    ]
    if dropout_rate is not None:
        layers.append(nn.Dropout(p=dropout_rate))
    layers += [
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ]
    return nn.Sequential(*layers)
"""
    T-Net : for Affine Transformation
    - input : [ B, n, k ]   * k = 3 or 64
    - output : [ B, k, k ]
"""
class TNet(nn.Module):
    def __init__(self, dim=64):
        super(TNet, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            # n x dim -> n x 64 -> n x 128 -> n x 1,024
            mlpblock(dim, 64),
            mlpblock(64, 128),
            mlpblock(128, 1024)
        )
        self.fc = nn.Sequential(
            # 1 x 1,024 -> 1 x 512 -> 1 x 256 -> dim x dim
            fcblock(1024, 512),
            fcblock(512, 256),
            nn.Linear(256, dim*dim)
        )
        
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 2, keepdim=True)[0] # MaxPool
        x = x.view(-1, 1024)

        x = self.fc(x)

        idt = torch.eye(self.dim, dtype=torch.float32).flatten().unsqueeze(0).repeat(x.size()[0], 1)
        idt = idt.to(x.device)
        x = x + idt
        x = x.view(-1, self.dim, self.dim)
        return x
class PointNetCls(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetCls, self).__init__()

        self.tnet = TNet(dim=3)     # T-Net 1
        self.mlp1 = nn.Sequential(
            mlpblock(3, 64),
            mlpblock(64, 64) # MLP : n x 3 -> n x 64 -> n x 64
        )
        self.tnet_feature = TNet(dim=64)        # T-Net 2

        self.mlp2 = nn.Sequential(              # MLP : n x 64 -> n x 128 -> n x 1,024
            mlpblock(64, 64),
            mlpblock(64, 128),
            mlpblock(128, 1024, act_f=False)   
        )

        self.mlp3 = nn.Sequential(              # FC : 1 x 1,024 -> 1 x 512 -> 1 x 256 -> 1 x k (for Classification)
            fcblock(1024, 512),
            fcblock(512, 256, dropout_rate=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        :input size: [ B, n_points, 3 ]
        :output size: [ B, num_classes ]
        """
        x = x.transpose(2, 1) # [ B, 3, n_points ]
        trans = self.tnet(x) # [ B, 3, 3 ]
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1) # [ B, 3, n_points ]
        x = self.mlp1(x) # [ B, 64, n_points ]

        trans_feat = self.tnet_feature(x) # [ B, 64, 64 ]
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1) # [ B, 64, n_points ] 이거 써야함

        x = self.mlp2(x) # [ B, 1024, n_points ]
        x = torch.max(x, 2, keepdim=False)[0] # [ B, 1024 ] (global feature)

        x = self.mlp3(x) # [ B, num_classes ]

        return x, trans_feat
"""
    PointNet for Segmentation
"""
class PointNetSG(nn.Module):
    def __init__(self, num_categories=5):
        super(PointNetSG, self).__init__()

        self.tnet = TNet(dim=3)     # T-Net 1
        self.mlp1 = nn.Sequential(
            mlpblock(3, 64),
            mlpblock(64, 64) # MLP : n x 3 -> n x 64 -> n x 64
        )

        self.tnet_feature = TNet(dim=64)        # T-Net 2

        self.mlp2 = nn.Sequential(              # MLP : n x 64 -> n x 128 -> n x 1,024
            mlpblock(64, 64),
            mlpblock(64, 128),
            mlpblock(128, 1024, act_f=False)   
        )

        self.mlp3 = nn.Sequential(
            mlpblock(1088, 512),
            mlpblock(512, 256),
            mlpblock(256, 128)
        )

        self.mlp4 = nn.Sequential(
            mlpblock(128, 128),
            mlpblock(128, num_categories)
        )

    def forward(self, x):
        """
        :input size: [ B, n_points, 3 ]
        :output size: [ B, n_points, num_categories ]
        """
        B, N, _ = x.shape

        x = x.transpose(2, 1) # [ B, 3, n_points ]
        trans = self.tnet(x) # [ B, 3, 3 ]
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1) # [ B, 3, n_points ]
        x = self.mlp1(x) # [ B, 64, n_points ]

        trans_feat = self.tnet_feature(x) # [ B, 64, 64 ]
        x_feature = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1) # [ B, 64, n_points ] 이거 써야함

        x_global = self.mlp2(x_feature) # [ B, 1024, n_points ]
        x_global = torch.max(x_global, 2, keepdim=False)[0] # [ B, 1024 ] (global feature)
        x_global = torch.ones(B, N, 1) @ x_global.view(-1, 1, 1024)   # [ B, n_points, 1024 ]
        x_global = x_global.transpose(2, 1) # [ B, 1024, n_points ]

        x = torch.cat([x_feature, x_global], dim=1)

        x = self.mlp3(x) # [ B, 128, n_points ]

        x = self.mlp4(x) # [ B, num_categories, n_points ]

        return x.transpose(2, 1)


if __name__=='__main__':
    print("[ PointNet_models.py ]")
    sample = torch.rand((16, 7, 3))
    model = PointNetSG()
    model.train()
    pred = model(sample)
    print(pred.shape)