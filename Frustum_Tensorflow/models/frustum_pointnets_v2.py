''' Frustum PointNets v2 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss


def get_instance_seg_v2_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v2 network.
- input
    - point_clou d : (B,N,4) 형태의 TF 텐서
        
        포인트 채널의 XYZ 및 강도가 있는 절두체 포인트 클라우드
        
        XYZ는 절두체 좌표에 있음
        
    - one_hot_vec : (B,3) 형태의 TF 텐서
        
        예측된 객체 유형을 나타내는 길이-3 벡터
        
    - is_training : TF boolean 스칼라
    - bn_decay : TF float 스칼라
    - end_points : dict
- output
    - logits : (B,N,2) 형태의 TF 텐서, bkg/clutter 및 개체에 대한 점수
    - end_points : dict
    '''

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,1])

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
        128, [0.2,0.4,0.8], [32,64,128],
        [[32,32,64], [64,64,128], [64,96,128]],
        is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
        32, [0.4,0.8,1.6], [64,64,128],
        [[64,64,128], [128,128,256], [128,128,256]],
        is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[128,256,1024],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l3_points = tf.concat([l3_points, tf.expand_dims(one_hot_vec, 1)], axis=2)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
        [128,128], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [128,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
        [128,128], is_training, bn_decay, scope='fa_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope='dp1')
    logits = tf_util.conv1d(net, 2, 1,
        padding='VALID', activation_fn=None, scope='conv1d-fc2')

    return logits, end_points

def get_3d_box_estimation_v2_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points):
    ''' 3D Box Estimation PointNet v2 network.
    - 3D 상자 추정 PointNet v2 네트워크
    - input
        - object_point_cloud : (B, M, C) 형태의 TF 텐서
            
            객체 좌표의 마스크된 point clouds
            
            XYZ는 절두체 좌표에 있음
            
        - one_hot_vec : (B,3) 형태의 TF 텐서
            
            예측된 객체 유형을 나타내는 길이-3 벡터
            
    - output
        - logits : (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4) 형태의 TF 텐서 상자 중심, 
        표제 빈 클래스 점수 및 residuals 포함 및 크기 cluster 점수 및 residuals
    ''' 
    # Gather object points
    batch_size = object_point_cloud.get_shape()[0].value

    l0_xyz = object_point_cloud
    l0_points = None
    # Set abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
        npoint=128, radius=0.2, nsample=64, mlp=[64,64,128],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
        npoint=32, radius=0.4, nsample=64, mlp=[128,128,256],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[256,256,512],
        mlp2=None, group_all=True,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, bn=True,
        is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')

    # 위 순서
# 🟡 **객체 포인트 수집**
# 🟡 **추상화 레이어 설정**
# 🟡 **Fully connected layers**
# - 1차원 배열의 형태로 평탄화된 행렬을 통해 이미지를 분류하는데 사용되는 계층
# 🟡 처음 3개의 숫자 : 상자 중심 좌표 (cx,cy,cz)**
# 🟡 다음 NUM_HEADING_BIN*2 : 표제 빈 클래스 점수 및 빈 residuals**
# 🟡 다음 NUM_SIZE_CLUSTER*4 : 상자 클러스터 점수 및 residuals**
    return output, end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    ''' Frustum PointNets model. The model predict 3D object masks and
모델은 3D 개체 마스크를 예측하고 frustum point clouds 의 객체에 대한 모델 경계 상자
- input
    - point_cloud : (B,N,4) 형태의 TF 텐서
        
        포인트 채널의 XYZ 및 강도가 있는 frustum 포인트 클라우드
        
        XYZ는 frustum 좌표에 있음
        
    - one_hot_vec : (B,3) 형태의 TF 텐서
        
        예측된 객체 유형을 나타내는 길이-3 벡터
        
    - is_training : TF boolean 스칼라
    - bn_decay: TF float 스칼라
- output
    - end_points : dict(이름 문자열에서 TF 텐서로 매핑)
    '''
    end_points = {}
    
    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v2_net(\
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = \
        point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net(\
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points)
    stage1_center = center_delta + mask_xyz_mean # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = \
        object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = get_3d_box_estimation_v2_net(\
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

    # 위 순서
# 🟡 **3D 인스턴스 분할 PointNet**
# 🟡 **Masking**
# 🟡 **마스킹된 포인트를 선택하고 마스킹된 포인트의 중심으로 변환**
# 🟡 **T-Net 및 좌표 변환**
# - point cloud가 rigid transformation에 invariant를 갖기 위함
# - T-Net을 사용해서 affine transformation matrix를 예측
#     - matrix를 입력 point cloud에 적용
#     - matrix가 point cloud에 affine transformation을 적용하여 rigid transformation에 강인해지도록 함
# 🟡 **객체 좌표에서 객체 포인트 클라우드 가져오기**
# 🟡 **Amodel Box 추정 PointNet**
# 🟡 **출력을 3D 상자 매개변수로 parse output**
    return end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,4))
        outputs = get_model(inputs, tf.ones((32,3)), tf.constant(True))
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.zeros((32,1024),dtype=tf.int32),
            tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,3)), outputs)
        print(loss)
