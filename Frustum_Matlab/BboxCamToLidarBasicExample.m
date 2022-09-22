%% Transfer Bounding Box from Image to Point Cloud
% 
% 
% Load ground truth data from a MAT-file into the workspace. Extract the image, 
% point cloud data, and camera intrinsic parameters from the ground truth data.

dataPath = fullfile(toolboxdir('lidar'),'lidardata','lcc','bboxGT.mat');
gt = load(dataPath);
im = gt.im;
pc = gt.pc;
intrinsics = gt.cameraParams;
%% 
% Extract the camera to lidar transformation matrix from the ground truth data.

tform = gt.camToLidar;
%% 
% Extract the 2-D bounding box information.

bboxImage = gt.box;
%% 
% Display the 2-D bounding box overlaid on the image.

annotatedImage = insertObjectAnnotation(im,'Rectangle',bboxImage,'Vehicle');
figure
imshow(annotatedImage)
%% 
% Estimate the bounding box in the point cloud.

[bboxLidar,indices] = ...
bboxCameraToLidar(bboxImage,pc,intrinsics,tform,'ClusterThreshold',1);
%% 
% Display the 3-D bounding box overlaid on the point cloud.

figure
pcshow(pc)
xlim([0 50])
ylim([0 20])
showShape('cuboid',bboxLidar,'Opacity',0.5,'Color','green')
%% 
% _Copyright 2020 The MathWorks, Inc._