clear
imagePath = fullfile(toolboxdir('lidar'),'lidardata','lcc','vlp16','images');
ptCloudPath = fullfile(toolboxdir('lidar'),'lidardata','lcc','vlp16','pointCloud');
cameraParamsPath = fullfile(imagePath,'calibration.mat');

% Load camera intrinscs.
intrinsic = load(cameraParamsPath);                  

% Load images using imageDatastore.
imds = imageDatastore(imagePath);                    
imageFileNames = imds.Files;

% Load point cloud files.
pcds = fileDatastore(ptCloudPath,'ReadFcn',@pcread); 
ptCloudFileNames = pcds.Files;

% Square size of the checkerboard in mm.
squareSize = 81;                                     

% Set random seed to generate reproducible results.
rng('default')

% Extract checkerboard corners from the images.
[imageCorners3d,checkerboardDimension,dataUsed] = ...
    estimateCheckerboardCorners3d(imageFileNames,intrinsic.cameraParams,squareSize);

% Remove the unused image files.
imageFileNames = imageFileNames(dataUsed);           

% Filter the point cloud files that are not used for detection.
ptCloudFileNames = ptCloudFileNames(dataUsed);

% Extract ROI from the detected checkerboard image corners.
roi = helperComputeROI(imageCorners3d,5);

% Extract checkerboard plane from point cloud data.
[lidarCheckerboardPlanes,framesUsed,indices] = detectRectangularPlanePoints( ...
    ptCloudFileNames,checkerboardDimension,RemoveGround=true,ROI=roi);
imageCorners3d = imageCorners3d(:,:,framesUsed);

% Remove ptCloud files that are not used.
ptCloudFileNames = ptCloudFileNames(framesUsed);

% Remove image files that are not used.
imageFileNames = imageFileNames(framesUsed);

[tform,errors] = estimateLidarCameraTransform(lidarCheckerboardPlanes, ...
    imageCorners3d,intrinsic.cameraParams);
helperFuseLidarCamera(imageFileNames,ptCloudFileNames,indices, ...
    intrinsic.cameraParams,tform);