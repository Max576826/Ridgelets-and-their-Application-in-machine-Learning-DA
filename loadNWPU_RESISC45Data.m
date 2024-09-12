function [XTrain, YTrain, XTest, YTest] = loadNWPU_RESISC45Data(dataFolder, trainRatio)
    % Load NWPU-RESISC45 dataset from the folder structure and split into training and test sets.
    % Images are loaded in their original RGB format.
    %
    % Inputs:
    % - dataFolder: folder containing the NWPU-RESISC45 dataset with subfolders for each class.
    % - trainRatio: the ratio of the dataset to use for training (e.g., 0.8 for 80% training data).
    %
    % Outputs:
    % - XTrain: RGB training images.
    % - YTrain: Training labels.
    % - XTest: RGB test images.
    % - YTest: Test labels.

    if nargin < 2
        trainRatio = 0.8; % Default training ratio if not provided
    end

    % Load images and labels
    imageDS = imageDatastore(dataFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    % Get labels from folder names
    labels = imageDS.Labels;
    
    % Split dataset into training and testing sets
    [trainDS, testDS] = splitEachLabel(imageDS, trainRatio, 'randomized');
    
    % Read the training images
    numTrain = numel(trainDS.Files);
    imgSize = size(readimage(trainDS, 1));  % Get the size of the first image
    XTrain = zeros([imgSize, numTrain], 'like', readimage(trainDS, 1));  % Preallocate array for training images
    YTrain = zeros(numTrain, 1, 'single');
    
    for i = 1:numTrain
        img = readimage(trainDS, i);
        XTrain(:, :, :, i) = imresize(img, [imgSize(1), imgSize(2)]);  % Resize to maintain consistency
        YTrain(i) = single(trainDS.Labels(i));
    end
    
    % Read the testing images
    numTest = numel(testDS.Files);
    XTest = zeros([imgSize, numTest], 'like', readimage(testDS, 1));  % Preallocate array for testing images
    YTest = zeros(numTest, 1, 'single');
    
    for i = 1:numTest
        img = readimage(testDS, i);
        XTest(:, :, :, i) = imresize(img, [imgSize(1), imgSize(2)]);  % Resize to maintain consistency
        YTest(i) = single(testDS.Labels(i));
    end
    
    % Convert labels to categorical
    YTrain = categorical(YTrain);
    YTest = categorical(YTest);
end
