if ~isempty(ver('parallel'))
    % Create a parallel pool with 8 workers
    parpool('local', 14);
end
imsize=64;
% Load CIFAR-10 dataset
%[XTrain, YTrain, XTest, YTest] = loadCIFAR10Data();
[XTrain, YTrain, XTest, YTest]=loadNWPU_RESISC45Data("C:\Users\MaxSc\Desktop\dataset\NWPU-RESISC45");
% Convert labels to categorical if they are not already


% Select only 10 categories (adjust according to your dataset's class names)
selectedClasses = {'1', '8', '15', '18', '19', '21', '28', '30', '33', '35'}; 

% Filter training and test data for the selected classes
isTrainSelected = ismember(YTrain, selectedClasses);  % Filter training data
isTestSelected = ismember(YTest, selectedClasses);    % Filter test data

% Check how many samples were selected
disp(['Training samples selected: ', num2str(sum(isTrainSelected))]);
disp(['Test samples selected: ', num2str(sum(isTestSelected))]);

% Filter the data for only selected classes
XTrain = XTrain(:, :, :, isTrainSelected);
YTrain = YTrain(isTrainSelected);
XTest = XTest(:, :, :, isTestSelected);
YTest = YTest(isTestSelected);

% Check if the filtered data is non-empty
if isempty(XTrain) || isempty(YTrain)
    error('Filtered training data is empty. Check the class names in selectedClasses.');
end

% Check number of unique classes in the filtered data
disp('Number of unique classes in YTrain:');
disp(numel(categories(YTrain)));  % Verify the number of unique classes in YTrain

% Preallocate arrays for grayscale images and resize them to 128x128
XTrainGray = zeros(imsize, imsize, 1, size(XTrain, 4), 'single'); % Preallocate for grayscale training images as 'single'
XTestGray = zeros(imsize, imsize, 1, size(XTest, 4), 'single');    % Preallocate for grayscale test images as 'single'

% Convert training images to grayscale and resize to 128x128
for i = 1:size(XTrain, 4)
    grayImage = rgb2gray(squeeze(XTrain(:, :, :, i)));  % Convert to grayscale
    XTrainGray(:, :, 1, i) = imresize(grayImage, [imsize, imsize]);  % Resize to 128x128
end

% Convert test images to grayscale and resize to 128x128
for i = 1:size(XTest, 4)
    grayImage = rgb2gray(squeeze(XTest(:, :, :, i)));  % Convert to grayscale
    XTestGray(:, :, 1, i) = imresize(grayImage, [imsize, imsize]);  % Resize to 128x128
end

% Normalize the pixel values between 0 and 1 (data should be in 'single' type)
XTrainGray = single(XTrainGray / 255);
XTestGray = single(XTestGray / 255);
YTrain = categorical(YTrain);
YTest = categorical(YTest);
layers = [
    imageInputLayer([imsize imsize 1])    % Input layer for grayscale 32x32 images
    
    RidgeletConvLayer([24 24],15, 'ridglet_conv') % Custom Ridglet layer
    %convolution2dLayer(56, 10, 'Padding', 'same')
    batchNormalizationLayer       % Batch normalization
    reluLayer                     % ReLU activation
    
    maxPooling2dLayer(2, 'Stride', 2)  % Max pooling
    
    RidgeletConvLayer([18 18],20, 'ridglet_conv') % Custom Ridglet layer
    %convolution2dLayer(3, 16, 'Padding', 'same') % Regular conv layer
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)  % Max pooling
    
    fullyConnectedLayer(32)       % Fully connected layer
    reluLayer
    
    fullyConnectedLayer(45)       % Output layer (for 10 CIFAR-10 classes)
    softmaxLayer
    %classificationLayer
    ];
% Define training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 60, ...            % Set maximum epochs
    'InitialLearnRate', 0.01, ...   % Learning rate
    'MiniBatchSize', 128, ...       % Mini-batch size
    'Shuffle', 'every-epoch', ...   % Shuffle data every epoch
    'Verbose', false, ...
    'Plots', 'training-progress',...
    'Metrics','accuracy',...
    'ExecutionEnvironment','parallel-cpu');  % Display training progress

% Train the network
net = trainnet(XTrainGray, YTrain, layers,'crossentropy', options);
% Classify test images
YPred = predict(net, XTestGray);

% % Calculate accuracy
% %accuracy = sum(YPred == YTest) / numel(YTest);
% accuracy = mean(YPred == YTest);
% disp(['Test accuracy: ', num2str(accuracy)]);
[~, YPredLabels] = max(YPred, [], 2);

% Compare predicted labels to actual labels
accuracy = mean(YPredLabels == double(YTest));

disp(['Test accuracy: ', num2str(accuracy)]);
