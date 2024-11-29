if ~isempty(ver('parallel'))
    % Create a parallel pool with 14 workers
    parpool('local', 14);
end

imsize = 32;

% Load NWPU-RESISC45 dataset
[XTrain, YTrain, XTest, YTest] = loadCIFAR10Data();

% Convert labels to categorical if they are not already
YTrain = categorical(YTrain);
YTest = categorical(YTest);

% Verify number of unique classes
disp('Number of unique classes in YTrain:');
disp(numel(categories(YTrain)));

% Filter for non-empty data
if isempty(XTrain) || isempty(YTrain)
    error('Filtered training data is empty. Check the class names in selectedClasses.');
end

% % Preallocate arrays for grayscale images and resize them to 64x64
XTrainGray = zeros(imsize, imsize, 1, size(XTrain, 4), 'single'); % Preallocate for grayscale training images as 'single'
XTestGray = zeros(imsize, imsize, 1, size(XTest, 4), 'single');    % Preallocate for grayscale test images as 'single'

% % Convert training images to grayscale and resize to 64x64
for i = 1:size(XTrain, 4)
    grayImage = rgb2gray(squeeze(XTrain(:, :, :, i)));  % Convert to grayscale
    XTrainGray(:, :, 1, i) = imresize(grayImage, [imsize, imsize]);  % Resize to 64x64
end
% 
% Convert test images to grayscale and resize to 64x64
for i = 1:size(XTest, 4)
    grayImage = rgb2gray(squeeze(XTest(:, :, :, i)));  % Convert to grayscale
    XTestGray(:, :, 1, i) = imresize(grayImage, [imsize, imsize]);  % Resize to 64x64
end

% Normalize the pixel values between 0 and 1 (data should be in 'single' type)
XTrainGray = single(XTrainGray / 255);
XTestGray = single(XTestGray / 255);

% Define layers for the network
layers = [
    imageInputLayer([32 32 1],"Mean",[])    % Input layer for grayscale images
    
    RidgeletConvLayer([14 14], 25, 'ridglet_conv') % Custom Ridglet layer
    convolution2dLayer(14,25)
    batchNormalizationLayer       % Batch normalization
    reluLayer                     % ReLU activation
    
    maxPooling2dLayer(2, 'Stride', 2)  % Max pooling
    
    RidgeletConvLayer([10 10], 20, 'ridglet_conv2') % Custom Ridglet layer
    batchNormalizationLayer
    reluLayer
     
    maxPooling2dLayer(2, 'Stride', 2)  % Max pooling
    
    fullyConnectedLayer(32)       % Fully connected layer
    reluLayer
    
    fullyConnectedLayer(10)       % Output layer (for 45 classes)
    softmaxLayer
    ];

% Define training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 50, ...            % Set maximum epochs
    'InitialLearnRate', 0.01, ...   % Learning rate
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.8,...
    'MiniBatchSize', 64, ...       % Mini-batch size
    'Shuffle', 'every-epoch', ...   % Shuffle data every epoch
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'Metrics', 'accuracy', ...
    'ExecutionEnvironment', 'parallel-cpu');  % Display training progress

% Train the network
net = trainnet(XTrainGray, YTrain, layers,'crossentropy', options);

% Classify test images
YPredProb = predict(net, XTestGray);

% Extract class labels from the predicted probabilities
[~, YPredIdx] = max(YPredProb, [], 2);
YPred = categorical(YPredIdx, 1:10, categories(YTrain));

% Compare predicted labels to actual labels
accuracy = mean(YPred == YTest);
disp(['Test accuracy: ', num2str(accuracy)]);

% Compute confusion matrix
confMat = confusionmat(YTest, YPred);

% Display confusion matrix
confMatChart = confusionchart(confMat, categories(YTrain), 'Title', 'Confusion Matrix', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
