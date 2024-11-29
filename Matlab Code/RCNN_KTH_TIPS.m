% Create a parallel pool with 14 workers if Parallel Toolbox is available
if ~isempty(ver('parallel'))
    parpool('local', 14);
end

% Set image size for resizing
imsize = 100;

% Load KTH-TIPS dataset using imageDatastore
dataFolder = "C:\Users\MaxSc\Desktop\dataset\KTH_TIPS";  % Specify the path to the dataset
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', '.png');  % Adjust according to the image file format (.jpg, .png, etc.)

% Split the datastore into training and test sets (80% train, 20% test)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Convert labels to categorical
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

% Preallocate arrays for images and resize them to 32x32 (since images are already grayscale)
XTrainGray = zeros(imsize, imsize, 1, numel(imdsTrain.Files), 'single'); % Preallocate for training images
XTestGray = zeros(imsize, imsize, 1, numel(imdsTest.Files), 'single');   % Preallocate for test images

% Process and resize training images
for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);            % Read image
    XTrainGray(:, :, 1, i) = imresize(img, [imsize, imsize]);  % Resize to 32x32 (grayscale assumed)
end

% Process and resize test images
for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);            % Read image
    XTestGray(:, :, 1, i) = imresize(img, [imsize, imsize]);  % Resize to 32x32 (grayscale assumed)
end

% Normalize pixel values between 0 and 1
XTrainGray = single(XTrainGray / 255);
XTestGray = single(XTestGray / 255);

% Define layers for the network
layers = [
    imageInputLayer([imsize imsize 1],"Mean",[])    % Input layer for grayscale images
    
    RidgeletConvLayer([30 30], 25, 'ridglet_conv') % Custom Ridglet layer
    %convolution2dLayer(14,25)
    batchNormalizationLayer       % Batch normalization
    reluLayer                     % ReLU activation
    
    maxPooling2dLayer(2, 'Stride', 2)  % Max pooling
    
    % RidgeletConvLayer([10 10], 20, 'ridglet_conv2') % Custom Ridglet layer
    % batchNormalizationLayer
    % reluLayer
    % 
    % maxPooling2dLayer(2, 'Stride', 2)  % Max pooling
    
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
    'MiniBatchSize', 32, ...       % Mini-batch size
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
