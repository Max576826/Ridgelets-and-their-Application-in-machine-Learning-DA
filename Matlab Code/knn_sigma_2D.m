
nTrain = 250; 
xTrain = linspace(-3,3,nTrain)'; 
yTrain = linspace(-3,3,nTrain)';
d=2;

fTrain = arrayfun(@(x, y) testfunc(x, y), xTrain, yTrain);


inputs = [xTrain, yTrain];
targets = fTrain; 


layers = [
    featureInputLayer(d)  
    fullyConnectedLayer(d) 
    CustomSigmaLayer('CustomSigma1')  
    fullyConnectedLayer(2*d+2)  
    CustomSigmaLayer('CustomSigma2')  
    fullyConnectedLayer(1)  
    %regressionLayer  
];


options = trainingOptions('adam', ...
    'MaxEpochs', 250, ... 
    'InitialLearnRate', 0.008, ... 
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.8, ...
    'LearnRateDropPeriod', 100, ...
    'MiniBatchSize', 50, ...
    'GradientThreshold', 1, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','parallel-cpu');

net = trainnet(inputs, targets, layers, "mse", options);

% Test
nTest = 100; 
[xTest, yTest] = meshgrid(linspace(-3,3, nTest), linspace(-3,3, nTest));
xTest = xTest(:);
yTest = yTest(:);


fPred = predict(net, [xTest, yTest]);

% True function values for comparison
fTrue = arrayfun(@(x, y) testfunc(x, y), xTest, yTest);


fPred = reshape(fPred, nTest, nTest);
fTrue = reshape(fTrue, nTest, nTest);


figure;
subplot(1, 2, 1);
mesh(linspace(-3, 3, nTest), linspace(-3, 3, nTest), fTrue);
title('True Function');
xlabel('x');
ylabel('y');
zlabel('f(x, y)');

subplot(1, 2, 2);
mesh(linspace(-3, 3, nTest), linspace(-3, 3, nTest), fPred);
title('Network Prediction');
xlabel('x');
ylabel('y');
zlabel('Predicted Value');