d=1;

layers = [
    featureInputLayer(1)  
    fullyConnectedLayer(d) 
    CustomSigmaLayer('CustomSigma1')  
    fullyConnectedLayer(2*d+2)  
    CustomSigmaLayer('CustomSigma2')  
    fullyConnectedLayer(1)  
    %regressionLayer  
];

options = trainingOptions('adam', ...
    'MaxEpochs', 700, ... 
    'InitialLearnRate', 0.015, ... 
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.8, ...
    'LearnRateDropPeriod', 150, ...
    'MiniBatchSize', 50, ...
    'GradientThreshold', 1, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','parallel-cpu');

    % 'GradientThreshold', 1.5, ...
    
% Training data
xTrain = linspace(0,1, 200)'; % 1000 training points
yTrain = testfunc_1d(xTrain); % Compute the function values
% Testing data
xTest = linspace(0, 1, 200)'; % 200 test points
yTest = testfunc_1d(xTest); % Compute the function values

% Training
% net = trainNetwork(xTrain, yTrain, layers, options);
net = trainnet(xTrain, yTrain, layers, "mse", options);
% Evaluation
yPred = predict(net, xTest);

% Plot
figure;
plot(xTest, yTest, 'b', 'DisplayName', 'True Function');
hold on;
plot(xTest, yPred, 'r--', 'DisplayName', 'Predicted Function');
xlabel('x');
ylabel('f(x)');
title('True vs Predicted Function');
legend;
grid on;