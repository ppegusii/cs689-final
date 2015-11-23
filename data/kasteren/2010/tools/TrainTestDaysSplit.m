function output = TrainTestDaysSplit(modelName, curExp)

% This function trains and tests a model for a given collection of training
% and test data. 
%
% Models should be placed in a directory named modelName, the directory
% should contain a modelNameTrain and modelNameTest file.

% Training (learn parameters)
tic;
modelTrain = strcat(modelName, 'TrainDaysSplit'); % create train functionname e.g. hmmTrain
output.training = feval(modelTrain, curExp);
output.training.elapsedTime = toc;

% Testing (infer labels on testdata)
tic;
modelTest = strcat(modelName, 'TestDaysSplit'); % create test functionname e.g. hmmTest
output.testing = feval(modelTest, curExp, output.training.learnedParams);
output.testing.elapsedTime = toc;