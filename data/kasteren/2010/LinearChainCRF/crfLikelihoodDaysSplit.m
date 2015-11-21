function [likeliError,derivatives] = crfLikelihoodDaysSplit(lambdas, params)
% tic;
%% Calculate likelihood Error and derivate of data given model parameters lambda
%%% As described in An introduction to conditional
%%% random fields for relational learning 2006 (page 12)
%
%%% Calculations for both likelihood error and derivative are made up of three steps
%%% First step calculates the expected value based on the training data
%%% Second step calculates the expected value summing over all possible state values
%%% Third step is a regularization term to prevent overfitting.


%% Initialize various variables related to training data
regulVariance = 1;

modelInfo = params.modelInfo;
trainFeatMatDays= params.trainFeatMat;
trainLabelsDays = params.trainLabels;

likClampedValDays = zeros(length(trainLabelsDays),1);
derClampedFvecDays = zeros(length(trainLabelsDays),length(lambdas));
likLogZDays = zeros(length(trainLabelsDays),1);
derExpectFvecDays = zeros(length(trainLabelsDays),length(lambdas));

for n=1:length(trainFeatMatDays),
    trainFeatMat = trainFeatMatDays{n};
    trainLabels = trainLabelsDays{n};
    
    [lambdaStruct, sizeStruct, logLocalEv] = initCRFFeatFunc(modelInfo, lambdas, trainFeatMat);
    %% 1: Clamped value based on training data
    % Initialize various for calculating likelihood and derivative
    % likClampedVal is the clamped value for the first step needed to calculate
    % the likelihood

    [likClampedValDays(n), derClampedFvecDays(n,:)] = clampCRFFeatFunc(modelInfo, lambdaStruct, logLocalEv, trainFeatMat, trainLabels);

    %% 2: Expected value summing over all possible state values
    doSum = 1;
   [likLogZDays(n), derExpectFvecDays(n,:)] = crfInference(modelInfo, lambdaStruct, trainFeatMat, sizeStruct, logLocalEv);
%     [likLogZDays(n), derExpectFvecDays(n,:)] = crfInferenceLog(modelInfo, lambdaStruct, trainFeatMat, sizeStruct, logLocalEv);
end

likClampedVal= sum(likClampedValDays,1)';
derClampedFvec = sum(derClampedFvecDays,1)';
likLogZ = sum(likLogZDays,1)';
derExpectFvec = sum(derExpectFvecDays,1)';

%% 3: Regularization terms
likRegularize = sum((lambdas'.^2)*(1/(2*regulVariance^2)));
derRegularize = lambdas./(regulVariance^2);

%% Put everything together
likeliError = likClampedVal - likLogZ - likRegularize;
derivatives = derClampedFvec - derExpectFvec - derRegularize;

%% We want to minimize NEG log lik
likeliError = -likeliError;
derivatives = -derivatives;

%% Check for NANs
if (any(isnan(likeliError)))
    disp('whoops');
end
if(any(isnan(derivatives)))
     disp('whoops');
end
assert(~any(isnan(likeliError)));
assert(~any(isnan(derivatives)));

%% allow for keyboard interrupt
drawnow
% toc;