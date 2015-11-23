function [lambdaStruct, sizeStruct, logLocalEv] = initCRFFeatFunc(modelInfo, lambdas, FeatMat)

%% Initialize CRF Feature function variables
% This function prepares various variables for working with CRF feature
% functions.
%
% When adding extra feature functions, they should be added at the end of
% the feature vector. That is after the observation and transition features.
% Realize that adding extra observations does not change any of this code,
% it will simply make the observation vector longer which won't change the
% matrix manipulation. However, adding extra transitions (higher order) will require some
% alterations.

Q = modelInfo.numAct;
D = modelInfo.numSense;

%% Calculate sizes of matrices
sizeObs = Q*D;
sizeTrans = Q*Q;
%%%% ADD EXTRA FEATURE SIZES HERE (e.g. sizeCustom =
%%%% actSize*actSize*numSense)

sizeStruct.sizeLambda = length(lambdas);
sizeStruct.sizeObs = sizeObs;
sizeStruct.sizeTrans = sizeTrans;
sizeStruct.numStates = Q;
sizeStruct.numPrevStates = Q;


%% Split lambda vector into seperate parts
lambdaObs = lambdas(1:sizeObs);
lambdaTrans = lambdas(sizeObs+1:sizeObs+sizeTrans);

%%%% ADD EXTRA FEATURE PARTS HERE (e.g. lambdaCustom =
%%%% lambdas(sizeObs+sizeTrans+1:sizeObs+sizeTrans+sizeCustom));

%% Reshape lambda vector into matrix form
lambdaStruct.logObsMat = reshape(lambdaObs, D, Q);
lambdaStruct.logTransMat = reshape(lambdaTrans, Q, Q);
%lambdaStruct.ObsMat = exp(lambdaStruct.logObsMat);
%lambdaStruct.TransMat = exp(lambdaStruct.logTransMat);
%%%% ADD EXTRA FEATURE MATRIX HERE (e.g. lambdaCustomMat =
%%%% reshape(lambdaCustom, actSize, actSize, numSense);

% Calculate local Evidence (lambdas * observed features)
%% INSERT FEATURE FUNCTION CALCULATION HERE (input FEATMAT)
logLocalEv = lambdaStruct.logObsMat'*FeatMat;%-repmat(log(modelInfo.classPrior'),1,size(FeatMat,2));
%logLocalEv = log(exp(lambdaStruct.logObsMat')*FeatMat);


%% Check for INFs
%assert(~any(any(isinf(lambdaStruct.ObsMat))))
%assert(~any(any(isinf(lambdaStruct.TransMat))))
