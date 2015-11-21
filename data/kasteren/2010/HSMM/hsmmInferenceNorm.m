function [inferedLabels, bestProb] = hmmInference(modelInfo, learnedParams, testFeatMat);

prior = learnedParams.prior;
obsModel = learnedParams.obsModel;
transModel = learnedParams.transModel;

actList = modelInfo.actList;
numAct = modelInfo.numAct;
numVals = length(modelInfo.obsList);
[numSense,numTimeSteps] = size(testFeatMat); 

% Viterbi algorithm as described in Rabiner 1989 (page 264)

inferedLabels = zeros(1,numTimeSteps);
bestProb = zeros(numAct,numTimeSteps);
bestState = zeros(numAct,numTimeSteps);

% Calculate observation probability
probObs = ones(1,numAct);
for i=1:numVals,
    idxVal = find(testFeatMat(:,1)==modelInfo.obsList(i));
    probCurObs = prod(obsModel(idxVal,i,:),1);
    probObs = probObs .* reshape(probCurObs,1,numAct);
end

%%% 1) Initialization
bestProb(:,1) = prior.*probObs';
bestState(:,1) = 0;

for k=2:numTimeSteps,

    %%% 2) Recursion
    
    % Calculate bestProb * transModel
    repBestProbs = repmat(bestProb(:,k-1),1,numAct);
    [maxLogTransProb, bestState(:,k)] = max(repBestProbs .* transModel,[],1);
    
% Calculate observation probability
probObs = ones(1,numAct);
for i=1:numVals,
    idxVal = find(testFeatMat(:,k)==modelInfo.obsList(i));
    probCurObs = prod(obsModel(idxVal,i,:),1);
    probObs = probObs .* reshape(probCurObs,1,numAct);
end

    bestProb(:,k) = maxLogTransProb.*probObs;
    
    % Normalize 
    bestProb(:,k) = normalize(bestProb(:,k));    
end

%%% 3) Termination
[P,inferedLabels(numTimeSteps)] = max(bestProb(:,numTimeSteps));

%%% 4) Path backtracking

for k=numTimeSteps-1:-1:1
    inferedLabels(k) = bestState(inferedLabels(k+1),k+1);
end