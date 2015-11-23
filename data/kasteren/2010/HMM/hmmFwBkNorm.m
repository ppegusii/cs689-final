function [fwbkProbs,probOfObservations] = hmmFwBkNorm(modelInfo, learnedParams, testFeatMat);

prior = learnedParams.prior;
obsModel = learnedParams.obsModel;
transModel = learnedParams.transModel;

actList = modelInfo.actList;
numAct = modelInfo.numAct;
numVals = length(modelInfo.obsList);

[numSense,numTimeSteps] = size(testFeatMat); 

% Forward Backward Procedure as  as described in Rabiner 1989 (page 262)

fwbkProbs = zeros(numAct, numTimeSteps);
forwVar = zeros(numAct, numTimeSteps);
backVar = zeros(numAct, numTimeSteps);
scale = zeros(1, numTimeSteps);
%% Forward
% alpha_t(i) = p(o_1, o_2, .. o_t, q_t= s_i|model_parameters)
% o = observation, q = state

% Calculate observation probability
probObs = ones(1,numAct);
for i=1:numVals,
    idxVal = find(testFeatMat(:,1)==modelInfo.obsList(i));
    probCurObs = prod(obsModel(idxVal,i,:),1);
    probObs = probObs .* reshape(probCurObs,1,numAct);
end

%%% 1) Initialization
forwVar(:,1) = prior.*probObs';
% Normalize 
forwVar(:,1) = normalize(forwVar(:,1));

for k=2:numTimeSteps,

    %%% 2) Induction
    
    % Calculate bestProb * transModel <-- sum over states
    maxLogTransProb = transModel' * forwVar(:,k-1);
    
    % Calculate observation probability
    probObs = ones(1,numAct);
    for i=1:numVals,
        idxVal = find(testFeatMat(:,k)==modelInfo.obsList(i));
        probCurObs = prod(obsModel(idxVal,i,:),1);
        probObs = probObs .* reshape(probCurObs,1,numAct);
    end
    
    forwVar(:,k) = maxLogTransProb.*probObs';

    % Normalize 
    [forwVar(:,k), scale(k)] = normalize(forwVar(:,k));
end

%%% 3) Termination
probOfObservations = sum(scale);

%% Backward
% Beta_t(i) = P(o_{t+1}, o_{t+2}, .., o_T | q_t=s_i)
%  o = observation, q = state

%%% 1) Initialization
backVar(:,numTimeSteps)=1;
% Normalize 
backVar(:,1) = normalize(backVar(:,1));

for k=(numTimeSteps-1):-1:1,
    %%% 2) Induction
    
    % Calculate observation probability
    probObs = ones(1,numAct);
    for i=1:numVals,
        idxVal = find(testFeatMat(:,k+1)==modelInfo.obsList(i));
        probCurObs = prod(obsModel(idxVal,i,:),1);
        probObs = probObs .* reshape(probCurObs,1,numAct);
    end

    backVar(:,k) = transModel*(probObs'.*backVar(:,k+1));
    
    % Normalize 
    backVar(:,k) = normalize(backVar(:,k));
end
    
%% Forward-Backward combined
% gamma_t(i) = P(q_t=S_i|O, model_parameters)
% O = all observations (t=1:T) q = state

for k=1:numTimeSteps,
    fwbkProbs(:,k) = forwVar(:,k).*backVar(:,k);

    % Normalize 
    fwbkProbs(:,k) = normalize(fwbkProbs(:,k));
end