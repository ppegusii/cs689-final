function outTesting = nbTestDaysSplit(curExp, learnedParams)

prior = learnedParams.prior;
obsModel = learnedParams.obsModel;

modelInfo = curExp.modelInfo;
actList = modelInfo.actList;
numAct = modelInfo.numAct;
numSense = modelInfo.numSense;
numVals = modelInfo.numVals;

testFeatMatDays = curExp.testFeatMat;

inferedLabelsDays = cell(length(testFeatMatDays),1);
bestProbDays = cell(length(testFeatMatDays),1);

for n=1:length(testFeatMatDays),
    % Viterbi algorithm as described in Rabiner 1989 (page 264)
    testFeatMat = testFeatMatDays{n};
    numTimeSteps = size(testFeatMat,2);

    inferedLabels = zeros(1,numTimeSteps);
    bestProb = zeros(numAct,numTimeSteps);
    bestState = zeros(numAct,numTimeSteps);

    for k=1:numTimeSteps,

        %%% 2) Recursion

        % Calculate observation probability
        probObs = ones(1,numAct);
        for i=1:numVals,
            idxVal = find(testFeatMat(:,k)==modelInfo.obsList(i));
            probCurObs = prod(obsModel(idxVal,i,:),1);
            probObs = probObs .* reshape(probCurObs,1,numAct);
        end

           bestProb(:,k) = (prior.*probObs');
    end
    [Y,inferedLabelsDays{n}]=max(bestProb);
end
    
outTesting.inferedLabels = inferedLabelsDays;
outTesting.fwProbs = bestProb;

% outTesting.fwbkProbs = fwbkProbs;
% outTesting.pOfObservations = pOfObservations;
