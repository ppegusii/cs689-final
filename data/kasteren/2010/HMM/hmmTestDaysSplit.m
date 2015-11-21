function outTesting = hmmTestDaysSplit(curExp, learnedParams)

[inferedLabels, fwProbs] = hmmInferenceDaysSplit(curExp.modelInfo, learnedParams, curExp.testFeatMat);
% [fwbkProbs,pOfObservations] = hmmFwBkDaysSplit(curExp.modelInfo, learnedParams, curExp.testFeatMat);

outTesting.inferedLabels = inferedLabels;
outTesting.fwProbs = fwProbs;

% outTesting.fwbkProbs = fwbkProbs;
% outTesting.pOfObservations = pOfObservations;
