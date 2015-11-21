function outTesting = hsmmTestDaysSplit(curExp, learnedParams)

%Hidden semi-Markov model, aka variable duration Markov model

[inferedLabels, fwProbs, bestDur, inferedSegments] = hsmmInferenceDaysSplit(curExp.modelInfo, learnedParams, curExp.testFeatMat);
%[dummy,bestDur] = hsmmFwBkDaysSplit(modelParams, learnedParams, testFeatMat);
% [Y,I] = max(bestDur);
% inferedLabels = I;

outTesting.inferedLabels = inferedLabels;
outTesting.inferedSegments = inferedSegments;
outTesting.fwProbs = fwProbs;
outTesting.bestDur = bestDur;