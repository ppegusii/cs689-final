function outTesting = linchaincrfTestDaysSplit(curExp, learnedParams)

% inferedLabelsVitDays = cell(length(curExp.testFeatMat),1);
% bestVitProbDays = cell(length(curExp.testFeatMat),1);
inferedLabelsDays = cell(length(curExp.testFeatMat),1);
bestProbDays = cell(length(curExp.testFeatMat),1);

for n=1:length(curExp.testFeatMat),
    testFeatMat = curExp.testFeatMat{n};    
    [lambdaStruct, sizeStruct, logLocalEv] = initCRFFeatFunc(curExp.modelInfo, learnedParams.lambdas, testFeatMat);

%     [likLogZ, derExpectFvec, inferedLabelsVitDays{n}, bestVitProbDays{n}] = crfInference(curExp.modelInfo, lambdaStruct, testFeatMat, sizeStruct, logLocalEv, 0); % Viterbi
    [likLogZ, derExpectFvec, inferedLabelsDays{n}, bestProbDays{n}] = crfInference(curExp.modelInfo, lambdaStruct, testFeatMat, sizeStruct, logLocalEv); % Forward-backward

end

outTesting.inferedLabels = inferedLabelsDays;
outTesting.fwbkProbs = bestProbDays;

% outTesting.inferedVitLabels = inferedLabelsVitDays;
% outTesting.fwVitProbs = bestVitProbDays;
