function [modelInfo] = createModelInfo(FeatMat, Labels, Dates, Segments, globPars, durationModel)

modelInfo.useIdle = globPars.useIdle;
if (isfield(globPars,'forcedActList'))
    modelInfo.actList = 1:length(globPars.forcedActList);
    if (globPars.useIdle)
        modelInfo.actList = 1:(length(globPars.forcedActList)+1);
    end
else
    modelInfo.actList = unique(Labels);
end
modelInfo.obsList = [0 1];%unique(FeatMat);
modelInfo.numAct = length(modelInfo.actList);
modelInfo.numSense = size(FeatMat,1);
modelInfo.numVals = length(modelInfo.obsList);
modelInfo.durDistr = durationModel; %1: gamma, 2: gauss, 3: Poisson, 4: MOG
fieldList = fieldnames(globPars);
for i=1:length(fieldList),
    modelInfo = setfield(modelInfo, fieldList{i}, getfield(globPars, fieldList{i}));
end


%         % NOTE providing full as!!!!!!!!!!!!
%         modelInfo.trainAs=as;
%         modelInfo.testAs=as;
% 
%         testSegments = testLabels(2,:);
%         testLabels = testLabels(1,:);
%         trainSegments = trainLabels(2,:);
%         trainLabels = trainLabels(1,:);
% 
%         modelInfo.testLabels = testLabels;
%         modelInfo.trainSegments = trainSegments;
%         modelInfo.testSegments = testSegments;
% 
%         [modelInfo.numSense,modelInfo.numTrainTimeSteps] = size(trainFeatMat); 
%         modelInfo.numTestTimeSteps = size(testFeatMat); 
%         if (isempty(testFeatMat))
%             continue;
%         end
% 
%         %% Calculate prior
%         modelInfo.classPrior = histc(trainLabels, modelInfo.actList)/size(trainLabels,2);
