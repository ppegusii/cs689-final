function outTrain = hsmmTrainDaysSplit(curExp)

modelInfo = curExp.modelInfo;

trainFeatMat = curExp.trainFeatMat;
trainLabels = curExp.trainLabels;
trainSegments = curExp.trainSegments;
testSegments = curExp.testSegments;

% Used to avoid zero probabilties
smallValue = modelInfo.smallValue;

% Learning of HMM parameters through maximum likelihood. 
% By counting parameters will be estimated
actList = modelInfo.actList;

numAct = modelInfo.numAct;
numSense = modelInfo.numSense;
numVals = modelInfo.numVals;
totalObsLabsPerClass = zeros(numAct,1);
totalTransLabsPerClass = zeros(numAct,1);

% Parameters
initial = zeros(numAct,1);
obsModel = zeros(numSense, numVals, numAct);
transModel = zeros(numAct, numAct);

for n=1:length(trainFeatMat),
    % initial state distribution: sum[Label(n,1)]/totalNumberDays

    [Y,I] = ismember(trainLabels{n}(1,1),actList);
    initial(I) = initial(I)+1;

    totalObsLabsPerClass = totalObsLabsPerClass + histc(trainLabels{n},actList)';   

    
    for i=1:length(actList),
        % Find all occurances of activity actList(i)
        idxObs = find(trainLabels{n}==actList(i));

        if (isempty(idxObs))
            continue;
        end
        % observation model: sum[activity=i][sensor=j]/sum[activity=i]
        for j=1:numVals,
            obsModel(:,j,i) = obsModel(:,j,i)+sum(trainFeatMat{n}(:,idxObs)==modelInfo.obsList(j),2);            
        end

        idxTrans = find(trainLabels{n}(1:end-1)==actList(i) & trainSegments{n}(1:end-1)~=trainSegments{n}(2:end));
        if (isempty(idxTrans))
            continue;
        end

        % transition model: sum[act_{t+1}=i][act_t=j]/sum[act_t=j]
        nextActidx = idxTrans + 1;

        % Count occurances of activity_{t+1}
        actCounts = histc(trainLabels{n}(nextActidx),actList);
        transModel(i, :)= transModel(i, :)+actCounts;
        totalTransLabsPerClass(i) = totalTransLabsPerClass(i) + length(idxTrans);
    end
end
% transModel(1:(numAct+1):(numAct*numAct)) = 0;

% Duration distribution
if (modelInfo.typeDurData==1)
    [durModel, dummy, dummy] = hsmmDurationASFitDaysSplit(modelInfo.as, modelInfo);
    [dummy, maxDur, dummy] = hsmmDurationASFitDaysSplit(modelInfo.as, modelInfo);
elseif (modelInfo.typeDurData==2)
    [durModel, dummy, dummy] = hsmmDurationSegmentFitDaysSplit(modelInfo, trainLabels, trainSegments);
    [dummy, maxDur, dummy] = hsmmDurationSegmentFitDaysSplit(modelInfo, curExp.testLabels, testSegments);
end

% La place smoothing
initial = (initial+smallValue)/(length(trainFeatMat)+smallValue);
obsModel = (obsModel+smallValue)./repmat(reshape((totalObsLabsPerClass+(smallValue*numVals)),1,1,numAct),numSense,numVals);
transModel = (transModel+smallValue)./repmat(totalTransLabsPerClass+(smallValue*numAct),1,numAct);

learnedParams.initial = initial;
learnedParams.obsModel = obsModel;
learnedParams.transModel = transModel;
learnedParams.durModel = durModel;
learnedParams.maxDur = maxDur;

outTrain.learnedParams = learnedParams;
