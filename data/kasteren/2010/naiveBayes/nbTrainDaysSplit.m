function outTrain = nbTrainDaysSplit(curExp)

modelInfo = curExp.modelInfo;
trainFeatMat = curExp.trainFeatMat;
trainLabels = curExp.trainLabels;

% Used to avoid zero probabilties
smallValue = 0.01;

% Learning of HMM parameters through maximum likelihood. 
% By counting parameters will be estimated
actList = modelInfo.actList;

numAct = modelInfo.numAct;
numSense = modelInfo.numSense;
numVals = modelInfo.numVals;
totalObsLabsPerClass = zeros(numAct,1);
totalLabsPerClass = zeros(numAct,1);

% Parameters
prior = zeros(numAct,1);
obsModel = zeros(numSense, numVals, numAct);

for n=1:length(trainFeatMat),
    % initial state distribution: sum[Label(n,1)]/totalNumberDays

    totalObsLabsPerClass = totalObsLabsPerClass + histc(trainLabels{n},actList)';
    totalLabsPerClass = totalLabsPerClass + histc(trainLabels{n}(1:end),actList)';

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
    end
end

% La place smoothing
prior = (totalLabsPerClass+smallValue)./repmat(sum(totalLabsPerClass+smallValue,1),numAct,1);
obsModel = (obsModel+smallValue)./repmat(reshape((totalObsLabsPerClass+(smallValue*numVals)),1,1,numAct),numSense,numVals);

learnedParams.prior = prior;
learnedParams.obsModel = obsModel;


outTrain.learnedParams = learnedParams;

   
    
    

