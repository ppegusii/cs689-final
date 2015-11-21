function outTrain = hmmTrainDaysSplit(curExp)

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
totalTransLabsPerClass = zeros(numAct,1);

% Parameters
prior = zeros(numAct,1);
obsModel = zeros(numSense, numVals, numAct);
transModel = zeros(numAct, numAct);

for n=1:length(trainFeatMat),
    % initial state distribution: sum[Label(n,1)]/totalNumberDays

    [Y,I] = ismember(trainLabels{n}(1,1),actList);
    prior(I) = prior(I)+1;

    totalObsLabsPerClass = totalObsLabsPerClass + histc(trainLabels{n},actList)';
    totalTransLabsPerClass = totalTransLabsPerClass + histc(trainLabels{n}(1:end-1),actList)';

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

        idxTrans = find(trainLabels{n}(1:end-1)==actList(i));
        if (isempty(idxTrans))
            continue;
        end

        % transition model: sum[act_{t+1}=i][act_t=j]/sum[act_t=j]
        nextActidx = idxTrans + 1;

        % Count occurances of activity_{t+1}
        actCounts = histc(trainLabels{n}(nextActidx),actList);
        transModel(i, :)= transModel(i, :)+actCounts;
    end
end

% La place smoothing
prior = (prior+smallValue)/(length(trainFeatMat)+smallValue);
obsModel = (obsModel+smallValue)./repmat(reshape((totalObsLabsPerClass+(smallValue*numVals)),1,1,numAct),numSense,numVals);
transModel = (transModel+smallValue)./repmat(totalTransLabsPerClass+(smallValue*numAct),1,numAct);

learnedParams.prior = prior;
learnedParams.obsModel = obsModel;
learnedParams.transModel = transModel;


outTrain.learnedParams = learnedParams;

   
    
    

