function [fwbkProbsDays,fwbkNormedProbsDays] = hmmFwBkDaysSplit(modelInfo, learnedParams, testFeatMatDays);

dummy='dummy';
prior = learnedParams.prior;
obsModel = learnedParams.obsModel;
transModel = learnedParams.transModel;

numAct = modelInfo.numAct;
numSense = modelInfo.numSense;
numVals = modelInfo.numVals;

fwbkProbsDays = cell(length(testFeatMatDays),1);
fwbkNormedProbsDays = cell(length(testFeatMatDays),1);

for n=1:length(testFeatMatDays),
    testFeatMat = testFeatMatDays{n};
    numTimeSteps = size(testFeatMat,2); 

    % Forward Backward Procedure as  as described in Rabiner 1989 (page 262)

    fwbkProbs = zeros(numAct, numTimeSteps);
    forwVar = zeros(numAct, numTimeSteps);
    backVar = zeros(numAct, numTimeSteps);

    %%% Forward
    % alpha_t(i) = p(o_1, o_2, .. o_t, q_t= s_i|model_parameters)
    % o = observation, q = state

    % Calculate observation probability
    tempProbObs = ones(1,numAct);
    for i=1:numVals,
        idxVal = find(testFeatMat(:,1)==modelInfo.obsList(i));
        probCurObs = prod(obsModel(idxVal,i,:),1);
        tempProbObs=tempProbObs .* reshape(probCurObs,1,numAct);
    end
    probObs = log(tempProbObs);

    %%% 1) Initialization
    forwVar(:,1) = log(prior) + probObs';

    for k=2:numTimeSteps,

        %%% 2) Induction

        % Calculate bestProb * transModel
        repBestProbs = repmat(forwVar(:,k-1),1,numAct);
        maxLogTransProb = logsum(repBestProbs + log(transModel),1);

        % Calculate observation probability
        tempProbObs = ones(1,numAct);
        for i=1:numVals,
            idxVal = find(testFeatMat(:,k)==modelInfo.obsList(i));
            probCurObs = prod(obsModel(idxVal,i,:),1);
            tempProbObs=tempProbObs .* reshape(probCurObs,1,numAct);
        end
        probObs = log(tempProbObs);

        forwVar(:,k) = maxLogTransProb' + probObs';
    end

    %%% 3) Termination
    %probOfObservations = sum(forwVar(:,numTimeSteps)); % Normalized, doesn't
                                                        % make sense

    %%% Backward
    % Beta_t(i) = P(o_{t+1}, o_{t+2}, .., o_T | q_t=s_i)
    %  o = observation, q = state

    %%% 1) Initialization
    backVar(:,numTimeSteps)=log(1);

    for k=(numTimeSteps-1):-1:1,
        %%% 2) Induction

        % Calculate bestProb * transModel
        repBestProbs = repmat(backVar(:,k+1),1,numAct);
        maxLogTransProb = logsum(repBestProbs + log(transModel),1);

        % Calculate observation probability
        tempProbObs = ones(1,numAct);
        for i=1:numVals,
            idxVal = find(testFeatMat(:,k+1)==modelInfo.obsList(i));
            probCurObs = prod(obsModel(idxVal,i,:),1);
            tempProbObs=tempProbObs .* reshape(probCurObs,1,numAct);
        end
        probObs = log(tempProbObs);

        backVar(:,k) = maxLogTransProb' + probObs';
    end

    %%% Forward-Backward combined
    % gamma_t(i) = P(q_t=S_i|O, model_parameters)
    % O = all observations (t=1:T) q = state

    for k=1:numTimeSteps,
        fwbkProbs(:,k) = forwVar(:,k)+backVar(:,k);

        % Normalize 
        fwbkNormedProbs(:,k) = normalize(exp(fwbkProbs(:,k)));
    end
    
    fwbkProbsDays{n} = fwbkProbs;
    fwbkNormedProbsDays{n} = fwbkNormedProbs;
end