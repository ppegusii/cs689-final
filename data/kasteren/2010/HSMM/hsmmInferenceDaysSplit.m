function [inferedLabelsDays, bestProbDays, bestDurDays, inferedSegmentsDays] = hsmmInferenceDaysSplit(modelInfo, learnedParams, testFeatMatDays)
initial = learnedParams.initial;
obsModel = learnedParams.obsModel;
transModel = learnedParams.transModel;

numAct = modelInfo.numAct;
numVals = modelInfo.numVals;

inferedLabelsDays = cell(length(testFeatMatDays),1);
bestProbDays = cell(length(testFeatMatDays),1);
bestDurDays = cell(length(testFeatMatDays),1);
inferedSegmentsDays = cell(length(testFeatMatDays),1);

for n=1:length(testFeatMatDays),
    testFeatMat = testFeatMatDays{n};
    numTimeSteps = size(testFeatMat,2); 

    maxDur = learnedParams.maxDur;

    % Viterbi algorithm as described in Rabiner 1989 (page 264)
    inferedSegments = zeros(1,numTimeSteps);
    inferedLabels = zeros(1,numTimeSteps);
    bestProb = zeros(numAct,numTimeSteps);
    bestState = zeros(numAct,numTimeSteps);
    bestDur = zeros(numAct,numTimeSteps);

    probObs = ones(numAct,numTimeSteps);
    probLogDurs= zeros(numAct,maxDur);

    % Precompute duration probabilities
    for d=1:maxDur,
        probLogDurs(:,d) = hsmmDurationModel(d,modelInfo, learnedParams);
    end
    probLogDurs = log(normalise(probLogDurs,2));

%     h = waitbar(0,'Please wait...');
    for k=1:numTimeSteps,
%         waitbar(k/numTimeSteps,h);
        %%% 2) Recursion

        % Calculate observation probability
        tempProbObs = ones(1,numAct);
        for i=1:numVals,
            idxVal = find(testFeatMat(:,k)==modelInfo.obsList(i));
            probCurObs = prod(obsModel(idxVal,i,:),1);
            tempProbObs=tempProbObs .* reshape(probCurObs,1,numAct);
        end
        probObs(:,k) = log(tempProbObs)';

        tempCumObsProb = 0;

        probAlphaNoTrans = zeros(numAct,min(k,maxDur));
        bestState4Dur = zeros(numAct,min(k,maxDur));

        D = min(k,maxDur);
        for d=1:D,
            if (k==d)        %%% 1) Initialization
                tempCumObsProb = tempCumObsProb + probObs(:, 1);
                probAlphaNoTrans(:,d) = log(initial) + tempCumObsProb + probLogDurs(:,d);
            else  %% Transition             
                % Determine best transition
                repBestProbs = repmat(bestProb(:,k-d),1,numAct);
                [bestProb4Dur(:,1),bestState4Dur(:,d)]= max(repBestProbs + log(transModel),[],1);

                % Add observation and duration
                tempCumObsProb = tempCumObsProb + probObs(:, k-d+1);
                probAlphaNoTrans(:,d) = bestProb4Dur + tempCumObsProb + probLogDurs(:,d);
            end
        end
%         assert(~any(isnan(probAlphaNoTrans)));

        % Determine best duration for each state
        [bestProb(:,k), bestDur(:,k)] = max(probAlphaNoTrans, [], 2);
        % Store best state to get from. NOTE! if zero it means it's the initial state
        for i=1:numAct,
            bestState(i,k)=bestState4Dur(i,bestDur(i,k));
        end
    end
%     close(h);

    %%% 3) Termination
    [P,lastState] = max(bestProb(:,numTimeSteps));
    inferedLabels(numTimeSteps) = lastState;
    curSegment = -1;
    inferedSegments(numTimeSteps)= curSegment;
    duration = bestDur(inferedLabels(numTimeSteps),numTimeSteps);
    tempDur = bestDur(inferedLabels(numTimeSteps),numTimeSteps);
    tempDur = tempDur -1;
    %%% 4) Path backtracking

    for k=numTimeSteps-1:-1:1
        if (tempDur > 0)
            inferedLabels(k) = lastState;
            inferedSegments(k)=curSegment;
            tempDur = tempDur -1;
        else
            lastState = bestState(inferedLabels(k+duration),k+duration);
            inferedLabels(k) = lastState;
            curSegment = curSegment - 1;
            inferedSegments(k)= curSegment;

            duration = bestDur(inferedLabels(k),k);
            tempDur = bestDur(inferedLabels(k),k)-1;
        end
    end
    inferedSegments = inferedSegments+abs(inferedSegments(1))+1;

    inferedLabelsDays{n} = inferedLabels;
    bestProbDays{n} = bestProb;
    bestDurDays{n} = bestDur;
    inferedSegmentsDays{n} = inferedSegments;
end