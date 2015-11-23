function [fwbkProbsDays, fwbkNormedProbsDays] = hsmmFwBkDaysSplit(modelInfo, learnedParams, testFeatMatDays)

%Hidden semi-Markov model, aka variable duration Markov model
initial = learnedParams.initial;
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

    maxDur = learnedParams.maxDur(n);

    % Forward Backward Procedure as  as described in Rabiner 1989 (page 269)
    % and Kevin Murphy's HSMM paper (page 5)
    fwbkNormedProbs= zeros(numAct, numTimeSteps);
    fwbkProbs = zeros(numAct, numTimeSteps);
    forwVar = zeros(numAct, numTimeSteps);
    backVar = zeros(numAct, numTimeSteps);

    % Precompute observation probabilities
    cumForwObsProb = zeros(numAct, maxDur, numTimeSteps); % p(y_{t-d+1:t|j,d)
    cumBackObsProb = zeros(numAct, maxDur, numTimeSteps); % p(y_{t+1:t+d|j,d)

    % p(y_{t-d+1:t|j,d) == b(j,d,t)
    %  b(j,d,t) = p(vec(y) | j)
    % b(j,d,t) = b(j,1,t) b(j,d-1,t-1)
    tic;
    for t=1:numTimeSteps,
        for d=1:min(t,maxDur)
            if (d==1)
                % Calculate observation probability
                tempProbObs = ones(1,numAct);
                for i=1:numVals,
                    idxVal = find(testFeatMat(:,t)==modelParams.obsList(i));
                    probCurObs = prod(obsModel(idxVal,i,:),1);
                    tempProbObs=tempProbObs .* reshape(probCurObs,1,numAct);
                end
                cumForwObsProb(:,d,t) = log(tempProbObs);
                if (t~=1)
                    cumBackObsProb(:,d,t-1) = log(tempProbObs);
                end
            else
                cumForwObsProb(:,d,t) = cumForwObsProb(:,1,t)+cumForwObsProb(:,d-1,t-1);
            end
        end
    end
    toc;
    % p(y_{t+1:t+d|j,d) == c(j,d,t)
    % c(j,d,t) = p(vec(y_{t+1}) | j)
    % c(j,d,t) = c(j,1,t)c(j,d-1,t+1)
    for t=(numTimeSteps-1):-1:1,
        for d=2:min(t,maxDur)
            cumBackObsProb(:,d,t) = cumBackObsProb(:,1,t)+cumBackObsProb(:,d-1,t+1);
        end
    end

    toc;

    probLogDurs= zeros(numAct,maxDur);
    % Precompute duration probabilities
    for d=1:maxDur,
        probLogDurs(:,d) = hsmmDurationModel(d,modelParams, learnedParams);
    end
    probLogDurs = log(normalize(probLogDurs,2));

    toc;
    %%% Forward
    % alpha_t(i) = p(o_1, o_2, .. o_t, q_t= s_i|model_parameters)
    % o = observation, q = state
    h = waitbar(0,'Please wait...');

    for k=1:numTimeSteps,
        waitbar(k/(2*numTimeSteps),h);
        tempForward = zeros(numAct,min(t,maxDur));
        for d=1:min(k,maxDur)
            %%% 2) Induction

            if (k == d)
                %%% 1) Initialization
                % Use initial state distribution
                sumLogTransProb = log(initial); 
            else
                % Calculate prevForw * transModel
                repPrevForwProbs = repmat(forwVar(:,k-d),1,numAct);
                sumLogTransProb = logsum(repPrevForwProbs + log(transModel),1)';
            end

            tempForward(:,d) = sumLogTransProb + cumForwObsProb(:,d,k)+ probLogDurs(:,d);%log(hsmmDurationModel(d,modelParams, learnedParams));
        end
        forwVar(:,k)= logsum(tempForward, 2);
    end

    %%% 3) Termination
    %probOfObservations = sum(forwVar(:,numTimeSteps)); % log, doesn't
                                                        % make sense

    %%% Backward
    % Beta_t(i) = P(o_{t+1}, o_{t+2}, .., o_T | q_t=s_i)
    %  o = observation, q = state

    %%% 1) Initialization
    backVar(:,numTimeSteps)=log(1);

    for k=(numTimeSteps-1):-1:1,
        waitbar((numTimeSteps+(numTimeSteps-k))/(2*numTimeSteps),h);
        tempBackward = zeros(numAct,min(numTimeSteps-k,maxDur));
        for d=1:min(numTimeSteps-k,maxDur)
            %%% 2) Induction

            % Calculate bestProb * transModel
            repPrevBackProbs = repmat(backVar(:,k+d),1,numAct);
            sumLogTransProb = logsum(repPrevBackProbs + log(transModel),1)';

            % NOTE: cumBackObsProb(:,:,t) gives p(y_{t+1:t+d|j,d)
            tempBackward(:,d) = sumLogTransProb + cumBackObsProb(:,d,k) + probLogDurs(:,d);%log(hsmmDurationModel(d,modelParams, learnedParams));
        end
        backVar(:,k)= logsum(tempBackward, 2);
    end
    close(h);

    %%% Forward-Backward combined
    % gamma_t(i) = P(q_t=S_i|O, model_parameters)
    % O = all observations (t=1:T) q = state

    for k=1:numTimeSteps,
        fwbkProbs(:,k) = forwVar(:,k)+backVar(:,k);

        % Normalize 
        fwbkNormedProbs(:,k) = exp(fwbkProbs(:,k)-logsum(fwbkProbs(:,k)));
    end
    
    fwbkProbsDays{n} = fwbkProbs;
    fwbkNormedProbsDays{n} = fwbkNormedProbs;
end
