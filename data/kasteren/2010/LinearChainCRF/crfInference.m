function [likLogZ, derExpectFvec, labels, fwprob] = crfInference(modelInfo, lambdaStruct, FeatMat, sizeStruct, logLocalEv)

%% Forward-Backward/Viterbi algorithm as described in Rabiner 1989 (page 264)
% Modified to work with CRFs as described in An introduction to conditional
% random fields for relational learning 2006 (page 15)
% Some parts of code were copied from Kevin Murphy's CRF Toolbox

%% Initialize variables
T = size(FeatMat,2);
Q = modelInfo.numAct;

doNormalization = 0;

bel = zeros(Q,T);
belTrans = zeros(sizeStruct.numStates,sizeStruct.numPrevStates,T-1);
scale = zeros(1,T);
fwdMsg = zeros(Q,T); % fwdMsg(:,t) = msg from t to t+1
bestState = zeros(Q,T);

doSum = 0;

%% Forwards
TransMat = lambdaStruct.logTransMat;

%[bel,fwdMsg] = crfFwProbC(TransMat', logLocalEv);

%%% 1) Initialization 

bel(:,1) = logLocalEv(:,1);
if doNormalization
%	[bel(:,1), scale(1)] = normaliseC(bel(:,1));
    scale(1) = logsum(bel(:,1),1);
    bel(:,1) = bel(:,1) - scale(1);   
end
%%% 2) Recursion
for t=2:T
    %% THIS SHOULD BE DONE IN ITERCRF, BUT SLOWS DOWN THE PROGRAM BY 0.3
    %% SECONDS
    belief = bel(:,t-1);
    %%% DONE ITER CRF
 
%    fwdMsg(:,t-1) = logmtimesC(TransMat',belief); % sum_{qt-1} pot(qt, qt-1) bel(qt-1)
    [fwdMsg(:,t-1),bestState(:,t-1)] = log_max_mult(TransMat, belief);

    bel(:,t)= logLocalEv(:,t)+fwdMsg(:,t-1);

%     if doNormalization
% %        [bel(:,t), scale(t)] = normaliseC(bel(:,t));
%         scale(t) = logsum(bel(:,t),1);
%         bel(:,t) = bel(:,t) - scale(t);   
%     end
end

fwprob = bel;
%% Backwards
for t=T-1:-1:1
%    [TransMat, belief]= iterCRFFeatFunc(lambdaStruct, FeatMat, bel, t, 'backward');
     %% THIS SHOULD BE DONE IN ITERCRF, BUT SLOWS DOWN THE PROGRAM BY 0.3
    %% SECONDS
    belief = bel(:,t+1);
    % DONE ITERCRF
    
    tmp = belief - fwdMsg(:,t); % undo effect of incoming msg to get beta from gamma
    belttp1 = repmat(bel(:,t), 1, sizeStruct.numPrevStates) + repmat(tmp(:)', sizeStruct.numPrevStates, 1); % alpha * beta
   belTrans(:,:,t) = exp((belttp1+TransMat)-repmat(logsum(logsum(belttp1+TransMat)),Q,Q)); % pot(qt,qt+1) * belttp1(qt,qt+1)
%     belTrans(:,:,t) = exp((belttp1+TransMat)-repmat(logsumAllDimC(belttp1+TransMat),Q,Q)); % pot(qt,qt+1) * belttp1(qt,qt+1)
    
    backMsg = logmtimes(TransMat, tmp); % sum_{qt+1} pot(qt,qt+1) * tmp(qt+1)
    
    bel(:,t) = bel(:,t) + backMsg;
    
%     if doNormalization
% %         bel(:,t) = normaliseC(bel(:,t));
%         scaleTemp = logsum(bel(:,t),1);
%         bel(:,t) = bel(:,t) - scaleTemp;   
%     end
end
%% 3) Termination
scale = logsum(bel,1);
bel = bel-repmat(scale,Q,1);
bel=exp(bel);
%%% Calculate expected derivative from belief data
belTrans = sum(belTrans,3);
derExpectFvec = termCRFFeatFunc(FeatMat, bel, belTrans);

%%% Calculate LogZ
if ~doNormalization
    likLogZ = scale(T);
else
    likLogZ = sum(scale);
end

%% 4) Infer labels
if (~doSum) % Path backtracking (Viterbi) 
    [dummy,labels(T)] = max(fwprob(:,T));

    for t=T-1:-1:1
        labels(t) = bestState(labels(t+1),t);
    end
else % Most probable state (Forward-Backward)
    [dummy, labels] = max(bel, [], 1);
end

