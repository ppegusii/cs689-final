function [likLogZ, derExpectFvec, labels, bel] = crfInference(modelInfo, lambdaStruct, FeatMat, sizeStruct, logLocalEv, doSum)

%% Forward-Backward/Viterbi algorithm as described in Rabiner 1989 (page 264)
% Modified to work with CRFs as described in An introduction to conditional
% random fields for relational learning 2006 (page 15)
% Some parts of code were copied from Kevin Murphy's CRF Toolbox

%% Initialize variables
T = size(FeatMat,2);
Q = modelInfo.numAct;

localEv = exp(logLocalEv);
doNormalization = 1;

bel = zeros(Q,T);
belTrans = zeros(sizeStruct.numStates,sizeStruct.numPrevStates,T-1);
scale = zeros(1,T);
fwdMsg = zeros(Q,T); % fwdMsg(:,t) = msg from t to t+1
bestState = zeros(Q,T);
labels = zeros(1,T);

%% Forwards
%%% 1) Initialization 
bel(:,1) = localEv(:,1);
if doNormalization
	[bel(:,1), scale(1)] = normaliseC(bel(:,1));
end
TransMat = exp(lambdaStruct.logTransMat);
%%% 2) Recursion
for t=2:T
%	 [TransMat, belief]= iterCRFFeatFunc(lambdaStruct, FeatMat, bel, t, 'forward');

    %% THIS SHOULD BE DONE IN ITERCRF, BUT SLOWS DOWN THE PROGRAM BY 0.3
    %% SECONDS
    belief = bel(:,t-1);
    %%% DONE ITER CRF
 
    if ~doSum
        [fwdMsg(:,t-1),bestState(:,t-1)] = max_mult(TransMat', belief);
    else
        fwdMsg(:,t-1) = TransMat' * belief; % sum_{qt-1} pot(qt, qt-1) bel(qt-1)
    end
    bel(:,t)= localEv(:,t) .* fwdMsg(:,t-1);
    %bel(:,t)= bel(:,t)./modelInfo.classPrior';
    

    if doNormalization
        [bel(:,t), scale(t)] = normaliseC(bel(:,t));
    end
end

%% Backwards
for t=T-1:-1:1
%    [TransMat, belief]= iterCRFFeatFunc(lambdaStruct, FeatMat, bel, t, 'backward');
     %% THIS SHOULD BE DONE IN ITERCRF, BUT SLOWS DOWN THE PROGRAM BY 0.3
    %% SECONDS
    belief = bel(:,t+1);
    % DONE ITERCRF
    
    tmp = belief ./ fwdMsg(:,t); % undo effect of incoming msg to get beta from gamma
    belttp1 = repmatC(bel(:,t), 1, sizeStruct.numPrevStates) .* repmatC(tmp(:)', sizeStruct.numPrevStates, 1); % alpha * beta
    belTrans(:,:,t) = normaliseC(belttp1 .* TransMat); % pot(qt,qt+1) * belttp1(qt,qt+1)
    
    if ~doSum
        backMsg = max_mult(TransMat, tmp);
    else
        backMsg = TransMat * tmp; % sum_{qt+1} pot(qt,qt+1) * tmp(qt+1)
    end
    
    bel(:,t) = bel(:,t) .* backMsg;
    
    if doNormalization
        bel(:,t) = normaliseC(bel(:,t));
    end
end

%% 3) Termination

%%% Calculate expected derivative from belief data
if (doSum)
    derExpectFvec = termCRFFeatFunc(FeatMat, bel, belTrans);
else
    derExpectFvec = -1; % Can't calculate if not summed
end

%%% Calculate LogZ
if ~doNormalization
    [junk, littleZ] = normaliseC(bel(:,1));
    bel = normaliseC(bel, 1);
    likLogZ = log(littleZ);
else
    likLogZ = sum(log(scale));
end

%% 4) Infer labels
if (~doSum) % Path backtracking (Viterbi) 
    [dummy,labels(T)] = max(bel(:,T));

    for t=T-1:-1:1
        labels(t) = bestState(labels(t+1),t);
    end
else % Most probable state (Forward-Backward)
    [dummy, labels] = max(bel, [], 1);
end
