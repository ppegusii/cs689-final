function [likLogZ, derExpectFvec] = crfCalcExpect(modelInfo, lambdaStruct, FeatMat, sizeStruct, logLocalEv)

%% Forward-Backward/Viterbi algorithm as described in Rabiner 1989 (page 264)
% Modified to work with CRFs as described in An introduction to conditional
% random fields for relational learning 2006 (page 15)
% Some parts of code were copied from Kevin Murphy's CRF Toolbox

%% Initialize variables
T = size(FeatMat,2);
Q = modelInfo.numAct;

bel = zeros(Q,T);
belTrans = zeros(sizeStruct.numStates,sizeStruct.numPrevStates,T-1);
scale = zeros(1,T);
fwdMsg = zeros(Q,T); % fwdMsg(:,t) = msg from t to t+1
bestState = zeros(Q,T);
labels = zeros(1,T);

expValObs = -inf(Q,sizeStruct.sizeObs,T);
expValTrans = -inf(Q,sizeStruct.sizeTrans,T);

%% Forwards
%%% 1) Initialization 
bel(:,1) = logLocalEv(:,1);
% Calc Expectation
tempEfw = logLocalEv(:,1);
expValObs(:,:,1) = computeLogExpectedFvec(log(repmat(FeatMat(:,1),1,Q)), log(eye(Q)) + repmat(tempEfw,1,Q), 0)';

TransMat = lambdaStruct.logTransMat;
%%% 2) Recursion
for t=2:T
    %% THIS SHOULD BE DONE IN ITERCRF, BUT SLOWS DOWN THE PROGRAM BY 0.3
    %% SECONDS
    belief = bel(:,t-1);
    %%% DONE ITER CRF
 
    fwdMsg(:,t-1) = logmtimes(TransMat',belief); % sum_{qt-1} pot(qt, qt-1) bel(qt-1)

    bel(:,t)= logLocalEv(:,t)+fwdMsg(:,t-1);

    tempEfw = repmat(logLocalEv(:,t),1,Q)'+TransMat;
    tempForw = logsumexpC(repmat(bel(:,t-1),1,Q)+tempEfw,1);

    expValObs(:,:,t)=logmtimes(expValObs(:,:,t-1)',tempEfw)';
    expValObs(:,:,t) = logsumexp(cat(3,expValObs(:,:,t),computeLogExpectedFvec(log(repmat(FeatMat(:,t),1,Q)), log(eye(Q))+repmat(tempForw',1,Q), 0)'),3);
    
    expValTrans(:,:,t) = logmtimes(expValTrans(:,:,t-1)', tempEfw)'; 
    expValTrans(:,:,t) =  logsumexp(cat(3,expValTrans(:,:,t),computeLogExpectedFvec(repmat(bel(:,t-1),1,Q) + tempEfw ,log(eye(Q)),0)'),3);            
end

%% 3) Termination
scale = logsum(bel,1);
bel = bel-repmat(scale,Q,1);
fwprob=exp(bel);

%%% Calculate LogZ
likLogZ = scale(T);

%%% Calculate expected derivative from belief data
derExpectFvec = termCRFFeatFuncExpect(expValObs, expValTrans, likLogZ);
