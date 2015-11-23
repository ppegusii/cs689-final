function [likLogZ, derExpectFvec, labels, fwprob] = crfInferenceLog(modelInfo, lambdaStruct, FeatMat, sizeStruct, logLocalEv)

%% Forward-Backward/Viterbi algorithm as described in Rabiner 1989 (page 264)
% Modified to work with CRFs as described in An introduction to conditional
% random fields for relational learning 2006 (page 15)
% Some parts of code were copied from Kevin Murphy's CRF Toolbox

%% Initialize variables
T = size(FeatMat,2);
Q = modelInfo.numAct;

%% Forwards
TransMat = lambdaStruct.logTransMat;
[bel,fwdMsg] = crfFwProbC(TransMat', logLocalEv);
fwprob = bel;

%% Backwards
[bel,belTrans] = crfBkProbC(fwprob, fwdMsg, TransMat);

%% 3) Termination
scale = logsum(bel,1);
bel = bel-repmat(scale,Q,1);
bel=exp(bel);

%%% Calculate expected derivative from belief data
derExpectFvec = termCRFFeatFunc(FeatMat, bel, belTrans);

%%% Calculate LogZ
likLogZ = scale(T);

%% 4) Infer labels
[dummy, labels] = max(bel, [], 1);
