function [likClampedVal, derClampedFvec] = clampCRFFeatFunc(modelInfo, lambdaStruct, logLocalEv, FeatMat, Labels)

%% Calculate Clamped versions of likelihood and derivative
% When adding extra feature functions, they should be added at the end of
% the feature vector. That is after the observation and transition features.
% Realize that adding extra observations does not change any of this code,
% it will simply make the observation vector longer which won't change the
% matrix manipulation. However, adding extra transitions (higher order) will require some
% alterations.

T = size(FeatMat,2);
Q = modelInfo.numAct;

%%%%%%% LIKELIHOOD STEP 1 CALCULATIONS %%%%%%%%%%

%% Calculate likelihood (clamped to training data)
% Observations
idx = sub2ind([Q,T], Labels, 1:T);
likClampedObsVal = sum(logLocalEv(idx));

% Transitions
idx2 = sub2ind([Q Q], Labels(1:T-1), Labels(2:T));
likClampedTransVal = sum(lambdaStruct.logTransMat(idx2));

% ADD EXTRA FEATURE SUMS HERE

% Calculate clamped likelihood (add everything together)
likClampedVal = likClampedObsVal + likClampedTransVal; % + likeClampedCustomVal;

%%%%%%%% DERIVATIVE STEP 1 CALCULATIONS %%%%%%%%%
%% Calculate expected value of true distribution (clamped to training data)
% Observations
trueBel = zeros(Q,T);
trueBel(idx) = 1;
doSum = 1;

derClampedObsFvec = computeExpectedFvec(FeatMat, trueBel, doSum);

% Transition
idx2 = sub2ind([Q Q T-1], Labels(1:T-1), Labels(2:T), 1:(T-1));
trueBelTrans = zeros(Q,Q,T-1);
trueBelTrans(idx2) = 1;
derClampedTransFvec = sum(trueBelTrans,3);
derClampedTransFvec = derClampedTransFvec(:); %convert to vector format

% ADD EXTRA FEATURE EXPECTATIONS HERE

% Concatenate clamped expected values (concatenate everything to long
% lambda vector format)
derClampedFvec = [derClampedObsFvec ; derClampedTransFvec];
