function derExpectFvec = termCRFFeatFuncExpect(expValObs, expValTrans,  likLogZ)
% Observation
derExpectObsFvec =exp(logsum(expValObs(:,:,end),1)'-likLogZ);

% Transition
derExpectTransFvec =exp(logsum(expValTrans(:,:,end),1)'-likLogZ);

% Concatenate clamped expected values (concatenate everything to long
% lambda vector format)
derExpectFvec = [derExpectObsFvec ; derExpectTransFvec];
