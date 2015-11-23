function derExpectFvec = termCRFFeatFunc(FeatMat, bel, derExpectTransFvec)

% Observation
derExpectObsFvec = computeExpectedFvec(FeatMat, bel, 1);
%derExpectObsFvec = computeLogExpectedFvec(FeatMat, bel, 1);

% Transition
% derExpectTransFvec = sum(belE,3);
%derExpectTransFvec = logsum(belE,3);
derExpectTransFvec = derExpectTransFvec(:); %convert to vector format


% Concatenate clamped expected values (concatenate everything to long
% lambda vector format)
derExpectFvec = [derExpectObsFvec ; derExpectTransFvec];