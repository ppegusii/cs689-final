function [TransMat, belief] = iterCRFFeatFunc(lambdaStruct, FeatMat, bel, t, msgType)

%% Iterative transition belief determination
% In this function the proper transition matrix and matching belief is
% gathered. This function allows us to change the transition function (for
% example making it higher order) without changing the original forward
% pass code. 

%% Simple first order transition
% Simply use transition matrix
switch(msgType)
    case 'forward'
        TransMat = lambdaStruct.TransMat;

        % Simply use belief from previous timestep
        belief = bel(:,t-1);
    case 'backward'
        TransMat = lambdaStruct.TransMat;
        
        % Simply use belief from next timestep
        belief = bel(:,t+1);
        
end

%% Higher order transitions
% You could do something more elaborate here for example higher order
% transitions. Next to having the proper transition matrix it would also
% require the belief of more than the previous timestep to be included. 

