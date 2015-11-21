function outTrain = linchaincrfTrainDaysSplit(curExp)

modelInfo = curExp.modelInfo;
trainFeatMat = curExp.trainFeatMat;
trainLabels = curExp.trainLabels;


%% Init Variables
sizeObs = modelInfo.numAct*modelInfo.numSense;
sizeTrans = modelInfo.numAct* modelInfo.numAct;

pars.fgname = 'crfLikelihoodDaysSplit';
pars.nvar = sizeObs+sizeTrans;
pars.modelInfo=modelInfo;
pars.trainFeatMat=trainFeatMat;
pars.trainLabels = trainLabels;

if (isfield(curExp,'initialCRFLambda')==1)
    options.x0 = curExp.initialCRFLambda;
end
maxIter = 250;


%% Minimize (matlab)
% [learnedParams.lambdas, fX, ngrad] =  minimize(rand(pars.nvar,1), 'crfLikelihood', maxIter, pars);

%% HANSO BFGS method

options.maxit = maxIter;
options.nstart= 1;
options.prtlevel = 2;
options.normtol = 1e-0;
%options.H0 = eye(pars.nvar)*0.01;
options.H0 = sparse(1:pars.nvar,1:pars.nvar,1)*0.01;
[x, f, g] = bfgs(pars, options);
learnedParams.lambdas = x(:,1);
learnedParams.funcVals = f;
learnedParams.grads = g;
%% HANSO HYBRID method
% options.maxit = maxIter;
% options.nstart= 1;
% options.prtlevel = 2;
% options.H0 = eye(pars.nvar)*0.01;
% 
% [x, f, loc, Y, G, w, H] = hanso(pars, options);
% learnedParams.lambdas = x;
%% Some guys found on web optimized BFGS method
% [x,histout,costdata] = bfgswopt(rand(pars.nvar,1),'crfLikelihood',1.d-6,maxIter,eye(pars.nvar),pars);
% 
% learnedParams.lambdas = x;

%% Matlab Optimisation kit
% options=optimset('LargeScale','off','Display', 'iter','MaxIter',maxIter,'TolX',0.0001,'TolFun',0.0001,'HessUpdate','bfgs');
% 
% [w, finalErr, exitFlag, out] = fminunc(@crfLikelihood, rand(pars.nvar,1), options, pars);
% learnedParams.lambdas = w;


outTrain.learnedParams = learnedParams;



