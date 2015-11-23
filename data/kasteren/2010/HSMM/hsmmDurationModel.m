function Y = hsmmDurationModel(duration, modelInfo, learnedParams)

numAct = modelInfo.numAct;
if (modelInfo.durDistr==1)% 'gamma'
    % Gamma function
    Y=gampdf(duration, learnedParams.durModel(:,1), learnedParams.durModel(:,2));
elseif(modelInfo.durDistr==2)% 'gauss'
    % Gauss function
    Y(:,1)=normpdf(ones(numAct,1)'*duration, learnedParams.durModel(:,1)', learnedParams.durModel(:,2)')';
elseif(modelInfo.durDistr==3)% 'poisson'
    for i=1:numAct,
        Y(i,1)=poisspdf(duration, learnedParams.durModel(i));
    end
elseif(modelInfo.durDistr==4)% 'mog'
    for i=1:numAct,
        Y(i,1) = gmmprob(learnedParams.durModel(i),duration);
    end
elseif(modelInfo.durDistr==5)% 'Multivariate'
    Y = learnedParams.durModel.values(:,duration);
elseif(modelInfo.durDistr==7)% 'Geometric'
    Y = learnedParams.durModel(:,1).^(duration-1) .* (1-learnedParams.durModel(:,1));
elseif(modelInfo.durDistr==8)% 'Histogram'
    % determine which bin duration falls in
    if (modelInfo.useNumBins)    
        for i=1:numAct,
            binIdx = floor((duration-1)/learnedParams.durModel.binSize(i))+1;
            if (binIdx>modelInfo.numBins)
                binIdx = modelInfo.numBins;
            end
            Y(i,1) = learnedParams.durModel.values(i,binIdx);
        end
    else
        binIdx = floor((duration-1)/modelInfo.binSize)+1;
        Y = learnedParams.durModel.values(:,binIdx);
    end

end

if(~isstruct(Y) & min(Y)==0)
    idx = find(Y==0);
    Y(idx)=realmin;
end

