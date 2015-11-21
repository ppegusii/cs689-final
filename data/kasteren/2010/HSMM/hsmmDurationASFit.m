function [durParms, maxDur] = hsmmDurationASFit(as, modelParams)

if (modelParams.durDistr==1)% 'gamma'
    % Gamma
    [durParms, maxDur, LabelsAndDurs] = getGammParmFromAS(as, modelParams.useIdle);
elseif(modelParams.durDistr==2)% 'gauss'
    % Gauss
    [durParms, maxDur, LabelsAndDurs] = getGaussParmFromAS(as, modelParams.useIdle);
end