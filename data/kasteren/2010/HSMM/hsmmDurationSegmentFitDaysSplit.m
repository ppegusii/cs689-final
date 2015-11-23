function [durParms, maxDur, LabelsAndDurs, varargout] = hsmmDurationSegmentFitDaysSplit(modelInfo, Labels, Segments)

actList = modelInfo.actList;
numAct = modelInfo.numAct;
maxDur = 1;

LabelsAndDurs = [];
durCell = cell(numAct,1);
numLabs = zeros(numAct,1);

for i=1:length(actList),
    durs = [];
    for n=1:length(Labels),
        idx = find(Labels{n}==actList(i));
        curDurs = histc(Segments{n}(idx), unique(Segments{n}(idx)));

        durs = [durs,curDurs];

        maxDur = max([curDurs  maxDur]);

        temp = [actList(i)*ones(1,length(curDurs)); curDurs];
        LabelsAndDurs = [LabelsAndDurs temp];
        
        numLabs(i) = numLabs(i) + length(Labels{n}(idx));
    end
    if (isempty(durs))
        durs = 0;
    end
    durCell{i} = durs;
    minlist(i) = max(floor(min(durs)*0.9),1);
    if (max(durs)==1)
        maxlist(i) = 1;
    else
        maxlist(i) = ceil(max(durs)*1.1);
    end
end
maxDur = ceil(maxDur*1.1);

% minlist = ones(1,length(actList));
% minlist = ones(1,length(actList))*maxDur;

if(modelInfo.durDistr==1)% 'gamma'
    durParms =  getGammParmFromSegmentsDaysSplit(modelInfo, durCell, max(maxDur));
elseif(modelInfo.durDistr==2)% 'gauss'
    durParms =  getGaussParmFromSegmentsDaysSplit(modelInfo, durCell, max(maxDur));
elseif(modelInfo.durDistr==3)% 'Poisson'
    durParms = getPoissonParmFromSegmentsDaysSplit(modelInfo, durCell, max(maxDur));
elseif(modelInfo.durDistr==4)% 'MOG'
    durParms = getMOGParmFromSegmentsDaysSplit(modelInfo, durCell, max(maxDur));
elseif(modelInfo.durDistr==5)% 'Multivariate'
    modelInfo.useNumBins =0;
    durParms = getMultiVarParmFromSegmentsDaysSplit(modelInfo, durCell, max(maxDur),1);
elseif(modelInfo.durDistr==7)% 'Geometric'
    for i=1:numAct,
        durParms(i,1) =(sum(durCell{i}-1)+modelInfo.smallValue)/(sum(durCell{i})+modelInfo.smallValue*numAct);
    end
elseif(modelInfo.durDistr==8)% 'Histogram'
    % determine which bin duration falls in
    durParms = getMultiVarParmFromSegmentsDaysSplit(modelInfo, durCell, max(maxDur),modelInfo.binSize);
end

varargout{1} = minlist;
varargout{2} = maxlist;
