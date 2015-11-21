function mvParms = getMultiVarParmFromSegmentsDaysSplit(modelInfo, durCell, maxDur, binSize)

doPlot = 0;
actList = modelInfo.actList;
%use maxdur from modelInfo

if (isfield(modelInfo,'maxDur'))
    maxDur = modelInfo.maxDur; 
end

if (modelInfo.useNumBins)    
    numParms = modelInfo.numBins;
else
    numParms = length(1:binSize:(ceil((maxDur-1)/binSize)*binSize+1));
end
mvParms.values= zeros(length(actList), numParms);

for i=1:length(durCell),
    if (modelInfo.useNumBins)    
        mvParms.maxDur4Act = max(max(durCell{i}),modelInfo.numBins);
        mvParms.binSize(i) = max(1,mvParms.maxDur4Act/modelInfo.numBins);
        mvParms.values(i,:)=histc(durCell{i},1:mvParms.binSize(i):mvParms.maxDur4Act);
    else
        mvParms.values(i,:)=histc(durCell{i},1:binSize:(ceil((maxDur-1)/binSize)*binSize+1));
    end
end

mvParms.values=mvParms.values + modelInfo.smallValue;

mvParms.values = mvParms.values./repmat(sum(mvParms.values,2),1,numParms);