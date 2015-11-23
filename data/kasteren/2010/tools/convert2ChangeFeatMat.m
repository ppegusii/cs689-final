function [FeatMat, Labels, Dates, varargout] = convert2ChangeFeatMat(ss, as, timeStepSize, varargin)

senseList = ss.getIDs;
numSense = length(senseList);

% minActCount = 5;
% idx = find(histc(as.id,as.getIDs)>=minActCount);
% tempActList = as.getIDs;
% idx2 = find(ismember(as.id,tempActList(idx)));
% as = as(idx2);

size1SegList = [];
if (nargin>3)
    size1SegList=varargin{1};
end  

%% check for globPars
globPars = [];
if (nargin>4)
    globPars = varargin{2};
end

%% Check for forcedActList
if (isfield(globPars,'forcedActList'))
    actList = globPars.forcedActList;
else
    actList = as.getIDs;
end

%%
startTimestamp = as(1).startsecs;
endTimestamp = as(as.len).endsecs;

numTimesteps = ceil((endTimestamp - startTimestamp)/timeStepSize)+1;

FeatMat = zeros(numSense, numTimesteps);
Labels = zeros(1, numTimesteps);
Dates = (startTimestamp + (0:(numTimesteps-1))*timeStepSize)/86400;
Segment = zeros(1, numTimesteps);

for i=1:ss.len,
    % Determine position in feature vector
    [dummy,idxS] = intersect(senseList,ss(i).id);
    
    % Determine time steps
    if (floor((ss(i).startsecs - startTimestamp)/timeStepSize)+1 > 0 & ceil((ss(i).endsecs - startTimestamp)/timeStepSize)+1>0)
        startSenseFire = floor((ss(i).startsecs - startTimestamp)/timeStepSize)+1;
        endSenseFire = ceil((ss(i).endsecs - startTimestamp)/timeStepSize)+1;
    else
        continue;
    end
    
    % In case sensor fired before first activity
    if (startSenseFire < 1)
        startSenseFire = 1;
    end        
    if (ss(i).startsecs > endTimestamp)
        continue;
    end
    if (ss(i).endsecs>endTimestamp)
        continue;
    end
    
    % Set value to 1 for sensor at time
    FeatMat(idxS, startSenseFire) = 1;
    FeatMat(idxS, endSenseFire) = 1;
end

segNum=1;
for i=1:as.len,
    % Determine position in feature vector
     [dummy,idxA] = intersect(actList,as(i).id);
    
    % Determine time steps
    startAct = floor((as(i).startsecs - startTimestamp)/timeStepSize)+1;
    endAct = ceil((as(i).endsecs - startTimestamp)/timeStepSize)+1;

    % Set value to 1 for sensor at time
    Labels(1, startAct:endAct) = idxA;
    
    if (ismember(as(i).id, size1SegList)) % use size 1 segments
        Segment(1, startAct:endAct) = segNum:segNum+(endAct-startAct);
        segNum = segNum+(endAct-startAct)+1;
    else
        Segment(1, startAct:endAct) = segNum;
        segNum= segNum+1;
    end
end

segDiff = Segment(2:end)-Segment(1:end-1);
startidx = find(segDiff<0);
startidx = startidx+1;
for i=1:length(startidx),
    idx = find(Segment(startidx(i):end)==Segment(startidx(i)));
    idx2 = find(idx(2:end) - idx(1:end-1) ~=1);
    if (isempty(idx2))
        sizSegm = length(idx)-1;
    else
        sizSegm = idx2(1)-1;
    end
    
    endidx = startidx(i) + sizSegm;
    
    if (Labels(startidx(i))==0)
        actID = 0;
    else
        actID = actList(Labels(startidx(i)));    
    end
    
    if (ismember(actID, size1SegList)) % use size 1 segments
        Segment(startidx(i):endidx) = segNum:segNum+sizSegm;
        segNum = segNum+sizSegm+1;
    else
        Segment(startidx(i):endidx) = segNum;
        segNum= segNum+1;
    end
end

varargout(1,:) = {Segment};