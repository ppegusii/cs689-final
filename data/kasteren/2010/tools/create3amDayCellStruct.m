function [FeatOut, LabOut, DatOut, SegOut, DSetInfo] = create3amDayCellStruct(FeatIn, LabIn, DatIn, SegIn,globPars)

FeatOut = {};
LabOut = {};
DatOut = {};
SegOut = {};

DSetInfo.firstDay = floor(DatIn(1));
DSetInfo.daysIdx = unique(floor(DatIn));
DSetInfo.numDays = length(DSetInfo.daysIdx);

idxSaves = [];

cutHour= globPars.cutHour;

index = 1;
for i=1:max((floor(DatIn)-DSetInfo.firstDay)+1),
    %idx = find(ismember((floor(DatIn)-DSetInfo.firstDay)+1,i)); 
    
    % day starts at 3 am and ends at 3 am   
    
    idx = find((DatIn>=(DSetInfo.firstDay+i+(cutHour/24)) & DatIn<(DSetInfo.firstDay+i+1+(cutHour/24))));
    if (isempty(idx))
        continue;
    end

    FeatOut{index} = FeatIn(:,idx);
    LabOut{index} = LabIn(:,idx);
    DatOut{index} = DatIn(:,idx);
    SegOut{index} = SegIn(:,idx);
    
    index = index + 1;
end

% remIdxList = [];
% for i=1:length(LabOut),
%     if (length(unique(LabOut{i}))==1)
%         remIdxList = [remIdxList i];
%     end        
% end

% FeatOut(remIdxList) = [];
% LabOut(remIdxList) = [];
% DatOut(remIdxList) = [];
% SegOut(remIdxList) = [];

DSetInfo.numDays = length(FeatOut);
