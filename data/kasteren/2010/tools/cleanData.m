function [FeatMat, Labels, Dates, Segments] = cleanData(FeatMat, Labels, Dates, Segments, globPars, as)


%% Correct Idle activity
actList = unique(Labels);
if (ismember(0,actList)) 
    if (globPars.useIdle)
        Labels = Labels + 1;
    else
        idx = find(Labels ~= 0);
        Labels = Labels(idx);
        FeatMat= FeatMat(:,idx);
        Dates = Dates(idx);
        Segments = Segments(idx);
    end          
end
actList = unique(Labels);

%% Filter days without training data
%% do this in create3amDaycell
% numDays = ceil(Dates(end) - Dates(1));
% firstDay = floor(Dates(1));
% for i=1:numDays,
%     % Calculate when first day was
%     curDate = firstDay+(i-1);
% 
%     if (curDate~= floor(as.start) & curDate~= floor(as.end))
%        idxInclude = find(~ismember((floor(Dates)),curDate));
%         Labels = Labels(idxInclude);
%         FeatMat= FeatMat(:,idxInclude);
%         Dates = Dates(idxInclude);
%         Segments= Segments(idxInclude);
%     end
% end


%% Reduce the idle space to percentage% (default=10%)
% percentage = 0.1;
% throwOut = [];
% startLenLabs = length(Labels);
% startDist = histc(Labels, actList);
% if (isfield(globPars,'reduceIdle'))
%     if (globPars.reduceIdle == 1 & globPars.useIdle)
%         idx = find(Labels==1);
%         idx2 = find(idx(1:end-1)~=idx(2:end)-1);
%         startPoint = idx(1);
%         for i=1:length(idx2),
%             endPoint = idx(idx2(i));
%             len = endPoint - startPoint;
%             redLen = floor((percentage * len)/2);
%             throwOut = [throwOut (startPoint+redLen):(endPoint-redLen)];
%                         
%             startPoint = idx(idx2(i)+1);                    
%         end
%     end
%     idx = find(~ismember(1:length(Labels),throwOut));
%     
%     Labels = Labels(idx);
%     FeatMat= FeatMat(:,idx);
%     Dates = Dates(idx);
%     Segments = Segments(idx);
% 
%     endLenLabs = length(Labels);
%     endDist = histc(Labels, actList);
%     disp(sprintf('Reduced dataset size from %d to %d timeslices, by reducing idle to %d%%',startLenLabs,endLenLabs,percentage*100));
%     szDist = sprintf('%6.2f%%,',100*startDist/sum(startDist)); 
%     disp(sprintf('Distribution at start: %s',szDist));
%     szDist = sprintf('%6.2f%%,',100*endDist/sum(endDist));
%     disp(sprintf('Distribution at end: %s',szDist));
% end

%% Throw out data between xxh and yyh
if (isfield(globPars,'startHour') & isfield(globPars,'endHour'))
    startLenLabs = length(Labels);
    startDist = histc(Labels, actList);

    idx = find( (Dates-floor(Dates)) >= globPars.startHour/24 & (Dates-floor(Dates)) <= globPars.endHour/24);
    
    Labels = Labels(idx);
    FeatMat= FeatMat(:,idx);
    Dates = Dates(idx);
    Segments = Segments(idx);
    
    endLenLabs = length(Labels);
    endDist = histc(Labels, actList);
    disp(sprintf('Reduced dataset size from %d to %d timeslices, by keeping data between %dh and %dh%',startLenLabs,endLenLabs,globPars.startHour,globPars.endHour));
    szDist = sprintf('%6.2f%%,',100*startDist/sum(startDist)); 
    disp(sprintf('Distribution at start: %s',szDist));
    szDist = sprintf('%6.2f%%,',100*endDist/sum(endDist));
    disp(sprintf('Distribution at end: %s',szDist));
end






