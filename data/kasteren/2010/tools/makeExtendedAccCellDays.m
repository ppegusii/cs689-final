function [AccCell, columnTitles, timeMeansVals, classMeanVals, precVals, recVals,FMes, precValsDays, recValsDays, FMesDays] = makeExtendedAccCellDays(name, trueLabels, inferedLabels)

columnTitles = {'ModelName'};
AccCell = {name};

timeMeansVals = [];
classMeanVals = [];
precVals = [];
recVals = [];
recValsDays = [];
precValsDays = [];
modelLabs = [];
trueLabs = [];
for i=1:length(inferedLabels),
    for j=1:length(inferedLabels{i}.testing.inferedLabels),
        modelLabs = [modelLabs inferedLabels{i}.testing.inferedLabels{j}];
        trueLabs = [trueLabs trueLabels{i}.testing.inferedLabels{j}];
        timeMeansVals(end+1) = mean(inferedLabels{i}.testing.inferedLabels{j} == trueLabels{i}.testing.inferedLabels{j});
        tempConfMat = calcConfMat(trueLabels{i}.testing.inferedLabels{j}', inferedLabels{i}.testing.inferedLabels{j}');
        diagConf = diag(tempConfMat);
        sumConf = sum(tempConfMat,2);
        idx = (sumConf~=0);
        classMeanVals(end+1) = mean(diagConf(idx)./sumConf(idx));
        
        if (sum(diagConf)==0)
            recValsDays(end+1) = 1/length(trueLabs);
        else
            recValsDays(end+1) = mean(diagConf(idx)./sumConf(idx));
        end
        
        sumConf1 = sum(tempConfMat,1);
        idx = (sumConf1==0);
        sumConf1(idx)=1;
        precValsDays(end+1) = mean(diagConf./sumConf1');

    end
end

FMesDays = 2*(precValsDays.*recValsDays)./(precValsDays+recValsDays);

AccCell{end+1} = mean(timeMeansVals);
AccCell{end+1} = std(timeMeansVals);
AccCell{end+1} = mean(classMeanVals);
AccCell{end+1} = std(classMeanVals);

tempConfMat = calcConfMat(trueLabs',modelLabs');
diagConf = diag(tempConfMat);
sumConf1 = sum(tempConfMat,1);
sumConf2 = sum(tempConfMat,2);
idx = (sumConf1==0);
sumConf1(idx)=1;
precVals = mean(diagConf./sumConf1');
idx = (sumConf2~=0);
recVals = mean(diagConf(idx)./sumConf2(idx));

AccCell{end+1} = precVals;
AccCell{end+1} = recVals;

if ((precVals+recVals)==0)
    FMes = 0;
else
    FMes = 2*(precVals.*recVals)./(precVals+recVals);
    idx = find (isnan(FMes));
    FMes(idx) = 0;
end
AccCell{end+1} = FMes;

% AccCell{end+1} = mean(classAvg);
% 
% % Recall
% AccCell{end+1} = mean(diag(confMat)./ sum(confMat,1)');
% % Precision
% AccCell{end+1} = mean(diag(confMat)./ sum(confMat,2));
% 
% AccCell{end+1} = var(trueLabels == inferedLabels);
% % for i=1:length(classAvg),
% %     columnTitles{end+1} = sprintf('Class%dAvg', i);
% %     AccCell{end+1} = classAvg(i);
% % end

