function [AccCell, columnTitles, precDays, recDays, FMesDays, accDays] = makeExtendedAccCellDaysAvg(name, trueLabels, inferedLabels)
% 1: modelName
columnTitles = {'ModelName'};
AccCell = {name};

precDays = [];
recDays = [];
FMesDays = [];
accDays = [];
for i=1:length(inferedLabels), %number of cross validations
    tempPrec = [];
    tempRec = [];
    tempAcc = [];
    for j=1:length(inferedLabels{i}.testing.inferedLabels), % number of test days per cross validation
        tempPrec(j) = calcPrecision(trueLabels{i}.testing.inferedLabels{j}, inferedLabels{i}.testing.inferedLabels{j});
        tempRec(j) = calcRecall(trueLabels{i}.testing.inferedLabels{j}, inferedLabels{i}.testing.inferedLabels{j});
        tempAcc(j) = calcTimeSliceAccuracy(trueLabels{i}.testing.inferedLabels{j}, inferedLabels{i}.testing.inferedLabels{j});
    end
    precDays(i) = mean(tempPrec);
    recDays(i) = mean(tempRec);
    if (precDays(i)==0 || precDays(i)==0)
        FMesDays(i) = 0;
    else
        FMesDays(i) = (2*precDays(i)*recDays(i))/(precDays(i)+recDays(i));
    end
    accDays(i) = mean(tempAcc);
end
% 2 (3): prec, mean (std)
columnTitles{end+1} = {'Precision Mean'};
AccCell{end+1} = mean(precDays);
columnTitles{end+1} = {'Precision Std'};
AccCell{end+1} = std(precDays);
% 4 (5): rec, mean (std)
columnTitles{end+1} = {'Recall Mean'};
AccCell{end+1} = mean(recDays);
columnTitles{end+1} = {'Recall Std'};
AccCell{end+1} = std(recDays);
% 6 (7): fmes, mean (std)
columnTitles{end+1} = {'F-measure Mean'};
AccCell{end+1} = mean(FMesDays);
columnTitles{end+1} = {'F-measure Std'};
AccCell{end+1} = std(FMesDays);
% 8 (9): acc, mean (std)
columnTitles{end+1} = {'Accuracy Mean'};
AccCell{end+1} = mean(accDays);
columnTitles{end+1} = {'Accuracy Std'};
AccCell{end+1} = std(accDays);

