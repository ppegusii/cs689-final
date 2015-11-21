function [AccCell, columnTitles] = makeAccCell(name, trueLabels, inferedLabels, confMat)

columnTitles = {'ModelName'};
AccCell = {name};

columnTitles{end+1} = 'TimeStepAvg';
AccCell{end+1} = sum(trueLabels == inferedLabels)/length(trueLabels);

columnTitles{end+1} = 'ClassesAvg';

diagConf = diag(confMat);
sumConf = sum(confMat,2);

idx = (sumConf~=0);
classAvg = diagConf(idx)./sumConf(idx);

AccCell{end+1} = mean(classAvg);

% for i=1:length(classAvg),
%     columnTitles{end+1} = sprintf('Class%dAvg', i);
%     AccCell{end+1} = classAvg(i);
% end
