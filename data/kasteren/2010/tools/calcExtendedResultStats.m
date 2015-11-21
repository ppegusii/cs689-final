function [statStruct] = calcExtendedResultStats(ModelName, data, settings, globPars, ModelOutput, orgLabels)


%% Process results
% Concatenate labels from all days
actLabels = makeActLabels(data.activity_labels, data.as.getIDs, globPars.useIdle); 

trueLabels = concatLabelsDaysSplit(orgLabels);
modelLabels = concatLabelsDaysSplit(ModelOutput);

if (globPars.realTimeEval)
    a = repmat(trueLabels,settings.timeStepSize,1);
    trueLabelsDiscretized = a(:)';
    a = repmat(modelLabels,settings.timeStepSize,1);
    modelLabels = a(:)';
    trueLabels = concatLabelsDaysSplit(globPars.onesecGroundTruth.Labels);
    orgLabels = makeTestInfStruct(globPars.onesecGroundTruth.Labels);
    trueLabelsDiscretized=trueLabelsDiscretized(1:length(trueLabels));
    modelLabels = modelLabels(1:length(trueLabels));
    statStruct.discretAcc = sum(trueLabelsDiscretized==trueLabels)/length(trueLabels);
    for j=1:length(ModelOutput),
        for i=1:length(ModelOutput{j}.testing.inferedLabels),
            a = repmat(ModelOutput{j}.testing.inferedLabels{i},settings.timeStepSize,1);
            ModelOutput{j}.testing.inferedLabels{i} =  a(:)';
            ModelOutput{j}.testing.inferedLabels{i} = ModelOutput{j}.testing.inferedLabels{i}(1:length(orgLabels{j}.testing.inferedLabels{i}));   
        end
    end
 end

modelConfMat = calcConfMat(trueLabels', modelLabels');
if (globPars.realTimeEval)
    AccCell = {ModelName};

    diagConf = diag(modelConfMat);
    sumConf1 = sum(modelConfMat,1);
    sumConf2 = sum(modelConfMat,2);
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
    modelAccCell = AccCell;
else
    [modelAccCell, columnAccTitles, tmMnVls1, clsMnVls1, prec1, rec1,fmes1] = makeExtendedAccCellDays(ModelName, orgLabels, ModelOutput);
    statStruct.columnAccTitles = columnAccTitles;
end
[modPrecRecFmesAccCell, columnPrecRecFmesAccTitles, precDays, recDays, FMesDays, accDays] = makeExtendedAccCellDaysAvg(ModelName, orgLabels, ModelOutput);

statStruct.modelConfMat = modelConfMat;
statStruct.modelAccCell = modelAccCell;
statStruct.modPRFACell = modPrecRecFmesAccCell;
statStruct.columnPRFATitles = columnPrecRecFmesAccTitles;
statStruct.precDays = precDays;
statStruct.recDays = recDays;
statStruct.FMesDays = FMesDays;
statStruct.accDays = accDays;

statStruct.oneSecAccuracyEval = globPars.realTimeEval;
statStruct.timeStepSize = settings.timeStepSize;
statStruct.model = ModelName;
statStruct.dataset = data.name;
statStruct.config = [ModelName '-' settings.name];
statStruct.modelInfo = settings.modelInfo;