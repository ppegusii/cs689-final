function allLabels = concatLabelsDaysSplit(outputStruct)

numIters = length(outputStruct);

allLabels = [];

for i=1:numIters,
    if (isstruct(outputStruct{i}))
        for j=1:length(outputStruct{i}.testing.inferedLabels),
            allLabels = [allLabels outputStruct{i}.testing.inferedLabels{j}];
        end
    else
        allLabels = [allLabels outputStruct{i}];
    end
end