function structLabels = makeTestInfStruct(Labels)


for i=1:length(Labels),
    structLabels{i}.testing.inferedLabels{1} = Labels{i};
end



