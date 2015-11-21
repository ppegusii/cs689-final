function [dataStruct] = prepHouseA()

loadHouseA;

excludeList = [3,18,19,20,22,23,25];  % Used for bookchapter Atlantic press
idx = find(~ismember(activityStructure.id,excludeList));
activityStructure = activityStructure(idx);

dataStruct.name = 'HouseA';
dataStruct.ss = sensorStructure;
dataStruct.as = activityStructure;
dataStruct.sensor_labels = sensor_labels;
dataStruct.activity_labels = activity_labels;