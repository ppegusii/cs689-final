function [dataStruct] = prepHouseC()

loadHouseC;

excludeList = [22,28];
idx = find(~ismember(activityStructure.id,excludeList));
activityStructure = activityStructure(idx);

% merge toileting activities into one
% idx = find(ismember(as.id,7)); % get toileting upstairs
% d = as.d;
% d(idx,3) = 4;
% as = actstruct(d);


dataStruct.name = 'HouseC';
dataStruct.ss = sensorStructure;
dataStruct.as = activityStructure;
dataStruct.sensor_labels = sensor_labels;
dataStruct.activity_labels = activity_labels;
