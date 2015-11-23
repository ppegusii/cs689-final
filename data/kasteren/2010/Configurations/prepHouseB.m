function [dataStruct] = prepHouseB()

loadHouseB;

excludeList = [9  16 29 33 34  35 36 37 38 40 42 43 44];
idx = find(~ismember(activityStructure.id,excludeList));
activityStructure = activityStructure(idx);

    
dataStruct.name = 'HouseB';
dataStruct.ss = sensorStructure;
dataStruct.as = activityStructure;
dataStruct.sensor_labels = sensor_labels;
dataStruct.activity_labels = activity_labels;
