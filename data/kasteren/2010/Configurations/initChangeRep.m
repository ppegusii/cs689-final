function [configStruct] = initChangeRep(dataStruct, globPars, varargin)

if (nargin >= 3)
    durationModel = varargin{1};
end
if (~isfield (globPars,'size1SegList'))
    globPars.size1SegList = [];
end


%% Create Feature Matrix and Labels (discritizing data)
[FeatMat, Labels, Dates, Segments] = convert2ChangeFeatMat(dataStruct.ss, dataStruct.as, globPars.timeStepSize,globPars.size1SegList, globPars);

%% Clean up data (keep/remove idle class, remove useless days, etc...)
[FeatMat, Labels, Dates, Segments] = cleanData(FeatMat, Labels, Dates, Segments, globPars, dataStruct.as);

%% Information to be used by models
configStruct.modelInfo = createModelInfo(FeatMat, Labels, Dates, Segments, globPars, durationModel);

%% Convert Featmatrix and Labels intro dayCell structure
[configStruct.FeatMat, configStruct.Labels, configStruct.Dates, configStruct.Segments, configStruct.DSetInfo] = create3amDayCellStruct(FeatMat, Labels, Dates, Segments,globPars);

if (isfield (dataStruct,'sensor_info'))
    configStruct.DSetInfo.sensor_info = dataStruct.sensor_info;
end
configStruct.name = 'ChangeRep';
configStruct.timeStepSize = globPars.timeStepSize;
configStruct.DSetInfo.name = dataStruct.name;
configStruct.DSetInfo.sensList = dataStruct.ss.getIDs;