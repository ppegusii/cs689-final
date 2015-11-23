function [configStruct] = initLastRep(dataStruct, globPars, varargin)

if (nargin >= 3)
    durationModel = varargin{1};
end

%% Create Feature Matrix and Labels (discritizing data)
[FeatMat, Labels, Dates, Segments] = convert2LastFiredFeatMat(dataStruct.ss, dataStruct.as, globPars.timeStepSize,globPars.size1SegList);

%% Clean up data (keep/remove idle class, remove useless days, etc...)
[FeatMat, Labels, Dates, Segments] = cleanData(FeatMat, Labels, Dates, Segments, globPars, dataStruct.as);

%% Information to be used by models
configStruct.modelInfo = createModelInfo(FeatMat, Labels, Dates, Segments, globPars, durationModel);

%% Convert Featmatrix and Labels intro dayCell structure
[configStruct.FeatMat, configStruct.Labels, configStruct.Dates, configStruct.Segments, configStruct.DSetInfo] = create3amDayCellStruct(FeatMat, Labels, Dates, Segments, globPars);

configStruct.name = 'LastRep';
configStruct.timeStepSize = globPars.timeStepSize;