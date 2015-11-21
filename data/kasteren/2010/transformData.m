clear outputNB outputHMM outputHSMM  outputCRF testLabelsSaves globPars

disp('Started run...');
%globPars.saveDir ='C:\temp\';
globPars.saveDir ='/home/patrick/storage/github/cs689-final/data/kasteren/2010/tmp/';
cd(globPars.saveDir);
% Load datasets
dSet={};
dSet{end+1} = prepHouseA;
dSet{end+1} = prepHouseB;
dSet{end+1} = prepHouseC;

disp(sprintf('Loaded %d datasets', length(dSet)));

% global experiment parameters
globPars.timeStepSize = 60; % seconds % size to discretize data with in seconds
globPars.stepDays = 1;      % Number of days per testset
globPars.useIdle = 1;       % Include idle class
globPars.verbose = 1;
globPars.max_iter = 25;
globPars.smallValue = 0.01;
globPars.realTimeEval = 0; % Use 1 second accuracy
globPars.cutHour = 3; % At which hour should the day start, 3 am is best because it cuts in the sleep cycle
globPars.realTimeEval = 0; % Use 1 second accuracy

%% Duration params:
globPars.typeDurData = 2;% 1: from AS, 2: from segments
globPars.numDurPars = 2;
globPars.size1SegList = [0,1,10]; % 0 = idle, rest according to as.getIDs
%globPars.size1SegList = [0,1,4,5,6,10,13,15,16,17,18,19];
durationModel =  8; %1: gamma, 2: gauss, 3)% 'Poisson', 4)% 'MOG', 5)% 'Multivariate', 7)% 'Geometric', 8)% 'Histogram'
globPars.binSize =1;
globPars.numBins =5;
globPars.useNumBins =1;
globPars.numMixtures = 2;

conf=cell(length(dSet),1);
res =cell(length(dSet),1);

%% Iterate over Datasets
for l=1:length(dSet), 
    % Load configurations
    globPars.timeStepSize = 60; % 1 minute
    conf{l}{end+1} = initBinRep(dSet{l}, globPars, durationModel);
    conf{l}{end+1} = initChangeRep(dSet{l}, globPars, durationModel);
    conf{l}{end+1} = initLastRep(dSet{l}, globPars, durationModel);
end %end cycle over datasets

outFile = sprintf('trans-ended-%s.mat', datestr(now, 'dd-mm-yyyy_HH.MM.SS'));
cd(globPars.saveDir);
save(outFile, 'conf');        

disp('Ended run');
