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

    if(globPars.realTimeEval) % create 1 sec accuracy labels
        realStepsize = globPars.timeStepSize;
        globPars.timeStepSize = 1;
        globPars.onesecGroundTruth = initChangeRep(dSet{l}, globPars, durationModel);
        globPars.timeStepSize = realStepsize;
    end
    
    disp(sprintf('Experimenting with dataset %d (out of %d) using %d types of configurations', l, length(dSet), length(conf{l})));
%% Iterate over experiment configs
    for k=1:length(conf{l}), 
        clear outputNB outputHMM outputHSMM outputCRF testLabelsSaves curExp
        
        curSet = conf{l}{k};
        conf{l}{k} = {}; %Clear some memory
        availDays = 1:curSet.DSetInfo.numDays;

        disp(sprintf('D%d/%d: Running experiment %d (out of %d), processing %d days', l,length(dSet),k, length(conf{l}), curSet.DSetInfo.numDays));
        index = 1;
%% Perform inference for all models and days
        for i=1:globPars.stepDays:curSet.DSetInfo.numDays,
            
            curExp.name = sprintf('day%dconf%ddataset%d',i,k,l);
            tic;
            testDays = i:min((i+(globPars.stepDays-1)),curSet.DSetInfo.numDays);
            trainDays = availDays(~ismember(availDays,testDays));
            
            % Create structure for current experiment
            curExp.trainFeatMat = curSet.FeatMat(trainDays);
            curExp.trainLabels = curSet.Labels(trainDays);
            curExp.trainSegments = curSet.Segments(trainDays);
            if (size(curExp.trainFeatMat,2)==0)
                error('Empty training matrix');
            end
            
            curExp.testFeatMat = curSet.FeatMat(testDays);
            curExp.testLabels = curSet.Labels(testDays);
            curExp.testSegments = curSet.Segments(testDays);

            if (size(curExp.testFeatMat,2)==0)
                error('Empty test matrix');
            end
            
            curExp.modelInfo = curSet.modelInfo;
            if (i>1 & exist('outputCRF'))
                curExp.initialCRFLambda = outputCRF{i-1}.training.learnedParams.lambdas;
            end

            %% Save original labels so we can evaluate results
            testLabelsSaves{index}.testing.inferedLabels = curExp.testLabels;


            % Generative
            outputNB{index} = TrainTestDaysSplit('nb', curExp);
            outputHMM{index} = TrainTestDaysSplit('hmm', curExp);
            outputHSMM{index} = TrainTestDaysSplit('hsmm', curExp);
            outputCRF{index} = TrainTestDaysSplit('linchaincrf', curExp);

              if (globPars.stepDays==1)
                  disp(sprintf('D%d/%d: C%d/%d: Processed day %d (out of %d)',l,length(dSet),k,length(conf{l}),i, curSet.DSetInfo.numDays));
              else
                  disp(sprintf('D%d/%d: C%d/%d: Processed days %s (out of %d)',l,length(dSet),k,length(conf{l}),num2str(testDays), curSet.DSetInfo.numDays));
              end
              
              index = index + 1;        
    
        end %end cycle over days
        res{l}{k} = {};

        
%%      Save results and stats

        %% RESULTS
       if (exist('outputNB'))
            res{l}{k}.statNB = calcExtendedResultStats('nb', dSet{l}, curSet, globPars, outputNB, testLabelsSaves);
        end
        if (exist('outputHMM'))
            res{l}{k}.statHMM = calcExtendedResultStats('hmm', dSet{l}, curSet, globPars, outputHMM, testLabelsSaves);
        end
        if (exist('outputHSMM'))
            res{l}{k}.statHSMM = calcExtendedResultStats('hsmm', dSet{l}, curSet, globPars, outputHSMM, testLabelsSaves);
        end
        if (exist('outputCRF'))
            res{l}{k}.statCRF = calcExtendedResultStats('crf', dSet{l}, curSet, globPars, outputCRF, testLabelsSaves);
        end
        res{l}{k}.testLabelsSaves = testLabelsSaves;
        outFile = sprintf('out%s-D%dC%d.mat', datestr(now, 'dd-mm-yyyy_HH.MM.SS'), l,k);
        toWrite = res{l}{k};
        cd(globPars.saveDir);
        save(outFile, 'toWrite');        
 	end %end cycle over configs
end %end cycle over datasets

outFile = sprintf('ended%s-D%dC%d.mat', datestr(now, 'dd-mm-yyyy_HH.MM.SS'), l,k);
cd(globPars.saveDir);
save(outFile, 'res');        

disp('Ended run');