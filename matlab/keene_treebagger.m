% main - keene

clear all;clc; close all
prtPath('alpha','beta')

%% load training data

load training.mat

train_features = [train_sbm,train_fnc]; % concatenate features

ds = prtDataSetClass(train_features, train_labels);

%% try some feature selection shit
% LLN
% doing this gives 86%, might be more stable than full features
featSel = prtFeatSelLlnn('verbosePlot',true);  % Create the feature
% selection object.
featSel.nMaxIterations = 10;                   % Set the max # of
% iterations.
featSel = featSel.train(ds);              % Train
featSel. weightChangeThreshold = .1;
features = featSel.selectedFeatures; % 7 11 24 26 276

ds = featSel.run(ds);

%%
 
% rvm = prtClassRvm;  % both kinds of RVMs give ~85%
% rvm.kernels.kernelCell{2}.sigma = .75; % .75 give 88%
% alg =      rvm;


%  deep = prtClassRasmusbergpalmDeepLearningNn('batchsize', 85);
%  alg = deep; 

alg = prtClassMatlabTreeBagger;  % 97 fuckin percent
out = alg.kfolds(ds);

figure;
prtScoreAuc(out)

prtScoreRoc(out)



