% Ethan Lusterman
% Julia Astrauckas
% Sam Keene
%
%   main.m
%
%       - train [classifier-type] and cross-validate using [k] folds
%       - experiment with different feature sets
%

%% load training data

load training.mat

%% feature extraction

% generated feature importances ranked most important to least important from 
% bigml.com
bigmlfeatures = [326, 296, 79, 28, 220, 288, 390] + 1;

train_features = [train_sbm,train_fnc]; % concatenate features

ds = prtDataSetClass(train_features, train_labels);
% zmuv = prtPreProcZmuv;
% zmuv = zmuv.train(ds);
% ds = zmuv.run(ds);

classifier = prtClassLibSvm;
classifier.kernelType = 0;
% classifier = prtClassRvm;

% feature selection
featSel = prtFeatSelSfs;
featSel.evaluationMetric = @(DS)prtEvalAuc(classifier,DS);
featSel.nFeatures = 5;
% featSel = prtFeatSelStatic;
% featSel.selectedFeatures = bigmlfeatures;

featSel = featSel.train(ds);
dsFeatSel = featSel.run(ds);
% dsFeatSel = ds;

%% train classifier

% preprocessing


%classifier = prtClassLibSvm;
% classifier.kernels.kernelCell{2}.sigma = .35; % .75 give 88%
clf = classifier;
clf.kernelType = 0;
alg = clf;
% alg = clf;

alg = alg.train(ds);

%% cross-validate using leave-one-out

result = alg.kfolds(ds);

%% results

figure(1)
prtScoreConfusionMatrix(result)

[~, ~, ~, auc ] = perfcurve(train_labels, result.data, 1)