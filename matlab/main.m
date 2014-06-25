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

classifier = prtClassLibSvm;

train_features = [train_sbm,train_fnc]; % concatenate features

ds = prtDataSetClass(train_features, train_labels);
zmuv = prtPreProcZmuv;
zmuv = zmuv.train(ds);
ds = zmuv.run(ds);

% feature selection
featSel = prtFeatSelSfs;
featSel.evaluationMetric = @(DS)prtEvalAuc(classifier,DS);
featSel.nFeatures = 3;
% featSel = prtFeatSelStatic;
% featSel.selectedFeatures = bigmlfeatures;

featSel = featSel.train(ds);
dsFeatSel = featSel.run(ds);

%% train classifier

% preprocessing
zmuv = prtPreProcZmuv;

clf = classifier;
clf.internalDecider = prtDecisionBinaryMinPe;
alg = zmuv + clf;

alg = alg.train(dsFeatSel);

%% cross-validate using leave-one-out

result = alg.kfolds(dsFeatSel, length(dsFeatSel.targets));

%% results

figure(1)
prtScoreConfusionMatrix(result)

[~, ~, ~, auc ] = perfcurve(train_labels, result.data, 1)