clc, clear all, close all


%% load training data

load training.mat

train_features = [train_sbm,train_fnc]; % concatenate features

ds = prtDataSetClass(train_features, train_labels);
rvm = prtClassRvm;
rvm.kernels.kernelCell{2}.sigma = .75; % .75 give 88%
alg =     rvm;

out = alg.kfolds(ds);
prtScoreAuc(out)

prtScoreRoc(out)