clear all; clc; close all
prtPath('alpha','beta')

%% load training and test data
load training.mat
% chop off some data to use as a validation set, then never touch
% SET HOLDOUT TO 0 TO TRAIN MODEL FOR SUBMISSION!
holdout = 20;
rows = size(train_sbm,1);
SBM_validation = train_sbm(1:holdout, :);
FNC_validation = train_fnc(1:holdout, :);
lab_validation = train_labels(1:holdout);

train_sbm = train_sbm(holdout+1:rows, :);
train_fnc = train_fnc(holdout+1:rows, :);
train_labels = train_labels(holdout+1:rows);

% select feature indexes
[sbm_inds, fnc_inds] = julia_select_features(...
    train_sbm,train_fnc,train_labels);

% generate feature vectors
[features, lensbm, lenfnc] = julia_get_features(train_sbm, train_fnc, ...
    sbm_inds, fnc_inds);

ds = prtDataSetClass(features, train_labels);

%% CLASSIFY
% configure individual classifiers
% SBM
nn = prtClassMatlabNnet;   
nn.Si = 2; % should probably be 2 < Si < 9

tb = prtClassMatlabTreeBagger;

% FNC
rvm = prtClassRvm;
svm_fnc = prtClassLibSvm;
svm_fnc.kernelType = 0;

% JOINT 
svm = prtClassLibSvm;
svm.kernelType = 0;

% FRANKENCLASSIFIER
feat1 = prtFeatSelStatic;
feat2 = prtFeatSelStatic;
feat1.selectedFeatures = 1:lensbm;
feat2.selectedFeatures = lensbm+1:size(ds.X,2);
% alg1 = feat1 + prtPreProcZmuv + nn;
% alg2 = feat2 + rvm;
alg1 = nn;
alg2 = svm;

%alg = alg2; % just the sbm features in a nn
alg =  alg1/alg2 + svm;

alg = alg.train(ds); % save for later
out = alg.kfolds(ds,5);
save('alg', 'alg')

figure;
prtScoreAuc(out)
prtScoreRoc(out)

% figure
% plot(out.actionCell{3})

if holdout ~= 0
    save('validating.mat','lab_validation', ...
        'SBM_validation', 'FNC_validation', ...
        'sbm_inds', 'fnc_inds');
end


