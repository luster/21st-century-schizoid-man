clear all; clc; close all
prtPath('alpha','beta')

%% load training and test data
load training.mat
train_features = [train_sbm,train_fnc]; % concatenate features
ds = prtDataSetClass(train_features, train_labels);

%% find features that seem to differ between classes
% SBM FEATURES
% in the spirit of Xu, Groth, et al (2009)
train_data = double(train_sbm);
schizos = train_data(train_labels==1,:);
control = train_data(train_labels==0,:);
control_stats.mean = mean(control,1);
control_stats.sdev = std(control,1);
schizos_stats.mean = mean(schizos,1);
schizos_stats.sdev = std(schizos,1);
[h,p] = ttest2(schizos, control);

% 3  7  11  16  22  24  25  26  32
selected_features_sbm = train_data(:,h==1);

% FNC FEATURES
fncDS = prtDataSetClass(double(train_fnc), train_labels);
train_data = double(train_fnc);
schizos = train_data(train_labels==1,:);
control = train_data(train_labels==0,:);
[h,p] = ttest2(schizos, control);
selected_features_fnc = train_data(:,p<.02);

% combine both sets of selections
ds = prtDataSetClass([selected_features_sbm, selected_features_fnc], ...
    train_labels);

%% CLASSIFY
% configure individual classifiers
% SBM
nn = prtClassMatlabNnet;   
nn.Si = 7; % should probably be 2 < Si < 9

tb = prtClassMatlabTreeBagger;

% FNC
rvm = prtClassRvm;

% JOINT 
svm = prtClassLibSvm;
svm.kernelType = 0;

% FRANKENCLASSIFIER
feat1 = prtFeatSelStatic;
feat2 = prtFeatSelStatic;
ind = size(selected_features_sbm,2)+1;
feat1.selectedFeatures = [1:size(selected_features_sbm,2)];
feat2.selectedFeatures = [ind:size(ds.X,2)];
alg1 = feat1 + prtPreProcZmuv + nn;
alg2 = feat2 + rvm;

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




