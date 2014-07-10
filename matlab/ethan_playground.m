clear all; clc; close all
prtPath('alpha','beta')

%% load training data
load training.mat
train_features = [train_sbm,train_fnc];
idx_sbm = 1:32;
idx_fnc = 33:(length(train_features-32));
ds = prtDataSetClass(train_features, train_labels);
p = randperm(86);
hold_out = 20;
ds_val = ds.retainObservations(p(1:hold_out));
ds = ds.removeObservations(p(1:hold_out));

%% features

sbm = prtFeatSelStatic;
sbm.selectedFeatures = [3  7  11  16  22  24  25  26  32];
fnc = prtFeatSelStatic;
llnn = [243 244];
genalg = [48 102 165 193 223 265 279 308 321 328];
sfs = [183 279 337];
fnc.selectedFeatures = [sfs,llnn,genalg];


%% sheeitttt

zmuv = prtPreProcZmuv;
rvm = prtClassRvm;
tb = prtClassTreeBaggingCap;
svm = prtClassLibSvm;
svm.kernelType = 0;
sbm_alg = sbm + svm;
fnc_alg = fnc + svm;

alg = sbm_alg/fnc_alg + svm;

alg_base = svm;
alg_base = alg_base.train(ds);
baseline = alg_base.run(ds_val);
baseline = prtScoreAuc(baseline)

kfold = alg.kfolds(ds);

kfolds = prtScoreAuc(kfold)
prtScoreRoc(kfold)

%% validate

alg = alg.train(ds);
out = alg.run(ds_val);

validate = prtScoreAuc(out)
prtScoreRoc(out)
