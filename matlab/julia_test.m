clc, clear all, close all

% MUST HAVE RUN TRAINING SCRIPT WITH NONZERO HOLDOUT SIZE for this to work
load('alg');
load('validating.mat');

% generate feature vectors
[features, lensbm, lenfnc] = julia_get_features(SBM_validation, ...
    FNC_validation, sbm_inds, fnc_inds);

ds = prtDataSetClass(features, lab_validation);
out = alg.run(ds);
prtScoreRoc(out)
prtScoreAuc(out)