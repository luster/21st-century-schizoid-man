clear;close all

%%

load training.mat
ds_fnc = prtDataSetClass(train_fnc, train_labels);
ds_sbm = prtDataSetClass(train_sbm, train_labels);

nstdout = prtOutlierRemovalNStd('runMode', 'replaceWithNan');
nstdout = nstdout.train(ds_fnc);
ds_fnc = nstdout.run(ds_fnc);
logical_remove = any(isnan(ds_fnc.getX),2);
idx_remove_fnc = find(logical_remove);

nstdout = prtOutlierRemovalNStd('runMode', 'replaceWithNan');
nstdout = nstdout.train(ds_sbm);
ds_sbm = nstdout.run(ds_sbm);
logical_remove = any(isnan(ds_sbm.getX),2);
idx_remove_sbm = find(logical_remove);

intscn = intersect(idx_remove_fnc,idx_remove_sbm);
ds = prtDataSetClass([train_sbm,train_fnc], train_labels);
ds = ds.removeObservations(intscn);

ds = ds.bootstrapByClass(min(ds.nObservationsByClass));

%%

feat1 = prtFeatSelStatic;
feat1.selectedFeatures = 1:32;
feat2 = prtFeatSelStatic;
feat2.selectedFeatures = 33:size(ds.getX,2);

%%

svm = prtClassLibSvm('kernelType',2);
svm_out = prtClassLibSvm('kernelType',0);
rvm = prtClassRvm;
nn = prtClassMatlabNnet('Si',5);
zmuv = prtPreProcZmuv;

alg1 = feat1 + zmuv + svm;
alg2 = feat2 + zmuv + svm;
alg = alg1/alg2 + svm_out;

%%

kfolds = alg.kfolds(ds);

prtScoreAuc(kfolds)
prtScoreRoc(kfolds)

alg = alg.train(ds);