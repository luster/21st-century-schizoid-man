%% deep learning shit
clear;clc;close all
prtPath('alpha','beta')

%%
load training

ds = prtDataSetClass(train_fnc, train_labels);
% deep = prtPreProcDeepLearning;
% deep.layerSpec = 3;
% deep = deep.train(ds);
% dout = deep.run(ds)
% plot(dout)



feat = prtFeatSelSfs;
feat.nFeatures = 3;
feat.evaluationMetric = @(DS)prtEvalAuc(prtClassLibSvm('kernelType',0),DS)
feat = feat.train(ds);
ds_out = feat.run(ds);

%%

svm = prtClassLibSvm;
svm.kernelType = 0;

alg = svm;

k = alg.kfolds(ds);

prtScoreAuc(k)
prtScoreRoc(k)