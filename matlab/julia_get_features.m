function [features, lensbm, lenfnc] = ...
    julia_get_features(train_sbm,train_fnc,train_labels)

%% find features that seem to differ between classes

% SBM FEATURES
% in the spirit of Xu, Groth, et al (2009)
train_data = double(train_sbm);
schizos = train_data(train_labels==1,:);
control = train_data(train_labels==0,:);
% control_stats.mean = mean(control,1);
% control_stats.sdev = std(control,1);
% schizos_stats.mean = mean(schizos,1);
% schizos_stats.sdev = std(schizos,1);
[h,p] = ttest2(schizos, control);
% 3  7  11  16  22  24  25  26  32
selected_features_sbm = train_data(:,h==1);

% FNC FEATURES
fncDS = prtDataSetClass(double(train_fnc), train_labels);
train_data = double(train_fnc);
schizos = train_data(train_labels==1,:);
control = train_data(train_labels==0,:);
[h,p] = ttest2(schizos, control);
selected_features_fnc = train_data(:,p<.01);

lensbm = size(selected_features_sbm, 2);
lenfnc = size(selected_features_fnc, 2);
features = [selected_features_sbm, selected_features_fnc];

end