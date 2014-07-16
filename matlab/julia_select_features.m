function [sbm_inds, fnc_inds] = julia_select_features(train_sbm,...
    train_fnc,train_labels)
% given both types of features, perform ttests to decide which we want
% requires labels and should only be done to training data
train_data = double(train_sbm);
schizos = train_data(train_labels==1,:);
control = train_data(train_labels==0,:);
% control_stats.mean = mean(control,1);
% control_stats.sdev = std(control,1);
% schizos_stats.mean = mean(schizos,1);
% schizos_stats.sdev = std(schizos,1);
[h,p] = ttest2(schizos, control);
% 3  7  11  16  22  24  25  26  32
sbm_inds = find(h==1);

% FNC FEATURES
train_data = double(train_fnc);
schizos = train_data(train_labels==1,:);
control = train_data(train_labels==0,:);
[h,p] = ttest2(schizos, control);
fnc_inds = find(p<.005)

end

