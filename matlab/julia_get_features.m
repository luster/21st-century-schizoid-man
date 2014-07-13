function [features, lensbm, lenfnc] = ...
    julia_get_features(train_sbm,train_fnc,sbm_inds, fnc_inds)

%% find features that seem to differ between classes

% SBM FEATURES
train_data = double(train_sbm);
selected_features_sbm = train_data(:,sbm_inds);

% FNC FEATURES
train_data = double(train_fnc);
selected_features_fnc = train_data(:,fnc_inds);

lensbm = size(selected_features_sbm, 2);
lenfnc = size(selected_features_fnc, 2);
features = [selected_features_sbm, selected_features_fnc];

end