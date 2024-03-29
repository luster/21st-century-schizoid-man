% Ethan Lusterman
% Julia Astrauckas
% Sam Keene

% submission

%% load trained alg/classifier and feature selection object (optional)
load alg % alg

%% load test data
load testing.mat
Id = double(SBM_test(:,1));
sbm_features = double(SBM_test(:,2:size(SBM_test,2)));
fnc_features = double(FNC_test(:,2:size(FNC_test,2)));
sbm_inds = [3  7  11  16  22  24  25  26  32];
fnc_inds = [33 37 48 61  4 78 171 183 189 211 220 226 244 295 ...
    302 328 333 353]; % .01 threshold 

sbm_features = sbm_features(:,sbm_inds);
fnc_features = fnc_features(:,fnc_inds);
testDs = prtDataSetClass([sbm_features, fnc_features]);

%% run this shit

result = alg.run(testDs);

%% output this shit
statout = result.data;
minstat = min(statout);
if minstat < 0
    statout = statout + abs(minstat);
end
maxstat = max(statout);
statout = statout / maxstat;

dstr = datestr(now,1);
filename = sprintf('submission_%s.csv',dstr);
fid = fopen(filename,'w');

fprintf(fid,'Id,Probability\n');
for i = 1:length(Id)
    fprintf(fid,'%i,%f\n',Id(i),statout(i));
end