% Ethan Lusterman
% Julia Astrauckas
% Sam Keene

% submission

%% load trained alg/classifier and feature selection object (optional)

load featsel_best5
load alg_svm_linearkernel_featsel5features

%% load test data

load testing
features = [testSBM,testFNC];
Id = Id(2:end);

%% run this shit

ds = prtDataSetClass(features);
ds = featSel.run(ds);

result = alg.run(ds);

%% output this shit

dstr = datestr(now,1);
filename = sprintf('submission_%s.csv',dstr);
fid = fopen(filename,'w');

fprintf(fid,'Id,Probability\n');
for i = 1:length(Id)
    fprintf(fid,'%i,%i\n',Id(i),result.data(i));
end