% Ethan Lusterman
% Julia Astrauckas
% Sam Keene

% submission

%% load trained alg/classifier and feature selection object (optional)

load Llnn_rvm_default % alg

%% load test data

load testDs

%% run this shit

result = alg.run(testDs);

%% output this shit

dstr = datestr(now,1);
filename = sprintf('submission_%s.csv',dstr);
fid = fopen(filename,'w');

fprintf(fid,'Id,Probability\n');
for i = 1:length(Id)
    fprintf(fid,'%i,%f\n',Id(i),result.data(i));
end