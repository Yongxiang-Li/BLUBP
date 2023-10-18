addpath('../routines/')
rng('default')
if ~exist('data','var')
   load('data.csv');
end

k = 200;    m = 200;
data = data(randperm(size(data,1)),:);
X = data(1:k*m,1:end-1); Y = data(1:k*m,end);
design = struct('S',nan,'Y',nan);
I = reshape(1:m*k, m, k);
for j = 1 : k
    design(j).S = X(I(:,j),:);
    design(j).Y = Y(I(:,j),:);
end
trail = struct('S',data(k*m+1:k*m+100000,1:end-1),'Y',data(k*m+1:k*m+100000,end));


p = size(design(1).S,2);
theta0 = [0.1*ones(1,p) 0.05];
lb = [1e-4*ones(1,p) 0.01];   ub = [5*ones(1,p) 0.5];

options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = true;
options.logOn = true;
poolobj = gcp;    addAttachedFiles(poolobj,{'regpoly0.m','corr_gauss_s.m'})

tic
[dmodelCML, ~] = dacefit_CML(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
dmodelCML.mtime = toc;
tic;
[dmodelCML_CLP, YCLP] = predictor_CLP(trail.S, dmodelCML, false, 4000);
dmodelCML_CLP.RMSE = sqrt(mean((trail.Y-YCLP).^2));
dmodelCML_CLP.ptime = toc;
tic;
[dmodelCML_BLUBP, YBLUBP] = predictor_BLUBP(trail.S, dmodelCML, 4000);
dmodelCML_BLUBP.RMSE = sqrt(mean((trail.Y-YBLUBP).^2));
dmodelCML_BLUBP.ptime = toc;

result = struct;
result.design = design;    result.trail = trail;
result.dmodelCML = dmodelCML;
result.dmodelCML_CLP = dmodelCML_CLP;
result.dmodelCML_BLUBP = dmodelCML_BLUBP;
save(['Distributed_CML_' char(dmodelCML_BLUBP.regr) '_k=' num2str(k) 'm=' num2str(m) '-' datestr(now,'yyyymmddHHMMSS')], 'result','theta0','lb','ub')

