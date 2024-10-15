addpath('..\routines\');

data_CCPP=textread('CCPP.txt'); % standardization
Xd = data_CCPP(:,1:end-1);    Yd = data_CCPP(:,end);
Xd = 10*(Xd - min(Xd))./(max(Xd) - min(Xd));

options.search_method = 'dace';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;

results = [];
fields = {'RMSE', 'ptime'};
pool = gcp('nocreate');    if isempty(pool),  parpool(4);  end
for i = 1:10
    rng(i);
    
    k = 40;    m = 40;    p = size(Xd,2);
    level = struct('S', {[]}, 'Y', {[]});
    design = repmat(level, k, 1);
    I = reshape(randperm(length(Yd),k*m), m, k);
    X = [];    Y = [];
    for j = 1 : k
        design(j).S = Xd(I(:,j),:);
        design(j).Y = Yd(I(:,j),:);
    end
    index = true(size(Yd));    index(I(:)) = false;
    Xt = Xd(index,:);    Yt = Yd(index);

    lb = [0.1*ones(1,p) 0.1];   ub = [5*ones(1,p) 3];    theta0 = [1*ones(1,p) 0.1];

    tic; dmodel = dacefit_OCL_GPU(design, @regpoly1, theta0, lb, ub, options); dmodel.mtime = toc;
    tic; [YBLUBP, ~, dmodel] = predictor_BLUBP_GPU(Xt, dmodel, 100); 
    dmodel.ptime = toc;
    dmodel.RMSE = sqrt(mean((Yt-YBLUBP).^2));

    % 打印 OCL 和 SOCL 的 RMSE
    fprintf('SOCL_full: %.4f\n', dmodel.RMSE);
    for field = fields
        result.(field{1}) = [dmodel.(field{1})]';
    end
    results = [results; result];
end
figure; boxplot([results(:).RMSE]')