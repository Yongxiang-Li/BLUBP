rng('default');    addpath('..\routines\');

options.search_method = 'dace';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = true;
options.logOn = false;
if options.FitInParallel && ~exist('cluster','var')
    cluster = parpool('local',4); 
end


ojbFun = @peak_fun;
n = 500;    k = 20;    p = 2;
lb = [0.01*ones(1,p), 0.01];    ub = [20*ones(1,p),1];     theta0 = (lb + ub) /2;
design = generate_design(n/k, p, k);
for k = 1 : length(design), design(k).Y = ojbFun(design(k).S); end  
tic; 
[dmodel,~] = dacefit_OCL_GPU(design, @regpoly0, theta0, lb, ub, options);
toc;

Xt = meshgrid_my(linspace(0,1,51),linspace(0,1,51));
Yhat = predictor_BLUBP_GPU(Xt, dmodel);     

figure; plot3(Xt(:,1), Xt(:,2), Yhat, 'd')


function Y = peak_fun(X)
    n = size(X,1);
    Y = peaks(6*X(:,1)-3, 6*X(:,2)-3) + randn(n,1)/10;
end

function [ design ] = generate_design(n, p, k)
    Sdata = slhd(n, k, p);
    level = struct('S', {[]}, 'Y', {[]});
    design = repmat(level, k, 1);
    for i = 1 : k
        design(i).S = Sdata(:,:,i);
    end
end

function [ X ] = meshgrid_my(X1, X2)
    [X1, X2] = meshgrid(X1, X2);
    X = [X1(:) X2(:)];
end