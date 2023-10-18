clear
addpath('..\routines\')

p = 8;
k = 10;
N = 80;
Nt = 10000;
Sdata  = slhd(N, k, p);
X = [];    Y = [];
level = struct('S', {[]}, 'Y', {[]});
design = repmat(level, k, 1);
trail = level;
trail.S = lhsdesign(Nt, p);
trail.Y = fun_BH(trail.S);
for j = 1 : k
    design(j).S = Sdata(:,:,j);
    design(j).Y = fun_BH(design(j).S);
    X = [X; design(j).S];
    Y = [Y; design(j).Y];
end
tic;
dmodelGPRR = dacefit_GPRR(X,Y,trail.S,1:N, @corrGaussian, 0.1*ones(1,p),10);
MSE1 = mean((trail.Y-dmodelGPRR.Yp).^2)
toc




tic
theta0 = [0.1*ones(1,p) 0.1];
lb = [0.01*ones(1,p) 0.0001];   ub = [5*ones(1,p) 1];
options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;
[dmodelIK, perfML] = dacefit(X, Y, @regpoly0, @corr_gauss_s, theta0, lb, ub, options.search_method, options.search_option);
[~, YBLUP] = predictor_BLUP(trail.S, dmodelIK);
MSE2 = mean((trail.Y-YBLUP).^2)
toc

tic
options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;
[dmodelOCL, perfOCL] = dacefit_OCL(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
[~, YBLUBP] = predictor_BLUBP(trail.S, dmodelOCL);
MSE3 = mean((trail.Y-YBLUBP).^2)
toc

% dmodelOCL.theta = dmodelIK.theta;
% dmodelOCL.beta = dmodelIK.beta;
% dmodelOCL.sigma2 = dmodelIK.sigma2;
% [~, YBLUBP] = predictor_BLUBP(trail.S, dmodelOCL);
% MSE3 = mean((trail.Y-YBLUBP).^2)

% [dmodelIK, perfML] = dacefit(X, Y, @regpoly0, @corr_gauss_x, dmodelOCL.theta);
% [~, YBLUP] = predictor_BLUP(trail.S, dmodelIK);
% MSE2 = mean((trail.Y-YBLUP).^2)

