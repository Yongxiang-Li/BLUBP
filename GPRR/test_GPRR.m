clear
addpath('..\routines\')
addpath('..\SPGP\')

data_CCPP=textread('CCPP.txt'); % standardization
Xd=data_CCPP(:,1:end-1);    Yd = data_CCPP(:,end);
Xd = (Xd - min(Xd))./(max(Xd) - min(Xd));

k = 70;    m = 70;    p = size(Xd,2);
level = struct('S', {[]}, 'Y', {[]});
design = repmat(level, k, 1);
I = reshape(randperm(length(Yd),k*m), m, k);
X = [];    Y = [];
for i = 1 : k
    design(i).S = Xd(I(:,i),:);
    design(i).Y = Yd(I(:,i),:);
	X = [X; design(i).S];
    Y = [Y; design(i).Y];
end
index = true(size(Yd));    index(I(:)) = false;
Xt = Xd(index,:);    Yt = Yd(index);

knots = 40;
tic;
dmodelSPGP = dacefit_SPGP(X,Y,Xt,1:knots);
RMSE1 = sqrt(mean((Yt-dmodelSPGP.Yp).^2))
toc

tic;
dmodelGPRR = dacefit_GPRR(X,Y,Xt,1:knots, @corrGaussian, 1*ones(1,p),20);
RMSE2 = sqrt(mean((Yt-dmodelGPRR.Yp).^2))
toc

theta0 = [100*ones(1,p) 1];
lb = [0.1*ones(1,p) 0.1];   ub = [200*ones(1,p) 3];

tic
options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;
[dmodelIK, perfML] = dacefit(X, Y, @regpoly2, @corr_gauss_s, theta0, lb, ub, options.search_method, options.search_option);
[~, YBLUP] = predictor_BLUP(Xt, dmodelIK);
RMSE3 = sqrt(mean((Yt-YBLUP).^2))
toc

tic
options.search_method = 'dace';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;
[dmodelOCL, perfOCL] = dacefit_OCL(design, @regpoly2, @corr_gauss_s, theta0, lb, ub, options);
[~, YBLUBP] = predictor_BLUBP(Xt, dmodelOCL);
RMSE4 = sqrt(mean((Yt-YBLUBP).^2))
[~, YCLP] = predictor_CLP(Xt, dmodelOCL);
RMSE_CLP4 = sqrt(mean((Yt-YCLP).^2))
toc

tic
options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;
[dmodelCCL, ~] = dacefit_CCL(design, @regpoly2, @corr_gauss_s, theta0, lb, ub, options);
[~, YBLUBP] = predictor_BLUBP(Xt, dmodelCCL);
RMSE5 = sqrt(mean((Yt-YBLUBP).^2))
[~, YCLP] = predictor_CLP(Xt, dmodelCCL);
RMSE_CLP5 = sqrt(mean((Yt-YCLP).^2))
toc

tic
options.search_method = 'dace';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;
[dmodelCML, ~] = dacefit_CML(design, @regpoly2, @corr_gauss_s, theta0, lb, ub, options);
[~, YBLUBP] = predictor_BLUBP(Xt, dmodelCML);
RMSE6 = sqrt(mean((Yt-YBLUBP).^2))
[~, YCLP] = predictor_CLP(Xt, dmodelCML);
RMSE_CLP6 = sqrt(mean((Yt-YCLP).^2))
toc
