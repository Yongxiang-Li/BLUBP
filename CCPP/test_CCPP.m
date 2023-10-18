clear
addpath('..\routines\')
addpath('..\SPGP\')
addpath('..\GPRR\')
rng('default');

data_CCPP=textread('CCPP.txt'); % standardization
Xd = data_CCPP(:,1:end-1);    Yd = data_CCPP(:,end);
Xd = 10*(Xd - min(Xd))./(max(Xd) - min(Xd));

options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = true;
options.logOn = true;
iter = 20;

results = [];

k = 80;    m = 80;    p = size(Xd,2);
for i = 1 : iter
  	if mod(i,10) == 0
        disp(i);
    end
    
    level = struct('S', {[]}, 'Y', {[]});
    design = repmat(level, k, 1);
    I = reshape(randperm(length(Yd),k*m), m, k);
    X = [];    Y = [];
    for j = 1 : k
        design(j).S = Xd(I(:,j),:);
        design(j).Y = Yd(I(:,j),:);
        X = [X; design(j).S];
        Y = [Y; design(j).Y];
    end
    index = true(size(Yd));    index(I(:)) = false;
    Xt = Xd(index,:);    Yt = Yd(index);
    
    %SPGP
    dmodelSPGP = struct;
    knots = 80;
    tic;
    YSPGP = dacefit_SPGP(X,Y,Xt,1:knots);
    dmodelSPGP.RMSE = sqrt(mean((Yt-YSPGP.Yp).^2));
    dmodelSPGP.time = toc;

    %GPRR
    dmodelGPRR = struct;
    tic;
    YGPRR = dacefit_GPRR(X,Y,Xt,1:knots, @corrGaussian, 1*ones(1,p),20);
    dmodelGPRR.RMSE = sqrt(mean((Yt-YGPRR.Yp).^2));
    dmodelGPRR.time = toc;

    theta0 = [1*ones(1,p) 1];
    lb = [0.1*ones(1,p) 0.1];   ub = [10*ones(1,p) 3];
    
    tic;
    [dmodelOCL, perfOCL] = dacefit_OCL(design, @regpoly2, @corr_gauss_s, theta0, lb, ub, options);
    tOCL = toc;
    tic;
    [dmodelCML, perfCML] = dacefit_CML(design, @regpoly2, @corr_gauss_s, theta0, lb, ub, options);
    tCML = toc;
    tic;
    [dmodelCCL, perfCCL] = dacefit_CCL(design, @regpoly2, @corr_gauss_s, theta0, lb, ub, options);
    tCCL = toc;
    tic;
    [dmodelIK, perfML] = dacefit(X, Y, @regpoly2, @corr_gauss_s, theta0, lb, ub, options.search_method, options.search_option);
    tML = toc;    
    modelTime = [tML, tOCL, tCML, tCCL];
    
    % BLUP
    dmodels = {dmodelIK dmodelOCL,dmodelCML,dmodelCCL};
    models = repmat(struct,length(dmodels),1);
    for j = 1 : length(dmodels)
        dmodel = dmodels{j};
        if j>1
            dmodelML = dmodelIK;
            dmodelML.theta = dmodel.theta;
            dmodelML.beta = dmodel.beta;
            dmodelML.sigma2 = dmodel.sigma2;
            dmodel = dmodelML;
        end
        tic;
        [~, YBLUP] = predictor_BLUP(Xt, dmodel);
        tBLUP = toc;
        models(j).theta = dmodel.theta(:);
        models(j).beta = dmodel.beta(:);
        models(j).sigma2 = dmodel.sigma2(:);
        models(j).mtime = modelTime(j);
        BLUP = struct;
        BLUP.ptime = tBLUP;
        BLUP.RMSE = sqrt(mean((Yt-YBLUP).^2));
        BLUP.time = modelTime(j)+tBLUP;
        models(j).BLUP = BLUP;
    end
    
    % CLP and BLUBP
    dmodelML = dmodelOCL;
    dmodelML.theta = dmodelIK.theta;
    dmodelML.beta = dmodelIK.beta;
    dmodelML.sigma2 = dmodelIK.sigma2;
    dmodels = {dmodelML dmodelOCL,dmodelCML,dmodelCCL};
    for j = 1 : length(dmodels)
        dmodel = dmodels{j};
        tic;
        [~, YCLP] = predictor_CLP(Xt, dmodel);
        tCLP = toc;
        CLP = struct;
        CLP.ptime = tCLP;
        CLP.time = modelTime(j)+tCLP;
        CLP.RMSE = sqrt(mean((Yt-YCLP).^2));
        models(j).CLP = CLP;
        
        tic;
        [~, YBLUBP] = predictor_BLUBP(Xt, dmodel);
        tBLUBP = toc;
        BLUBP = struct;
        BLUBP.ptime = tBLUBP;
        BLUBP.RMSE = sqrt(mean((Yt-YBLUBP).^2));
        BLUBP.time = modelTime(j)+tBLUBP;
        models(j).BLUBP = BLUBP;
    end

    result = struct();
    result.dmodelML = models(1);
    result.dmodelOCL = models(2);
    result.dmodelCML = models(3);
    result.dmodelCCL = models(4);
    result.dmodelSPGP = dmodelSPGP;
    result.dmodelGPRR = dmodelGPRR;
    result.design = design;
    result.Xt = Xt;
    result.Yt = Yt;
    result.i = i;
    results = [results; result];
end
save('Results_CCPP.mat','results')
