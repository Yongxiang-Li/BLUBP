clear
addpath('..\routines\')
addpath('..\SPGP\')
addpath('..\GPRR\')

results = [];
for s = 1 : 20
    k = 150; m = 200;
    [design, trail] = get_proteindata(k, m);

    p = size(design(1).S,2);
    theta0 = [ones(1,p) 0.5];
    lb = [0.01*ones(1,p) 0.01];   ub = [100*ones(1,p) 2];

    options.search_method = 'dace';
    options.search_option = optimset('Algorithm','active-set', 'Display','off');
    options.FitInParallel = true;
    options.logOn = true;

    tic
    [dmodelCML, ~] = dacefit_CML(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
    dmodelCML.mtime = toc;
    tic;
    [dmodelCML_CLP, YCLP] = predictor_CLP(trail.S, dmodelCML, false, length(trail.Y)+1);
    dmodelCML_CLP.RMSE = sqrt(mean((trail.Y-YCLP).^2));
    dmodelCML_CLP.ptime = toc;
    tic;
    [dmodelCML_BLUBP, YBLUBP] = predictor_BLUBP(trail.S, dmodelCML, length(trail.Y)+1);
    dmodelCML_BLUBP.RMSE = sqrt(mean((trail.Y-YBLUBP).^2));
    dmodelCML_BLUBP.ptime = toc;

    tic
    [dmodelCCL, ~] = dacefit_CCL(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
    dmodelCCL.mtime = toc;
    tic;
    [dmodelCCL_CLP, YCLP] = predictor_CLP(trail.S, dmodelCCL, false, length(trail.Y)+1);
    dmodelCCL_CLP.RMSE = sqrt(mean((trail.Y-YCLP).^2));
    dmodelCCL_CLP.ptime = toc;
    tic;
    [dmodelCCL_BLUBP, YBLUBP] = predictor_BLUBP(trail.S, dmodelCCL, length(trail.Y)+1);
    dmodelCCL_BLUBP.RMSE = sqrt(mean((trail.Y-YBLUBP).^2));
    dmodelCCL_BLUBP.ptime = toc;

    tic
    [dmodelOCL, ~] = dacefit_OCL(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
    dmodelOCL.mtime = toc;
    tic;
    [dmodelOCL_CLP, YCLP] = predictor_CLP(trail.S, dmodelOCL, false, length(trail.Y)+1);
    dmodelOCL_CLP.RMSE = sqrt(mean((trail.Y-YCLP).^2));
    dmodelOCL_CLP.ptime = toc;
    tic;
    [dmodelOCL_BLUBP, YBLUBP] = predictor_BLUBP(trail.S, dmodelOCL, length(trail.Y)+1);
    dmodelOCL_BLUBP.RMSE = sqrt(mean((trail.Y-YBLUBP).^2));
    dmodelOCL_BLUBP.ptime = toc;

    X = [];
    Y = [];
    for j = 1:length(design)
        X = [X;design(j).S];
        Y = [Y;design(j).Y];
    end
    
    %GPRR
    knots = 512;
    dmodelGPRR = struct;
    tic;
    YGPRR = dacefit_GPRR(X,Y,trail.S,1:knots, @corrGaussian, 1*ones(1,p),20);
    dmodelGPRR.RMSE = sqrt(mean((trail.Y-YGPRR.Yp).^2));
    dmodelGPRR.time = toc;
    dmodelGPRR.knots = knots;
    
    %SPGP
    dmodelSPGP = struct;
    tic;
    YSPGP = dacefit_SPGP(X,Y,trail.S,1:knots);
    dmodelSPGP.RMSE = sqrt(mean((trail.Y-YSPGP.Yp).^2));
    dmodelSPGP.time = toc;
    dmodelSPGP.knots = knots;

    result = struct();
    result.search_method = options.search_method;
    result.dmodelCML = dmodelCML;
    result.dmodelCML_CLP = dmodelCML_CLP;
    result.dmodelCML_BLUBP = dmodelCML_BLUBP;
    result.dmodelCCL = dmodelCCL;
    result.dmodelCCL_CLP = dmodelCCL_CLP;
    result.dmodelCCL_BLUBP = dmodelCCL_BLUBP;
    result.dmodelOCL = dmodelOCL;
    result.dmodelOCL_CLP = dmodelOCL_CLP;
    result.dmodelOCL_BLUBP = dmodelOCL_BLUBP;  
    result.dmodelSPGP = dmodelSPGP;
    result.dmodelGPRR = dmodelGPRR;
    result.design = design;
    result.trail = trail;
    results = [results result];
end
save('results_protein', 'results');