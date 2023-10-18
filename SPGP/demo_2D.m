clear
addpath('..\routines\')


k = 10;              % number of partitions
N = 20;             % number of design sites in each partition
p = 2;              % dimension of input space X
iter = 2;       % number of iterations
domain = 10;       % the domain of input space starting from 0
n = N*k;            % total design sites
m = 4;              % number of methods to be compared
q = 1;              % dimension of regression functions
Nt = 1600; 
dbstop if error

phi = zeros(iter, m*p);
beta = zeros(iter, m*q);
sigma2 = zeros(iter, m*1);
theta = zeros(iter, m*p);
RMSE = zeros(iter, m+2);

options.search_method = 'fmincon';
options.search_option = optimset('Algorithm','active-set', 'Display','off');
options.FitInParallel = false;
options.logOn = false;







result = struct('design',0,'trail',0,'dmodelML',0,'dmodelOCL',0,'dmodelCML',0,'dmodelCCL',0,'i',0);
results = repmat(result, iter,1);

for i = 1 : iter
    
    if mod(i,100) == 0
        disp(i);
    end
    
    [design, trail] = SGP_sampling(Nt, k, N, p, domain, 0, 1, [2*ones(p,1); 0.1]);
    theta0 = [(2+sign(randn))*ones(1,p) 0.1];
    lb = [0.5*ones(1,p) 0.00001];   ub = [3.5*ones(1,p) 1];
    
    X = [];     Y = [];
    for j = 1 : length(design)
        X = [X; design(j).S];
        Y = [Y; design(j).Y];
    end
    
    tic;
    [dmodelOCL, perfOCL] = dacefit_OCL(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
    tOCL = toc;
    tic;
    [dmodelCML, perfCML] = dacefit_CML(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
    tCML = toc;
    tic;
    [dmodelCCL, perfCCL] = dacefit_CCL(design, @regpoly0, @corr_gauss_s, theta0, lb, ub, options);
    tCCL = toc;
    tic;
    [dmodelIK, perfML] = dacefit(X,Y, @regpoly0, @corr_gauss_s, theta0, lb, ub, options.search_method, options.search_option);
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
        [~, YBLUP] = predictor_BLUP(trail.S, dmodel);
        tBLUP = toc;
        models(j).theta = dmodel.theta(:);
        models(j).beta = dmodel.beta(:);
        models(j).sigma2 = dmodel.sigma2(:);
        models(j).mtime = modelTime(j);
        BLUP = struct;
        BLUP.ptime = tBLUP;
        BLUP.RMSE = sqrt(mean((YBLUP-trail.Y).^2));
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
        [~, YCLP] = predictor_CLP(trail.S, dmodel);
        tCLP = toc;
        CLP = struct;
        CLP.ptime = tCLP;
        CLP.RMSE = sqrt(mean((YCLP-trail.Y).^2));
        models(j).CLP = CLP;
        
        tic;
        [~, YBLUBP] = predictor_BLUBP(trail.S, dmodel);
        tBLUBP = toc;
        BLUBP = struct;
        BLUBP.ptime = tBLUBP;
        BLUBP.RMSE = sqrt(mean((YBLUBP-trail.Y).^2));
        models(j).BLUBP = BLUBP;
    end
    
    % SPGP
    tic;
    YSPGP = spgp(X,Y,trail.S,40);
    dmodelSPGP.SPGP.time = toc;
    dmodelSPGP.SPGP.RMSE = sqrt(mean((YSPGP-trail.Y).^2));
    dmodelSPGP.SPGP
    
    result = struct('design',0,'trail',0,'dmodelML',0,'dmodelOCL',0,'dmodelCML',0,'dmodelCCL',0,'i',0);
    result.dmodelML = models(1);
    result.dmodelOCL = models(2);
    result.dmodelCML = models(3);
    result.dmodelCCL = models(4);
    result.design = design;
    result.trail = trail;
    result.i = i;
    results(i) = result;
end
save('Results_1D_SLHD_1',results')
% save(['Results_1D_SLHD (k=',num2str(k),',N=',num2str(N),')'],'results')

