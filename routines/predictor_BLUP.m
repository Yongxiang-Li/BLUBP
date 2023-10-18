function  [dmodel, Y] = predictor_BLUP(X, dmodel)
    or1 = NaN;   or2 = NaN;  dmse = NaN;  % Default return values
    if  isnan(dmodel.beta)
    y = NaN;   
    error('DMODEL has not been found')
    end

    [m, n] = size(dmodel.S);  % number of design sites and number of dimensions
    [mx, nx] = size(X);            % number of trial sites and their dimension
    if  nx ~= n
    error('Dimension of trial sites should be %d',n)
    end

    % Get regression function and correlation
    f = feval(dmodel.regr, X);
    F = feval(dmodel.regr, dmodel.S);
    r = feval(dmodel.corr, dmodel.theta(:), dmodel.S, X);
    R = feval(dmodel.corr, dmodel.theta(:), dmodel.S);
    C = chol(R)';
    Ft = C \ F;
    Yt = C \ dmodel.Y;
    rho = Yt - Ft*dmodel.beta;
    gamma = rho' / C;

    % predictor 
    Y = f * dmodel.beta + (gamma * r).';
    
    % MSE
    rt = C \ r;
    V = feval(dmodel.corr, dmodel.theta(:), X(1)) - dot(rt, rt)';
    
    dmodel.X0 = X;
    dmodel.Y0 = Y;
    dmodel.V0 = sqrt(dmodel.sigma2*V);
    
  