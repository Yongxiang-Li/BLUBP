function  [dmodel, Y] = predictor(X, dmodel)
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
    r = feval(dmodel.corr, dmodel.theta(:), dmodel.S, X);
    q = size(f,2);

    % predictor 
    Y = f * dmodel.beta + (dmodel.gamma * r).';
    
    % MSE
    rt = dmodel.C \ r;
    u = dmodel.G \ (dmodel.Ft.' * rt - f.');
    V = repmat((1 + colsum(u.^2) - colsum(rt.^2))',1,q);
    
    dmodel.X0 = X;
    dmodel.Y0 = Y;
    dmodel.V0 = sqrt(dmodel.sigma2*V);
    
  
% >>>>>>>>>>>>>>>>   Auxiliary function  ====================

function  s = colsum(x)
% Columnwise sum of elements in  x
if  size(x,1) == 1,  s = x; 
else,                s = sum(x);  end