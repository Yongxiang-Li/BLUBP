function [ dmodel ] = dacefit_SPGP( X, Y, Xp, I )
% --- input -----------
% X, Y: training data
% I: index of knots
% theta0: initial values of correlation function
% Xp: data for prediction

% ----- output ------------
% dmodel: 
    m = length(I);
    [N,dim] = size(X);
    xb_init = X(I,:);
    % initialize hyperparameters sensibly (see spgp_lik for how the hyperparameters are encoded)
    if nargin==5
        hyp_init = -2*log(theta0(:));
    else
        hyp_init(1:dim,1) = -2*log((max(X)-min(X))'/2); % log 1/(lengthscales)^2
        theta0 = exp(hyp_init);
    end
    hyp_init(dim+1,1) = log(var(Y,1)); % log size 
    hyp_init(dim+2,1) = log(var(Y,1)/4); % log noise

    % optimize hyperparameters and pseudo-inputs
    w_init = [reshape(xb_init,m*dim,1);hyp_init];
    [w,f] = minimize(w_init,'spgp_lik',-200,Y,X,m);
    % [w,f] = lbfgs(w_init,'spgp_lik',200,10,y0,x,M); % an alternative
    xb = reshape(w(1:m*dim,1),m,dim);
    hyp = w(m*dim+1:end,1);

    % PREDICTION
    [Yp, s2] = spgp_pred(Y, X, xb, Xp, hyp);

  	dmodel.X = X;
    dmodel.Y = Y;
    dmodel.Xp = Xp;
    dmodel.Yp = Yp;
    dmodel.Vp = s2;
    dmodel.I = I;
    dmodel.theta0 = exp(hyp(1:end-2));
    dmodel.sigma2 = exp(hyp(end-1));
    dmodel.sigmaN2 = exp(hyp(end));

