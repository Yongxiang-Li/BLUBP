function [ Yp ] = spgp( X, Y, Xp, m)
    meanY = mean(Y); Y = Y - meanY; % zero mean the data
    [N,dim] = size(X);

    % initialize pseudo-inputs to a random subset of training inputs
    [~,I] = sort(rand(N,1));
    I = I(1:m);
    xb_init = X(I,:);

    % initialize hyperparameters sensibly (see spgp_lik for how
    % the hyperparameters are encoded)
    hyp_init(1:dim,1) = -2*log((max(X)-min(X))'/2); % log 1/(lengthscales)^2
    hyp_init(dim+1,1) = log(var(Y,1)); % log size 
    hyp_init(dim+2,1) = log(var(Y,1)/4); % log noise

    % optimize hyperparameters and pseudo-inputs
    w_init = [reshape(xb_init,m*dim,1);hyp_init];
    [w,f] = minimize(w_init,'spgp_lik',-200,Y,X,m);
    % [w,f] = lbfgs(w_init,'spgp_lik',200,10,y0,x,M); % an alternative
    xb = reshape(w(1:m*dim,1),m,dim);
    hyp = w(m*dim+1:end,1);

    % PREDICTION
    [mu0,s2] = spgp_pred(Y, X, xb, Xp,hyp);
    Yp = mu0 + meanY; % add the mean back on
end

