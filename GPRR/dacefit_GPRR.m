function [ dmodel ] = dacefit_GPRR( X, Y, Xp, I, corr, theta0, iterations )
% --- input -----------
% X, Y: training data
% I: index of knots
% theta0: initial values of correlation function
% Xp: data for prediction

% ----- output ------------
% dmodel: 

    [n, d]=size(X);    nug=1e-8;
    m = length(I);
    A = X(I,:);    gamma0 = Y(I,:);
    
	dmodel.X = X;
    dmodel.Y = Y;
    dmodel.Xp = Xp;
    dmodel.Yp = nan;
    dmodel.I = I;
    dmodel.corr = corr;
    dmodel.theta0 = theta0;
	
    GA=[ones(m,1) A]; GX=[ones(n,1) X];
    minLS0=costfun(X, Y, A, gamma0, corr, theta0);
    dLS=minLS0;
    options=optimset('Algorithm','active-set', 'Display','off');
    k=0;
    while dLS>0.001 && k<iterations    %(k can be smaller otherwise the convergence is time-consuming)
        k=k+1; 
        theta0=fmincon(@(theta) costfun(X, Y, A, gamma0, corr, theta), ...
            theta0, [], [], [], [], 0.0001*ones(size(theta0)), 100*ones(size(theta0)), [], options);
        RA = corr(theta0(:), A); RAX = corr(theta0(:), A, X);
        K=RA\GA; H=GA'*K+nug*eye(d+1);
        B=K*(H\GX')+(eye(m)-K*(H\GA'))*(RA\RAX); B=B';
        gamma0=(B'*B+nug*eye(m))\B'*Y;
        minLS=costfun(X, Y, A, gamma0, corr, theta0);
        dLS=minLS0-minLS; 
        minLS0=minLS;
    end
    gammaE=gamma0; thetaE=theta0; yE=B*gammaE;

    Npre=length(Xp);
    GXp=[ones(Npre,1) Xp];
    R=corr(thetaE(:), A, Xp);
    Bp=K*(H\GXp')+(eye(m)-K*(H\GA'))*(RA\R); Bp=Bp';
    Yp=Bp*gammaE;
    
    dmodel.Yp = Yp;
    dmodel.theta = thetaE;
    dmodel.iterations = iterations;
end

function z=costfun(X, Y, A, gamma, corr, theta)  
% input: the GP reconstruction is based on (A, gamma) with kernel parameter theta
% output: z is LS of (X,y)

    [n, d]=size(X); m=length(A); nug=1e-8;
    GA = [ones(m,1) A]; GX=[ones(n,1) X];
    RA = corr(theta(:), A); 
    RAX = corr(theta(:), A, X);
    gamma = RA\gamma;
    K = RA\GA; H=GA'*K+nug*eye(d+1);
    Yhat = ((GX-(RAX'*K))/H)*(GA'*gamma) + RAX'*gamma;
    z = mean((Y-Yhat).^2);
end

