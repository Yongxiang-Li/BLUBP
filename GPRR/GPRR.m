function [gammaE, thetaE, yE, Yp]=GPRR(X, Y, Xp, theta0, A, gamma0, iterations)

% Gaussian process reconstruction regression estimation 
% --- input -----------
% A: knot set, taken as a subset of X
% X, Y: training data
% theta0, gamma0 are initial values
% X_pre: data for prediction
% ---------------------

% ----- output ------------
% gammaE: coefficents of kernel reconstruction estimator of mu
% thetaE: estimator of correlation parameters in Gaussian kernel
% yE: estimators at X
% y_pre: prediction at X_pre
% -------------------------
[n, d]=size(X);    nug=1e-8;
if nargin <= 4
    m = 10*d;    iterations = 10; 
    I0 = randperm(n, m);
    minD = maxcd(X(I0,:));
    for i = 1 : 1000
        I0 = randperm(length(Y), m);
        maxD = maxcd(X(I0,:));
        if minD > maxD
            A = X(I0,:);
            gamma0 = Y(I0,:);
            minD = maxD;
        end
    end
else
    m=length(gamma0); 
end


GA=[ones(m,1) A]; GX=[ones(n,1) X];
minLS0=fun_LS(gamma0,A,X,Y,theta0);
dLS=minLS0;
options=optimset('Algorithm','active-set', 'Display','off');
k=0;
while dLS>0.001 && k<iterations    %(k can be smaller otherwise the convergence is time-consuming)
    k=k+1; 
    theta0=fmincon(@(theta) fun_LS(gamma0,A,X,Y,theta), theta0, [], [], [], [], 0.001*ones(1,d), 20*ones(1,d), [], options);
    RA = corrGaussian(theta0(:), A); RAX = corrGaussian(theta0(:), A, X);
    K=RA\GA; H=GA'*(K)+nug*eye(d+1);
    B=K*(H\GX')+(eye(m)-K*(H\GA'))*(RA\RAX); B=B';
    gamma0=(B'*B+nug*eye(m))\B'*Y;
    minLS=fun_LS(gamma0,A,X,Y,theta0); 
    dLS=minLS0-minLS; 
    minLS0=minLS;
end
gammaE=gamma0; thetaE=theta0; yE=B*gammaE;

Npre=length(Xp);
GX_pre=[ones(Npre,1) Xp];
R=corrGaussian(thetaE(:), A, Xp);
Bp=K*(H\GX_pre')+(eye(m)-K*(H\GA'))*(RA\R); Bp=Bp';
Yp=Bp*gammaE;