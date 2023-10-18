function z=fun_LS(gamma,A,X,y,theta)  
% input: the GP reconstruction is based on (A, gamma) with kernel parameter theta
% output: z is LS of (X,y)

    [n, d]=size(X); m=length(A); nug=1e-8;

%     GA=[ones(m,1) A]; GX=[ones(n,1) X];
%     RA=corrGaussianmat(A,theta)+nug*eye(m); RAX=corrGaussianmat2(A,X,theta);
%     gamma = RA\gamma;
%     K = RA\GA; H=GA'*(RA\GA)+nug*eye(d+1);
%     Yhat = ((GX-(RAX'*K))/H)*(GA'*gamma) + RAX'*gamma;
%     z = mean((y-Yhat).^2)
    
    GA = [ones(m,1) A]; GX=[ones(n,1) X];
    RA = corrGaussian(theta(:), A); 
    RAX = corrGaussian(theta(:), A, X);
    gamma = RA\gamma;
    K = RA\GA; H=GA'*K+nug*eye(d+1);
    Yhat = ((GX-(RAX'*K))/H)*(GA'*gamma) + RAX'*gamma;
    z = mean((y-Yhat).^2);
    