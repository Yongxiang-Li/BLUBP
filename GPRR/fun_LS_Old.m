function z=fun_LS_Old(gamma,A,X,y,theta)  
% input: the GP reconstruction is based on (A, gamma) with kernel parameter theta
% output: z is LS of (X,y)

[n d]=size(X); m=length(A); nug=1e-8;

GA=[ones(m,1) A]; GX=[ones(n,1) X];
RA=corrGaussianmat(A,theta)+nug*eye(m); RAX=corrGaussianmat2(A,X,theta);
K=RA\GA; H=GA'*(RA\GA)+nug*eye(d+1);
B=K*(H\GX')+(eye(m)-K*(H\GA'))*(RA\RAX); B=B';
z=mean((y-B*gamma).^2);

function z=fun_LS_bak(gamma,A,X,y,theta)  
% input: the GP reconstruction is based on (A, gamma) with kernel parameter theta
% output: z is LS of (X,y)

    [n, d]=size(X); m=length(A); nug=1e-8;

    GA=[ones(m,1) A]; GX=[ones(n,1) X];
    RA=corrGaussianmat(A,theta)+nug*eye(m); RAX=corrGaussianmat2(A,X,theta);
    C = chol(RA)';
    GAt = C \ GA;
    RAXt = C \ RAX;
    H = GAt'*GAt+nug*eye(d+1);
    Gt = C \ gamma;
    Yhat = ((GX-(RAXt'*GAt))/H)*GAt'*Gt + RAXt'*Gt;
    z = mean((y-Yhat).^2);
    gamma = RA\gamma;
    [RAXt'*Gt RAX'*gamma];
    K = RA\GA; H=GA'*(RA\GA)+nug*eye(d+1);
    [((GX-(RAXt'*GAt))/H)*GAt'*Gt ((GX-(RAX'*K))/H)*(GA'*gamma)];
    z = mean((y-Yhat).^2);

function z=fun_LS(gamma,A,X,y,theta)  
% input: the GP reconstruction is based on (A, gamma) with kernel parameter theta
% output: z is LS of (X,y)

    [n d]=size(X); m=length(A); nug=1e-8;

    GA=[ones(m,1) A]; GX=[ones(n,1) X];
    RA=corrGaussianmat(A,theta)+nug*eye(m); RAX=corrGaussianmat2(A,X,theta);
    C = chol(RA)';
    GAt = C \ GA;
    RAXt = C \ RAX;
    H = GAt'*GAt+nug*eye(d+1);
    Gt = C \ gamma;
%     Yhat = (GX / H)*GAt'*Gt + RAXt'*Gt-((RAXt'*GAt)/H)*(GAt'*Gt);
%     Yhat = ((GX / H)-((RAXt'*GAt)/H))*GAt'*Gt + RAXt'*Gt;
    Yhat = ((GX-(RAXt'*GAt))/H)*GAt'*Gt + RAXt'*Gt;
    mean((y-Yhat).^2);
    
    K=RA\GA; H=GA'*K+nug*eye(d+1);
    B=K*(H\GX')+(eye(m)-K*(H\GA'))*(RA\RAX);
    z=mean((y-B'*gamma).^2);
%     [(gamma'*K*(H\GX'))' (GX / H)*GAt'*Gt];
%     [(gamma'*(eye(m))*(RA\RAX))'  RAXt'*Gt ];
%     [(gamma'*(K*(H\GA'))*(RA\RAX))'  ((RAXt'*GAt)/H)*(GAt'*Gt) ];
%     [(gamma'*(eye(m)-K*(H\GA'))*(RA\RAX))'  RAXt'*Gt-((RAXt'*GAt)/H)*(GAt'*Gt) ];
    [mean((y-B'*gamma).^2) mean((y-Yhat).^2)];