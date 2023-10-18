clear
addpath('..\routines\')

p = 8; 
k = 20;
N = 20;
Nt = 1000;
Sdata  = slhd(N, k, p);
X = [];    Y = [];
level = struct('S', {[]}, 'Y', {[]});
design = repmat(level, k, 1);
trail = level;
trail.S = lhsdesign(Nt, p);
trail.Y = fun_BH(trail.S);
for j = 1 : k
    design(j).S = Sdata(:,:,j);
    design(j).Y = fun_BH(design(j).S);
    X = [X; design(j).S];
    Y = [Y; design(j).Y];
end

% find a best space-filling design that has minimum distance
m = 20*p;    n = length(Y);    nug=1e-8;
I0 = randperm(n, m);
minD = maxcd(X(I0,:));
for i = 1 : 100
    I0 = randperm(length(Y), m);
    maxD = maxcd(X(I0,:));
    if minD > maxD
        A = X(I0,:);
        gamma0 = Y(I0,:);
        minD = maxD;
    end
end


theta0 = ones(1,p);
GA=[ones(m,1) A]; GX=[ones(n,1) X];


minLS0=fun_LS(gamma0,A,X,Y,theta0);
dLS=minLS0;
tic
options=optimset('Display', 'none', 'LargeScale', 'off', 'Algorithm', 'sqp');
k=0;
while dLS>0.001 && k<10    %(k can be smaller otherwise the convergence is time-consuming)
    k=k+1; 
    theta0=fmincon(@(theta) fun_LS(gamma0,A,X,Y,theta), theta0, [], [], [], [], 0.0001*ones(1,p), 20*ones(1,p), [], options);
    RA = corrGaussian(theta0(:), A); RAX = corrGaussian(theta0(:), A, X);
    K=RA\GA; H=GA'*(RA\GA)+nug*eye(p+1);
    B=K*(H\GX')+(eye(m)-K*(H\GA'))*(RA\RAX); B=B';
    gamma0=(B'*B+nug*eye(m))\B'*Y;
    minLS=fun_LS(gamma0,A,X,Y,theta0); 
    dLS=minLS0-minLS; 
    minLS0=minLS;
end
gammaE=gamma0; thetaE=theta0; yE=B*gammaE;
toc

X_pre = trail.S;
Npre=length(X_pre);
GX_pre=[ones(Npre,1) X_pre];
R=corrGaussian(thetaE(:), A, X_pre);
Bp=K*(H\GX_pre')+(eye(m)-K*(H\GA'))*(RA\R); Bp=Bp';
y_pre=Bp*gammaE;
disp(sqrt(mean((y_pre-trail.Y).^2)))

