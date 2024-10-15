function  [dmodel, perf] = dacefit_OCL_GPU(design, regr, theta0, lob, upb, options)
    if nargin < 3
        error('insufficient number of inputs');
    end
    if nargin == 3
        if  any(theta0 < 0)
            error('theta0 must be strictly positive');
        end
        lob = [];
        upb = [];
        options.search_method = 'none';
        options.FitInParallel = false;
        options.logOn = false;
    else
        lth = length(theta0);
        if  length(lob) ~= lth || length(upb) ~= lth
            error('theta0, lob and upb must have the same length');
        end
        if  any(lob < 0) || any(upb < lob)
            error('The bounds must satisfy  0 < lob <= upb')
        end
    end
    
    if isempty(design)
        error('input design data cannot be empty')
    end
    
    k = length(design); % number of partitions
    
    % Normalization and init
    for i = 1 : k
        design(i).F = feval(regr, design(i).S);
        design(i).n = size(design(i).F,1);
    end

    par = struct('regr',regr, 'UseParallel',options.FitInParallel, 'logOn', options.logOn);
    par.S = gpuArray(cat(3, design.S));    par.Y = gpuArray(cat(3, design.Y));
    par.F = gpuArray(cat(3, design.F));    par.n = cat(2, design.n);

    if strcmp(options.search_method, 'none')
        % evaluate the object function directly
        [f, fit] = objfunc(theta0, par);
        theta = theta0';
        perf = struct('perf',[theta; f; 1], 'nv',1);
    else
        if strcmp(options.search_method, 'dace')
            % pattern search method originally provided in DACE package
            [theta, f, fit, perf] = boxmin(theta0, lob, upb, par);
            if  isinf(f)
                error('Bad parameter region.  Try increasing  upb');
            end
        elseif strcmp(options.search_method, 'fmincon')
            % search algorithm in matlab 'fmincon'
            if ~isfield(options, 'search_option')
                options.search_option = optimset('Algorithm','active-set', 'Display','off');
            end
            [theta, f, fit, output] = mleparam(theta0, lob, upb, par, options.search_option);
            if  isinf(f)
                error('Bad parameter region.  Try increasing  upb');
            end
            perf = output;
        else
            error(['No searching method "' options.search_method '" available!']);
        end
    end
        
    dmodel = struct('regr',regr, 'theta',theta(:), 'options',options,...
          'beta',fit.beta, 'sigma2',fit.sigma2, 'design',design, 'likelihood',f);

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================

function  [obj, fit] = objfunc(theta, par)
    theta = abs(theta(:));
    if length(theta)>size(par.S(:,:,1),2)
        gamma2 = theta(end)^2;      phi = theta(1:end-1);
    else
        gamma2 = sqrt(eps);         phi = theta;
    end
    n = sum(par.n);    k = size(par.Y,3);
    par.S = par.S .* sqrt(phi');
    par.S2 = sum(par.S.^2, 2);
    
    FKF = 0;    FKY = 0;    YKY = 0;    detK = 0;
    C1 =  chol(exp(2*par.S(:,:,1)*par.S(:,:,1)'-(par.S2(:,:,1)+par.S2(:,:,1)'))+gamma2*eye(par.n(1)), 'lower');
    IF1 = C1 \ par.F(:,:,1);
    IY1 = C1 \ par.Y(:,:,1);
    FKF = FKF + (IF1'*IF1)/n;
    FKY = FKY + (IF1'*IY1)/n;
    YKY = YKY + (IY1'*IY1)/n;
    detK = detK + 2*sum(log(full(diag(C1))))/n;
    R21 = exp(2*par.S(:,:,2)*par.S(:,:,1)'-(par.S2(:,:,2)+par.S2(:,:,1)')) / C1';
    Y2 = par.Y(:,:,2) - R21*IY1;     F2 = par.F(:,:,2) - R21*IF1;
    K2 = exp(2*par.S(:,:,2)*par.S(:,:,2)'-(par.S2(:,:,2)+par.S2(:,:,2)')) ...
        + gamma2*eye(par.n(2)) - R21*R21';
    C2 = chol(K2+eps*eye(size(K2)))';
    IY2 = C2 \ Y2;                      IF2 = C2 \ F2;
    FKF = FKF + (IF2'*IF2)/n;  
    FKY = FKY + (IF2'*IY2)/n;
    YKY = YKY + (IY2'*IY2)/n;
    detK = detK + 2*sum(log(full(diag(C2))))/n;
    
    if par.UseParallel
        pool = gcp;
        poolsize = pool.NumWorkers;
        estTime = 0.1*(k:-1:3).^2 + 0.9*(k:-1:3);
        if poolsize>length(estTime), poolsize=length(estTime); end
        sumTime = zeros(poolsize, length(estTime));
        for p = 1 : poolsize
            sumTime(p,p) = estTime(p);
            for j = p :poolsize:length(estTime)
                if all(sumTime(:,j)==0) && sum(sumTime(p,:))+estTime(j)<sum(estTime)/poolsize
                    sumTime(p,j) = estTime(j);
                end
            end
        end
        for j = 1:length(estTime)
            if all(sumTime(:,j)==0)
                [~, index] = min(sum(sumTime,2));
                sumTime(index,j) = estTime(j);
            end
        end
        parfor p = 1 : poolsize % parfor is not well implemented so I specify the order of the loop
            t = now;
            loop = k:-1:3;
            if mod(p,2)==0
                [FKFp, FKYp, YKYp, detKp] = BCCL_GPU(par, gamma2, loop(sumTime(p,:)>0));
            else
                [FKFp, FKYp, YKYp, detKp] = BCCL_GPU(par, gamma2, fliplr(loop(sumTime(p,:)>0)));
            end
            FKF = FKF + FKFp;   
            FKY = FKY + FKYp;
            YKY = YKY + YKYp;
            detK = detK + detKp;
            if par.logOn,    disp(['Task ' num2str(p) ' costs ' num2str((now-t)*24*3600) ' seconds']); end
        end
    else
        t = now;
        [FKFp, FKYp, YKYp, detKp] = BCCL_GPU(par, gamma2);
        if par.logOn,    disp(['Computing one likelihood costs ' num2str((now-t)*24*3600) ' seconds']); end
        FKF = FKF + FKFp;  
        FKY = FKY + FKYp;
        YKY = YKY + YKYp;
        detK = detK + detKp;
    end
    beta = FKF \ FKY;
    sigma2 = (YKY - 2*beta'*FKY + beta'*FKF*beta);
    obj = gather(log(sigma2) + detK);
    if par.logOn
        disp(['theta: ' num2str(theta(:)',' %.4f') '--beta:' num2str(beta(:)',' %.4f') '--sigma2:' num2str(sigma2,' %.4f')  '--likelihood:' num2str(obj,' %.4f')])
    end

    if  nargout > 1
        fit = struct('sigma2',gather(sigma2), 'beta',gather(beta));
    end
    
function [FKF, FKY, YKY, detK] = BCCL_GPU(par, gamma2, loop)
    n = sum(par.n);    k = size(par.Y,3);    N = n/k;  
    rho = 1 + gamma2;    delta2 = eps;   
	S = par.S;    S2 = par.S2;
    invRd = pagefun(@inv, (exp(2*pagefun(@mtimes, S, pagefun(@transpose,S)) - ...
        (S2+rot90(S2))) + gamma2*eye(N,'gpuArray'))); % inverse of block diagnoal matrix
    
    FKF = gpuArray(0);    FKY = gpuArray(0);    YKY = gpuArray(0);    detK = gpuArray(0);
    if nargin<3, loop = k: -1 : 3; end
    for b = loop
        i = b - 1;
        Sr = S(:,:,1:i);    Sr2 = S2(:,:,1:i);
        R = exp(2*pagefun(@mtimes, S(:,:,b), pagefun(@transpose,Sr)) - (S2(:,:,b)+rot90(Sr2))); % Rir
        invRir = pagefun(@mtimes, R, invRd(:,:,1:i));
        Di = permute(sum(invRir.*R, 2), [3 2 1]) + delta2;
        Fi = permute(par.F(:,:,b) - pagefun(@mtimes, invRir, par.F(:,:,1:i)), [3 2 1]);
        Yi = permute(par.Y(:,:,b) - pagefun(@mtimes, invRir, par.Y(:,:,1:i)), [3 2 1]);
        Ki = zeros(par.n(b),i,i, 'gpuArray');
        for r = 2 : ceil((i+1)/2)
            r1 = r;    r2 = i-r+2;
            indexS = [1:r1-1 1:r2-1];
            Ss = S(:,:,indexS);    Ss2 = S2(:,:,indexS);
            indexR = [r1*ones(1,r1-1) r2*ones(1,r2-1)];
            Sr = S(:,:,indexR);    Sr2 = S2(:,:,indexR);
            R = exp(2*pagefun(@mtimes, Sr, pagefun(@transpose,Ss)) - (Sr2+rot90(Ss2))); % Rrs
            Krs = sum(pagefun(@mtimes, invRir(:,:,indexR), R) .* invRir(:,:,indexS),2);
            Ki(:,r1,1:r1-1) = Krs(:,:,1:r1-1);
            Ki(:,r2,1:r2-1) = Krs(:,:,r1:end);
        end
        Ki = permute(Ki, [3 2 1]);
        Ki = Ki + pagefun(@transpose, Ki) + pagefun(@times, repmat(Di,[1,i,1]), eye(i));

        Il = pagefun(@mldivide, Ki, Di); % inverse of lambda
        Ii = pagefun(@mldivide, Ki, ones(i,1,'gpuArray'));
        iRi = pagefun(@mtimes, ones(1,i,'gpuArray'), Ii);
        iRl = pagefun(@mtimes, ones(1,i,'gpuArray'), Il);
        w = pagefun(@mtimes, Ii, (ones(1,'gpuArray')-iRl)./iRi)+Il;
        IKI = ones(1,'gpuArray')*rho + (ones(1,'gpuArray')-iRl).^2./iRi ...
            -pagefun(@mtimes, rot90(Di), Il) + delta2*(2-pagefun(@mtimes, rot90(w), w));  
        detK = detK + sum(log(IKI),3)/n;
        IKI = n.*IKI;
        wFi = pagefun(@mtimes,rot90(w),Fi);
        wYi = pagefun(@mtimes,rot90(w),Yi);
        FKF = FKF + sum(pagefun(@mtimes,pagefun(@transpose,wFi),wFi)./IKI,3);
        FKY = FKY + sum(pagefun(@mtimes,pagefun(@transpose,wFi),wYi)./IKI,3);
        YKY = YKY + sum(pagefun(@mtimes,rot90(wYi),wYi)./IKI,3);
    end
    
% --------------------------------------------------------

function  [t, f, fit, perf] = boxmin(t0, lo, up, par)
%BOXMIN  Minimize with positive box constraints

% Initialize
[t, f, fit, itpar] = start(t0, lo, up, par);
if  ~isinf(f)
  % Iterate
  p = length(t);
  if  p <= 2,  kmax = 2; else,  kmax = min(p,4); end
  for  k = 1 : kmax
    th = t;
    [t, f, fit, itpar] = explore(t, f, fit, itpar, par);
    [t, f, fit, itpar] = move(th, t, f, fit, itpar, par);
  end
end
perf = struct('nv',itpar.nv, 'perf',itpar.perf(:,1:itpar.nv));

% --------------------------------------------------------

function  [t, f, fit, itpar] = start(t0, lo, up, par)
% Get starting point and iteration parameters

% Initialize
t = t0(:);  lo = lo(:);   up = up(:);   p = length(t);
D = 2 .^ ([1:p]'/(p+2));
ee = find(up == lo);  % Equality constraints
if  ~isempty(ee)
  D(ee) = ones(length(ee),1);   t(ee) = up(ee); 
end
ng = find(t < lo | up < t);  % Free starting values
if  ~isempty(ng)
  t(ng) = (lo(ng) .* up(ng).^7).^(1/8);  % Starting point
end
ne = find(D ~= 1);

% Check starting point and initialize performance info
[f  fit] = objfunc(t,par);   nv = 1;
itpar = struct('D',D, 'ne',ne, 'lo',lo, 'up',up, ...
  'perf',zeros(p+2,200*p), 'nv',1);
itpar.perf(:,1) = [t; f; 1];
if  isinf(f)    % Bad parameter region
  return
end

if  length(ng) > 1  % Try to improve starting guess
  d0 = 16;  d1 = 2;   q = length(ng);
  th = t;   fh = f;   jdom = ng(1);  
  for  k = 1 : q
    j = ng(k);    fk = fh;  tk = th;
    DD = ones(p,1);  DD(ng) = repmat(1/d1,q,1);  DD(j) = 1/d0;
    alpha = min(log(lo(ng) ./ th(ng)) ./ log(DD(ng))) / 5;
    v = DD .^ alpha;   tk = th;
    for  rept = 1 : 4
      tt = tk .* v; 
      [ff  fitt] = objfunc(tt,par);  nv = nv+1;
      itpar.perf(:,nv) = [tt; ff; 1];
      if  ff <= fk 
        tk = tt;  fk = ff;
        if  ff <= f
          t = tt;  f = ff;  fit = fitt; jdom = j;
        end
      else
        itpar.perf(end,nv) = -1;   break
      end
    end
  end % improve
  
  % Update Delta  
  if  jdom > 1
    D([1 jdom]) = D([jdom 1]); 
    itpar.D = D;
  end
end % free variables

itpar.nv = nv;

% --------------------------------------------------------

function  [t, f, fit, itpar] = explore(t, f, fit, itpar, par)
% Explore step

nv = itpar.nv;   ne = itpar.ne;
for  k = 1 : length(ne)
  j = ne(k);   tt = t;   DD = itpar.D(j);
  if  t(j) == itpar.up(j)
    atbd = 1;   tt(j) = t(j) / sqrt(DD);
  elseif  t(j) == itpar.lo(j)
    atbd = 1;  tt(j) = t(j) * sqrt(DD);
  else
    atbd = 0;  tt(j) = min(itpar.up(j), t(j)*DD);
  end
  [ff  fitt] = objfunc(tt,par);  nv = nv+1;
  itpar.perf(:,nv) = [tt; ff; 2];
  if  ff < f
    t = tt;  f = ff;  fit = fitt;
  else
    itpar.perf(end,nv) = -2;
    if  ~atbd  % try decrease
      tt(j) = max(itpar.lo(j), t(j)/DD);
      [ff  fitt] = objfunc(tt,par);  nv = nv+1;
      itpar.perf(:,nv) = [tt; ff; 2];
      if  ff < f
        t = tt;  f = ff;  fit = fitt;
      else
        itpar.perf(end,nv) = -2;
      end
    end
  end
end % k

itpar.nv = nv;

% --------------------------------------------------------

function  [t, f, fit, itpar] = move(th, t, f, fit, itpar, par)
% Pattern move

nv = itpar.nv;   ne = itpar.ne;   p = length(t);
v = t ./ th;
if  all(v == 1)
  itpar.D = itpar.D([2:p 1]).^.2;
  return
end

% Proper move
rept = 1;
while  rept
  tt = min(itpar.up, max(itpar.lo, t .* v));  
  [ff  fitt] = objfunc(tt,par);  nv = nv+1;
  itpar.perf(:,nv) = [tt; ff; 3];
  if  ff < f
    t = tt;  f = ff;  fit = fitt;
    v = v .^ 2;
  else
    itpar.perf(end,nv) = -3;
    rept = 0;
  end
  if  any(tt == itpar.lo | tt == itpar.up), rept = 0; end
end

itpar.nv = nv;
itpar.D = itpar.D([2:p 1]).^.25;

function [theta, fmin, fit, output] = mleparam(theta0, lo, up, par, options)
% Matlab Optimization Toolbox is needed to use "fmincon"
    ofun = @(theta)objfunc(theta,par);
    options.UseParallel = []; % do not use parallel here
    [theta, fmin, exitflag, output] = fmincon(ofun,theta0,[],[],[],[],lo,up,[],options);
    [f, fit] = objfunc(theta, par);
