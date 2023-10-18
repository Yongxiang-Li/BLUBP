function  [dmodel, perf] = dacefit_OCL(design, regr, corr, theta0, lob, upb, options)
    if nargin < 4
        error('insufficient number of inputs');
    end
    if nargin == 4
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
        mS = zeros(1, size(design(i).S,2));
        sS = ones(1, size(design(i).S,2));
        design(i).Ssc = [mS; sS];
        mY = 0;   sY = 1;
        design(i).Ysc = [mY; sY];
        design(i).F = feval(regr, design(i).S);
        design(i).n = size(design(i).F,1);
        design(i).Ft = design(i).F;
        design(i).Yt = design(i).Y;
    end

    par = struct('corr',corr, 'regr',regr, 'design',design, ...
        'UseParallel',options.FitInParallel, 'logOn',false);
    if isfield(options,'logOn')
        par.logOn = options.logOn;
    end

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
        
    dmodel = struct('regr',regr, 'corr',corr, 'theta',theta(:)', 'theta0',theta0(:)', ...
          'beta',fit.beta, 'sigma2',fit.sigma2, 'design',fit.design, ...
          'likelihood',f);

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================

function  [obj, fit] = objfunc(theta, par)
	design = par.design;
    k = length(design);
    theta = abs(theta(:));
    corr = par.corr;
    useParallel = par.UseParallel;
    logOn = par.logOn;
    
    n = sum([design.n]);
    FKF = 0;    FKY = 0;    YKY = 0;    detK = 0;
    
    C1 = chol(feval(corr,theta,design(1).S))';
    IF1 = C1 \ design(1).F;
    IY1 = C1 \ design(1).Y;
    FKF = FKF + (IF1'*IF1)/n;
    FKY = FKY + (IF1'*IY1)/n;
    YKY = YKY + (IY1'*IY1)/n;
    detK = detK - 2*sum(log(full(diag(C1))))/n;
    R21 = feval(corr,theta(:),design(2).S,design(1).S) / C1';
    Y2 = design(2).Y - R21*IY1;     F2 = design(2).F - R21*IF1;
    K2 = feval(corr,theta,design(2).S) - R21*R21';
    C2 = chol(K2+eps*eye(size(K2)))';
    IY2 = C2 \ Y2;                      IF2 = C2 \ F2;
    FKF = FKF + (IF2'*IF2)/n;  
    FKY = FKY + (IF2'*IY2)/n;
    YKY = YKY + (IY2'*IY2)/n;
    detK = detK - 2*sum(log(full(diag(C2))))/n;
    
    if useParallel
        pool = gcp;
        poolsize = pool.NumWorkers;
        estTime = (n/k)^3*k + (n/k)^3*(k:-1:3).^2 + (k:-1:3).^3*(n/k);
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
                [FKFp, FKYp, YKYp, detKp] = best_composite_likelihood_parellel(design, corr, theta, loop(sumTime(p,:)>0));
            else
                [FKFp, FKYp, YKYp, detKp] = best_composite_likelihood_parellel(design, corr, theta, fliplr(loop(sumTime(p,:)>0)));
            end
            FKF = FKF + FKFp;  
            FKY = FKY + FKYp;
            YKY = YKY + YKYp;
            detK = detK + detKp;
            if logOn,    disp(['Task ' num2str(p) ' costs ' num2str((now-t)*24*3600) ' seconds']); end
        end
    else
        [FKFp, FKYp, YKYp, detKp] = best_composite_likelihood(design, corr, theta);
        FKF = FKF + FKFp;  
        FKY = FKY + FKYp;
        YKY = YKY + YKYp;
        detK = detK + detKp;
    end
    beta = FKF \ FKY;
    sigma2 = (YKY - 2*beta'*FKY + beta'*FKF*beta);
    obj = log(sigma2) - detK;
    if par.logOn
        disp(['theta: ' num2str(theta(:)',' %.4f') '--beta:' num2str(beta(:)',' %.4f') '--sigma2:' num2str(sigma2,' %.4f')  '--likelihood:' num2str(obj,' %.4f')])
    end

    if  nargout > 1
        fit = struct('sigma2',sigma2, 'beta',beta,'design',design);
    end

    
function [FKFp, FKYp, YKYp, detKp] = best_composite_likelihood(design, corr, theta)
    n = sum([design.n]);
    delta2 = eps;    rho = feval(corr,theta,design(1).S(1));
    FKFp = 0;    FKYp = 0;    YKYp = 0;    detKp = 0;
    for i = 1 : length(design)
        design(i).C = chol(feval(corr,theta,design(i).S))';
        design(i).Ft = design(i).C\design(i).F;
        design(i).Yt = design(i).C\design(i).Y;
    end
    T = cell(length(design), length(design));
    T{1,2} = design(1).C \ (feval(corr,theta,design(1).S,design(2).S) / design(2).C');
    for i = 3 : length(design) 
        ki = i - 1;
        Ri = zeros(ki,ki,design(i).n); 
        Fi = zeros(ki,size(design(i).F,2),design(i).n);
        Yi = zeros(design(i).n, ki);
        for r = 1 : ki
            IRri = design(r).C \ feval(corr,theta,design(r).S,design(i).S);
            Ri(r,r,:) = dot(IRri, IRri) + delta2; % Ri(r,r,:) = dot(IRri, IRri) + delta2;
            Fi(r,:,:) = design(i).F' - design(r).Ft'*IRri;
            Yi(:,r) = design(i).Y - (design(r).Yt'*IRri)';
            design(r).IRri = IRri;
            T{r,i} = IRri / design(i).C';
        end
        for r = 1 : ki-1
            for s = r+1 : ki
                Ri(r,s,:) = dot((design(r).IRri), T{r,s}*design(s).IRri); % Ri(r,s,:) = dot((design(r).IRri), T{r,s}*design(s).IRri);
            end
        end
        for j = 1 : design(i).n
            Fij = Fi(:,:,j);
            Rij = Ri(:,:,j);
            Yij = Yi(j,:)';
            Cij = chol(Rij)';

            Iu = Cij \ diag(Rij);                   Ii = Cij \ ones(ki,1);  
            Iw = (1-Ii'*Iu)/(Ii'*Ii)*Ii + Iu;       w = Cij' \ Iw; 
            IKI = 1 / (rho + (1-Ii'*Iu)^2/(Ii'*Ii)-Iu'*Iu + delta2*(2-w'*w));
            FKFp = FKFp + IKI*(Fij'*w*w'*Fij)/n;      
            FKYp = FKYp + IKI*(Fij'*w*w'*Yij)/n;
            YKYp = YKYp + IKI*(Yij'*w*w'*Yij)/n;      
            detKp = detKp + log(IKI)/n;
        end
    end

function [FKFp, FKYp, YKYp, detKp] = best_composite_likelihood_parellel(design, corr, theta, loop)
    n = sum([design.n]);
    delta2 = eps;    rho = feval(corr,theta,design(1).S(1));
    FKFp = 0;    FKYp = 0;    YKYp = 0;    detKp = 0;
    for i = 1 : length(design)
        design(i).C = chol(feval(corr,theta,design(i).S))';
        design(i).Ft = design(i).C\design(i).F;
        design(i).Yt = design(i).C\design(i).Y;
    end
    for i = loop
        ki = i - 1;
        Ri = zeros(ki,ki,design(i).n); 
        Fi = zeros(ki,size(design(i).F,2),design(i).n);
        Yi = zeros(design(i).n, ki);
        for r = 1 : ki
            IRri = design(r).C \ feval(corr,theta,design(r).S,design(i).S);
            Ri(r,r,:) = dot(IRri, IRri) + delta2; % Ri(r,r,:) = dot(IRri, IRri) + delta2;
            Fi(r,:,:) = design(i).F' - design(r).Ft'*IRri;
            Yi(:,r) = design(i).Y - (design(r).Yt'*IRri)';
            design(r).IRri = design(r).C' \ IRri;
        end
        for r = 1 : ki-1
            for s = r+1 : ki
                Rrs = feval(corr,theta,design(r).S,design(s).S);
                Ri(r,s,:) = dot(design(r).IRri, Rrs*design(s).IRri); % Ri(r,s,:) = dot(design(r).IRri, Rrs*design(s).IRri);
            end
        end
        for j = 1 : design(i).n
            Fij = Fi(:,:,j);
            Rij = Ri(:,:,j);
            Yij = Yi(j,:)';
            Cij = chol(Rij)';

            Iu = Cij \ diag(Rij);                   Ii = Cij \ ones(ki,1);  
            Iw = (1-Ii'*Iu)/(Ii'*Ii)*Ii + Iu;       w = Cij' \ Iw;     
            IKI = 1 / (rho + (1-Ii'*Iu)^2/(Ii'*Ii)-Iu'*Iu + delta2*(2-w'*w));
            FKFp = FKFp + IKI*(Fij'*w*w'*Fij)/n;      
            FKYp = FKYp + IKI*(Fij'*w*w'*Yij)/n;
            YKYp = YKYp + IKI*(Yij'*w*w'*Yij)/n;      
            detKp = detKp + log(IKI)/n;
        end 
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
