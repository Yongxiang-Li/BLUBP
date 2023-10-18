function  [dmodel, perf] = dacefit_CCL(design, regr, corr, theta0, lob, upb, options)
    nInput = 4;
    if nargin < nInput
        error('insufficient number of inputs');
    end
    if nargin == nInput
        if  any(theta0 < 0)
            error('theta0 must be strictly positive');
        end
        lob = [];
        upb = [];
    else
        lth = length(theta0);
        if  length(lob) ~= lth || length(upb) ~= lth
            error('theta0, lob and upb must have the same length');
        end
        if  any(lob < 0) || any(upb < lob)
            error('The bounds must satisfy  0 < lob <= upb')
        end 
        if isempty(options)
            options.search_method = 'none';
            options.FitInParallel = false;
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
    end

    par = struct('corr',corr, 'regr',regr, 'design',design, 'UseParallel',options.FitInParallel);
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

    k = length(par.design);
    design = par.design;
    theta = abs(theta(:));
    corr = par.corr;
    
    n = sum([par.design.n]);
    FKF = 0;    FKY = 0;    YKY = 0;    detK = 0; 
    if par.UseParallel
        parfor r = 1 : k
            Cr = chol(feval(corr,theta,design(r).S))';
            IFr = Cr \ design(r).F;
            IYr = Cr \ design(r).Y; 
            FKFp = 0;   FKYp = 0;   YKYp = 0;   detKp = 0;
            for s = 1 : k
                if s==r,    continue;       end
                Rsr = feval(corr,theta,design(s).S,design(r).S) / Cr';
                Ysr = design(s).Y - Rsr*IYr;     Fsr = design(s).F - Rsr*IFr;
                Ksr = feval(corr,theta,design(s).S) - Rsr*Rsr';
                Csr = chol(Ksr+eps*eye(size(Ksr)))';
                IYsr = Csr \ Ysr;                      IFsr = Csr \ Fsr;
                FKFp = FKFp + (IFsr'*IFsr)/(n*(k-1));  
                FKYp = FKYp + (IFsr'*IYsr)/(n*(k-1));
                YKYp = YKYp + (IYsr'*IYsr)/(n*(k-1));
                detKp = detKp + 2*sum(log(full(diag(Csr))))/(n*(k-1));
            end
            FKF = FKF + FKFp;
            FKY = FKY + FKYp;
            YKY = YKY + YKYp;
            detK = detK + detKp;
        end
    else
        for r = 1 : k
            Cr = chol(feval(corr,theta,design(r).S))';
            IFr = Cr \ design(r).F;
            IYr = Cr \ design(r).Y; 
            for s = 1 : k
                if s==r,    continue;       end
                Rsr = feval(corr,theta,design(s).S,design(r).S) / Cr';
                Ysr = design(s).Y - Rsr*IYr;     Fsr = design(s).F - Rsr*IFr;
                Ksr = feval(corr,theta,design(s).S) - Rsr*Rsr';
                Csr = chol(Ksr+eps*eye(size(Ksr)))';
                IYsr = Csr \ Ysr;                      IFsr = Csr \ Fsr;
                FKF = FKF + (IFsr'*IFsr)/(n*(k-1));  
                FKY = FKY + (IFsr'*IYsr)/(n*(k-1));
                YKY = YKY + (IYsr'*IYsr)/(n*(k-1));
                detK = detK + 2*sum(log(full(diag(Csr))))/(n*(k-1));
            end
        end
    end
    beta = FKF \ FKY;
    sigma2 = (YKY - 2*beta'*FKY + beta'*FKF*beta);
    obj = log(sigma2) + detK;
    if par.logOn
        disp(['theta: ' num2str(theta(:)',' %.4f') '--beta:' num2str(beta(:)',' %.4f') '--sigma2:' num2str(sigma2,' %.4f')  '--likelihood:' num2str(obj,' %.4f')])
    end

    if  nargout > 1
        fit = struct('sigma2',sigma2, 'beta',beta, 'design',par.design);
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
    [theta, fmin, exitflag, output] = fmincon(ofun,theta0,[],[],[],[],lo,up,[],options);
    [f, fit] = objfunc(theta, par);
