function  [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb, varargin)
   
    nInput = 5;
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
        if isempty(varargin)
            error('A search algorithm must be specified for the optimization');
        else
            lth = length(theta0);
            if  length(lob) ~= lth || length(upb) ~= lth
                error('theta0, lob and upb must have the same length');
            end
            if  any(lob < 0) || any(upb < lob)
                error('The bounds must satisfy  0 < lob <= upb')
            end 
            if strcmpi(varargin{1}, 'dace')
                SA = 0; % search algorithm is 'dace'
            elseif strcmpi(varargin{1}, 'fmincon')
                SA = 1; % search algorithm is 'fmincon'
                if length(varargin) == 2
                   options = varargin{2}; % options are specified and will be passed to 'fmincon'
                end
            end
        end
    end
    
    par = struct('corr',corr, 'regr',regr, 'S',S, 'Y',Y, 'F',feval(regr, S));

    if nargin == nInput
        % evaluate the object function directly
        [f, fit] = objfunc(theta0, par);
        theta = theta0';
        perf = struct('perf',[theta; f; 1], 'nv',1);
    else
        if SA == 0
            % pattern search method originally provided in DACE package
            [theta, f, fit, perf] = boxmin(theta0, lob, upb, par);
            if  isinf(f)
                error('Bad parameter region.  Try increasing  upb');
            end
        else
             % search algorithm in matlab 'fmincon'
            if length(varargin) == 1
                optionsdflt = optimset('Algorithm','interior-point','Display','off');
                [theta, f, fit, output] = mleparam(theta0, lob, upb, par, optionsdflt);
            elseif length(varargin) == 2
                [theta, f, fit, output] = mleparam(theta0, lob, upb, par, options);
            end
            if  isinf(f)
                error('Bad parameter region.  Try increasing  upb');
            end
            perf = output;
        end
    end
        
    dmodel = struct('regr',regr, 'corr',corr, 'theta',theta(:)', ...
          'beta',fit.beta, 'sigma2',fit.sigma2, 'S',S, 'Y',Y, 'Ft',fit.Ft, ...
          'G',fit.G','C',fit.C, 'gamma',fit.gamma, 'likelihood',f);

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================

function  [obj, fit] = objfunc(theta, par)
    % Initialize
    obj = inf; 
    fit = struct('sigma2',NaN, 'beta',NaN, 'gamma',NaN, ...
        'C',NaN, 'Ft',NaN, 'G',NaN);
    m = size(par.F,1);
    % Set up  R
    R = feval(par.corr, theta(:), par.S);
    C = chol(R)';

    % Get least squares solution
    Ft = C \ par.F;
    [Q G] = qr(Ft,0);
    if  rcond(G) < 1e-10
      % Check   F  
      if  cond(par.F) > 1e15 
        T = sprintf('F is too ill conditioned\nPoor combination of regression model and design sites');
        error(T)
      else  % Matrix  Ft  is too ill conditioned
        return 
      end 
    end
    Yt = C \ par.Y;   beta = G \ (Q'*Yt);
    % beta = zeros(size(beta));
    rho = Yt - Ft*beta;  sigma2 = sum(rho.^2)/m;
    % detR = prod( full(diag(C)) .^ (2/m) );
    % obj = sum(sigma2) * detR;
    obj = m*log(sum(sigma2)) + 2*sum(log(full(diag(C)))); 

    if  nargout > 1
      fit = struct('sigma2',sigma2, 'beta',beta, 'gamma',rho' / C, ...
        'C',C, 'Ft',Ft, 'G',G','R',full(R + R'-eye(length(R))));
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
