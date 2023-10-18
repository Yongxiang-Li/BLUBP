function  [dmodel, Y, V] = predictor_CLP(X, dmodel, calVar, dif)
    if nargin==2, calVar = false; end

    if calVar
        if nargin>3
            dmodel = predict_mean_var( X, dmodel, dif );
        else
            dmodel = predict_mean_var( X, dmodel );
        end
    else
        if nargin>3
            dmodel = predict_mean( X, dmodel, dif );
        else
            dmodel = predict_mean( X, dmodel );
        end
    end
    Y = dmodel.Y0;    V = dmodel.V0;
end

function [ dmodel ] = predict_mean( X, dmodel, dif )

   	k = length(dmodel.design);     
    theta = dmodel.theta(:);  	beta = dmodel.beta;    
    corr = dmodel.corr;    regr = dmodel.regr;
    delta2 = eps;    rho = feval(corr,theta,dmodel.design(1).S(1));
    Y = [];    V = [];
    if nargin<3
        Xi = X;
        F =  dmodel.regr(Xi); 
        n = size(Xi,1);
        Di = zeros(k,n);
        Ei = zeros(n,k);
        design = dmodel.design;
        for r = 1 : k
            C = chol(feval(corr,theta,design(r).S))';
            Ft = C\feval(regr,design(r).S);
            Yt = C\design(r).Y;
            IRri = C \ feval(corr,theta,design(r).S,Xi);
            Di(r,:) = dot(IRri, IRri) + delta2;  % dot(IRri, IRri) + delta2;
            Ei(:,r) = F*beta + IRri'*(Yt-Ft*beta);
        end
        Yi = zeros(n,1);    Vi = nan(n,1);
        for j = 1 : n
            w = 1./(rho-Di(:,j));    w = w/sum(w);
            Yi(j) = Ei(j,:)*w;
            Vi(j) = nan;
        end
        Y = [Y; Yi];    V = [V; Vi];
    else
        if dif > size(X,1), dif = size(X,1); end
        indeces = 0 : dif : size(X,1);
        if indeces(end)<size(X,1)
            indeces = [indeces(1:end-1) floor((indeces(end-1)+size(X,1))/2) size(X,1)];
        end
        parfor i = 2 : length(indeces)
            Xi = X(indeces(i-1)+1:indeces(i),:);
            F =  dmodel.regr(Xi);
            n = size(Xi,1);
            Di = zeros(k,n);
            Ei = zeros(n,k);
            design = dmodel.design;
            for r = 1 : k
                C = chol(feval(corr,theta,design(r).S))';
                Ft = C\feval(regr,design(r).S);
                Yt = C\design(r).Y;
                IRri = C \ feval(corr,theta,design(r).S,Xi);
                Di(r,:) = dot(IRri, IRri) + delta2; % dot(IRri, IRri) + delta2;
                Ei(:,r) = F*beta + IRri'*(Yt-Ft*beta);
            end
            Yi = zeros(n,1);    Vi = nan(n,1);
            for j = 1 : n
                w = 1./(rho-Di(:,j));    w = w/sum(w);
                Yi(j) = Ei(j,:)*w;
                Vi(j) = nan;
            end
            Y = [Y; Yi];    V = [V; Vi];
        end
    end
  
    dmodel.X0 = X;
    dmodel.Y0 = Y;
    dmodel.V0 = sqrt(dmodel.sigma2*V);
end

function [ dmodel ] = predict_mean_var( X, dmodel, dif )

	k = length(dmodel.design);     
    theta = dmodel.theta(:);  	beta = dmodel.beta;    
    corr = dmodel.corr;    regr = dmodel.regr;
    delta2 = eps;    rho = feval(corr,theta,dmodel.design(1).S(1));
    Y = [];    V = [];    
    if nargin<3
        Xi = X;
        F =  dmodel.regr(Xi); 
        n = size(Xi,1);
        Ei = zeros(n,k);
        design = dmodel.design;
        Ri = zeros(k,k,n); 
        for r = 1 : k
            design(r).C = chol(feval(corr,theta,design(r).S))';
            design(r).Ft = design(r).C\feval(regr,design(r).S);
            design(r).Yt = design(r).C\design(r).Y;
            IRri = design(r).C \ feval(corr,theta,design(r).S,Xi);
            Ri(r,r,:) = dot(IRri, IRri) + delta2; % Ri(r,r,:) = dot(IRri, IRri) + delta2;
            Ei(:,r) = F*beta + IRri'*(design(r).Yt-design(r).Ft*beta);
            design(r).IRri = design(r).C' \ IRri;
            for s = 1 : r-1
                Rrs = feval(corr,theta,design(r).S,design(s).S);
                Ri(r,s,:) = dot(design(r).IRri, Rrs*design(s).IRri); % Ri(r,s,:) = dot(design(r).IRri, Rrs*design(s).IRri);
                Ri(s,r,:) = Ri(r,s,:);
            end
        end
        Yi = zeros(n,1);    Vi = zeros(n,1);
        for j = 1 : n
            Eij = Ei(j,:);
            Rij = Ri(:,:,j);
            w = 1./(rho-diag(Rij));    w = w/sum(w);
            Yi(j) = Eij*w;
            Vi(j) = (rho + w'*Rij*w - 2*w'*diag(Rij) + delta2*(2-w'*w));
        end
        Y = [Y; Yi];    V = [V; Vi];
    else
        if dif>size(X,1), dif = size(X,1); end
        indeces = 0 : dif : size(X,1);      indeces(end) = size(X,1);
        parfor i = 2 : length(indeces)
            Xi = X(indeces(i-1)+1:indeces(i),:);
            F =  dmodel.regr(Xi);
            n = size(Xi,1);
            Ei = zeros(n,k);
            design = dmodel.design;
            
            Ri = zeros(k,k,n);
            for r = 1 : k
                C = chol(feval(corr,theta,design(r).S))';
                Ft = C\feval(regr,design(r).S);
                Yt = C\design(r).Y;
                IRri = C \ feval(corr,theta,design(r).S,Xi);
                rRi = zeros(k,n);
                rRi(r,:) = dot(IRri, IRri) + delta2; % dot(IRri, IRri) + delta2;
                Ei(:,r) = F*beta + IRri'*(Yt-Ft*beta);
                IRri = C' \ IRri;
                for s = 1 : r-1
                    C = chol(feval(corr,theta,design(s).S))';
                    IRsi = C \ feval(corr,theta,design(s).S,Xi);
                    IRsi = C' \ IRsi;
                    Rrs = feval(corr,theta,design(r).S, design(s).S);
                    rRi(s, :) = dot(IRri, Rrs*IRsi); % dot(IRri, Rrs*IRsi);
                end
                Ri(r,:,:) = rRi;
            end
            Yi = zeros(n,1);    Vi = nan(n,1);
            for j = 1 : n
                Eij = Ei(j,:);
                Rij = Ri(:,:,j);
                Rij = Rij + Rij' - diag(diag(Rij));
                w = 1./(rho-diag(Rij));    w = w/sum(w);
                Yi(j) = Eij*w;
                Vi(j) = (rho + w'*Rij*w - 2*w'*diag(Rij) + delta2*(2-w'*w));
            end
            Y = [Y; Yi];    V = [V; Vi];
        end
    end
    
    dmodel.X0 = X;
    dmodel.Y0 = Y;
    dmodel.V0 = sqrt(dmodel.sigma2*V);
end