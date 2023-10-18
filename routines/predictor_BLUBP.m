function  [dmodel, Y, V] = predictor_BLUBP(X, dmodel, dif)

    k = length(dmodel.design);
    theta = dmodel.theta(:);  	beta = dmodel.beta;
    corr = dmodel.corr;    regr = dmodel.regr;
    delta2 = eps;    rho = feval(corr,theta,dmodel.design(1).S(1));

    Y = [];    V = [];
    if nargin==2
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
            end
        end
        Yi = zeros(n,1);    Vi = zeros(n,1);
        for j = 1 : n
            Eij = Ei(j,:);
            Rij = Ri(:,:,j);
            Cij = chol(Rij')';

            Iu = Cij \ diag(Rij);                   Ii = Cij \ ones(k,1);
            Iw = (1-Ii'*Iu)/(Ii'*Ii)*Ii + Iu;       w = Cij' \ Iw;
            Yi(j) = Eij*w;
            Vi(j) = (rho + (1-Ii'*Iu)^2/(Ii'*Ii)-Iu'*Iu + delta2*(2-w'*w));
        end
        Y = [Y; Yi];    V = [V; Vi];
    else
        if dif > size(X,1), dif = size(X,1); end
        indeces = 0 : dif : size(X,1);
        if indeces(end)<size(X,1)
            indeces = [indeces(1:end-1) floor((indeces(end-1)+size(X,1))/2) size(X,1)];
        end
        parfor i = 2 : length(indeces)
            worker = getCurrentWorker;
            t = now;
            Xi = X(indeces(i-1)+1:indeces(i),:);
            F =  dmodel.regr(Xi);
            n = size(Xi,1);
            Ei = zeros(n,k);
            design = dmodel.design;
            Ri = zeros(k,k,n);
            for r = 1 : k
                design(r).C = sparse(tril(rand(size(design(r).S,1))));
                design(r).IRri = rand(size(design(r).S,1));
            end
            for r = 1 : k
                design(r).C = sparse(chol(feval(corr,theta,design(r).S), 'lower'));
                design(r).Ft = design(r).C\feval(regr,design(r).S);
                design(r).Yt = design(r).C\design(r).Y;
                IRri = design(r).C \ feval(corr,theta,design(r).S,Xi);
                Ri(r,r,:) = dot(IRri, IRri) + delta2; % Ri(r,r,:) = dot(IRri, IRri) + delta2;
                Ei(:,r) = F*beta + IRri'*(design(r).Yt-design(r).Ft*beta);
                design(r).IRri = design(r).C' \ IRri;
                for s = 1 : r-1
                    Rrs = feval(corr,theta,design(r).S,design(s).S);
                    Ri(r,s,:) = dot(design(r).IRri, Rrs*design(s).IRri); % Ri(r,s,:) = dot(design(r).IRri, Rrs*design(s).IRri);
                end
            end
            Yi = zeros(n,1);    Vi = zeros(n,1);
            for j = 1 : n
                Eij = Ei(j,:);
                Rij = Ri(:,:,j);
                Cij = chol(Rij')';

                Iu = Cij \ diag(Rij);                   Ii = Cij \ ones(k,1);
                Iw = (1-Ii'*Iu)/(Ii'*Ii)*Ii + Iu;       w = Cij' \ Iw;
                Yi(j) = Eij*w;
                Vi(j) = (rho + (1-Ii'*Iu)^2/(Ii'*Ii)-Iu'*Iu + delta2*(2-w'*w));
            end
            Y = [Y; Yi];    V = [V; Vi];
            disp(['Task ' num2str(i-1, '%03d') ' @ ' worker.Host ' Costs ' num2str(num2str((now-t)*24*3600), '%.4f') ' seconds']);
        end
    end


    dmodel.X0 = X;
    dmodel.Y0 = Y;
    dmodel.V0 = sqrt(dmodel.sigma2*V);
end