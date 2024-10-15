function  [Y, V, dmodel] = predictor_BLUBP_GPU(X, dmodel, dif)
    design = dmodel.design;
    k = length(design);    N = length(design(1).Y);       
    theta = dmodel.theta(:);  	beta = dmodel.beta;
    if length(theta)>size(design(1).S,2)
        gamma2 = theta(end)^2;      phi = theta(1:end-1);
    else
        gamma2 = eps;               phi = theta;
    end
    rho = 1 + gamma2;    delta2 = eps;
    if ~isfield(design,'F')
         for i = 1 : k
             design(i).F = feval(dmodel.regr, design(i).S);
         end
    end
    par.S = gpuArray(cat(3, design.S) .* sqrt(phi'));    par.S2 = sum(par.S.^2, 2);
    par.Y = gpuArray(cat(3, design.Y));    par.F = gpuArray(cat(3, design.F));
    S = gpuArray(X .* sqrt(phi'));              S2 = sum(S.^2, 2);

    if nargin==2
        indeces = [0 size(X,1)];
    else
        if dif > size(X,1), dif = size(X,1); end
        indeces = 0 : dif : size(X,1); 
        if indeces(end)<size(X,1)
            indeces = [indeces(1:end-1) floor((indeces(end-1)+size(X,1))/2) size(X,1)];
        end
    end
    
    invRd = pagefun(@inv, (exp(2*pagefun(@mtimes, par.S, pagefun(@transpose,par.S)) - ...
        (par.S2+rot90(par.S2))) + gamma2*eye(N,'gpuArray')));
    Y = [];    V = [];    
    t = now;
    for i = 2 : length(indeces)
        index = indeces(i-1)+1:indeces(i);   n = length(index);
        Sr = par.S;    Sr2 = par.S2;
        R = exp(2*pagefun(@mtimes, S(index,:), pagefun(@transpose,Sr)) - (S2(index)+rot90(Sr2))); % Rir
        invRir = pagefun(@mtimes, R, invRd);
        Di = permute(sum(invRir.*R, 2), [3 2 1]) + delta2;
        Fi = permute(dmodel.regr(X(index,:)) - pagefun(@mtimes, invRir, par.F), [3 2 1]);
        Ki = zeros(n,k,k, 'gpuArray');    Ki_ = zeros(n,k,k, 'gpuArray');  
        Yi = permute(pagefun(@mtimes, invRir, par.Y(:,:,1:k)), [3 2 1]);
        parfor r = 2 : ceil((k+1)/2)
            r1 = r;    r2 = k-r+2;
            indexS = [1:r1-1 1:r2-1];
            Ss = par.S(:,:,indexS);    Ss2 = par.S2(:,:,indexS);
            indexR = [r1*ones(1,r1-1) r2*ones(1,r2-1)];
            Sr = par.S(:,:,indexR);    Sr2 = par.S2(:,:,indexR);
            R = exp(2*pagefun(@mtimes, Sr, pagefun(@transpose,Ss)) - (Sr2+rot90(Ss2))); % Rrs
            Krs = sum(pagefun(@mtimes, invRir(:,:,indexR), R) .* invRir(:,:,indexS),2);
            v1 = zeros(n,k, 'gpuArray');    v2 = zeros(n,k, 'gpuArray');
            v1(:,1:r1-1) = Krs(:,:,1:r1-1);    v2(:,1:r2-1) = Krs(:,:,r1:end);
            Ki(:,:,r) = v1;     Ki_(:,:,r) = v2;
        end
        Ki(:,:,k-(2:ceil((k+1)/2))+2) = Ki_(:,:,2:ceil((k+1)/2));
        Ki = permute(Ki, [2 3 1]); % Ki = permute(Ki, [3 2 1]);
        Ki = Ki + pagefun(@transpose, Ki) + pagefun(@times, repmat(Di,[1,k,1]), eye(k));

        Il = pagefun(@mldivide, Ki, Di); % inverse of lambda
        Ii = pagefun(@mldivide, Ki, ones(k,1,'gpuArray'));
        iRi = pagefun(@mtimes, ones(1,k,'gpuArray'), Ii);
        iRl = pagefun(@mtimes, ones(1,k,'gpuArray'), Il);
        w = pagefun(@mtimes, Ii, (ones(1,'gpuArray')-iRl)./iRi)+Il;
        Ei = pagefun(@mtimes,pagefun(@mtimes,rot90(w),Fi),beta) + pagefun(@mtimes,rot90(w),Yi);
        Vi = ones(1,'gpuArray')*rho + (ones(1,'gpuArray')-iRl).^2./iRi ...
            -pagefun(@mtimes, rot90(Di), Il) + delta2*(2-pagefun(@mtimes, rot90(w), w));  
        Y = [Y; gather(permute(Ei,[3,2,1]))];    V = [V; gather(permute(Vi,[3,2,1]))];
        if dmodel.options.logOn,    disp(['Predict ' num2str(100*(i-1)/(length(indeces)-1)) '% data, which costs ' num2str((now-t)*24*3600) ' seconds']); end
    end
    
    dmodel.X0 = X;
    dmodel.Y0 = Y;
    dmodel.V0 = sqrt(dmodel.sigma2*V);
end
  