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
        Yi = permute(pagefun(@mtimes, invRir, par.Y(:,:,1:k)), [3 2 1]);
        if k > 1
            % Build upper-triangle index pairs (row > col) on CPU — one shot
            idx_all_r = repelem(1:k, k);
            idx_all_s = repmat(1:k, [1 k]);
            keep = idx_all_r > idx_all_s;
            idx_r = idx_all_r(keep);   % 1 × k(k-1)/2
            idx_s = idx_all_s(keep);

            % Two GPU copies of par.S and par.S2 (was: ~3k per r-iteration)
            Sr_b = par.S(:,:,idx_r);    Ss_b = par.S(:,:,idx_s);
            S2r_b = par.S2(:,:,idx_r);  S2s_b = par.S2(:,:,idx_s);

            R_batch = exp(2 * pagefun(@mtimes, Sr_b, pagefun(@transpose, Ss_b)) ...
                    - (S2r_b + rot90(S2s_b)));

            invRr_b = invRir(:,:,idx_r);  invRs_b = invRir(:,:,idx_s);
            T = pagefun(@mtimes, invRr_b, R_batch);
            Krs_up = squeeze(sum(T .* invRs_b, 2));  % n × k(k-1)/2
            if n == 1
                Krs_up = Krs_up(:)';
            end

            % Scatter upper-triangle values into Ki via linear indexing
            lin_idx = sub2ind([k k], idx_r, idx_s);
            Ki = zeros(n, k*k, 'gpuArray');
            Ki(:, lin_idx) = Krs_up;
            Ki = reshape(Ki, [n k k]);
        else
            Ki = zeros(n, k, k, 'gpuArray');
        end
        Ki = permute(Ki, [2 3 1]);
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
  