 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  sliceLHD: Function to generate sliced Latin hypercube designs
 %  Arguments: 
 %        m: the sample size of the small LHD 
 %        n: the sample size of the large LHD, where n = mt
 %        t: the number of slices
 %        q: the number of factors 
 %    Value: a n-by-q LHD with t slices forming a small LHD of size m
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xmat] = slhd(m, t, q)
	n = m*t ;
	P = zeros(m, t);
	Q = zeros(m, t);
	xmat = zeros(m,q,t) ;
    for l = 1:q  
        for i = 1:m 
            Q(i,:) = randsample(1:t, t);
        end
        for j = 1:t 
            I = randsample(1:m, m)';
            P(:,j) = (I-1)*t + Q(I,j);
        end
        xmat(:,l,:)  = ((P-rand(size(P)))/n);
    end

