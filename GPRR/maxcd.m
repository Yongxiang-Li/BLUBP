function maxd=maxcd(D)   % space-filling crieterion of D (negative power transformation with lambda=0.5)

    [n p]=size(D); maxd=0;
    for i=1:n-1
        for j=i+1:n
            d=sum(1./abs(D(i,:)-D(j,:)));
            if maxd<d
                maxd=d;
            end
        end
    end