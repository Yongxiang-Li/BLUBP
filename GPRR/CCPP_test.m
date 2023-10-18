% standardization
data_CCPP=textread('CCPP.txt'); 
data_CCPP_X = data_CCPP(:,1:4);
data_CCPP_Y = data_CCPP(:,5);
maxdata=max(data_CCPP_X);
mindata=min(data_CCPP_X);
[N0, d]=size(data_CCPP_X); DATA_st=zeros(N0,d);
for j=1:d
    for i=1:N0
        DATA_st(i,j)=(data_CCPP(i,j)-mindata(j))/(maxdata(j)-mindata(j));
    end
end
data_CCPP_X=DATA_st;