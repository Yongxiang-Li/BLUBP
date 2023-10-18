function z=GCV(B,y,yE)  % GCV for GPRR

n=length(B); 
H=B*((B'*B)\B');
tH=trace(H);
z=n*(y-yE)'*(y-yE)/(n-tH)^2;