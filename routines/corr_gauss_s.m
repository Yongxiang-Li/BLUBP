function [ C ] = corr_gauss_s( phi, X, Y )
%COV Summary of this function goes here
%   Detailed explanation goes here
    delta = phi(end);    phi = phi(1:end-1);
    X = X.*sqrt(phi');
    if nargin == 2
        XY = X*X';
        C = exp(2*XY-(diag(XY)+diag(XY)'));
        C = C + delta^2*eye(size(C));
    else
        Y = (Y.*sqrt(phi'))';
        C = exp(2*X*Y-((X.^2)*ones(size(phi))+ones(size(phi'))*(Y.^2)));
    end
end

