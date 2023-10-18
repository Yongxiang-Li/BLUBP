function [ C ] = corr_gauss_x( phi, X, Y )
%COV Summary of this function goes here
%   Detailed explanation goes here
    X = X.*sqrt(phi');
    if nargin == 2
        XY = X*X';
        C = exp(2*XY-(diag(XY)+diag(XY)'));
        C = C + eps*eye(size(C));
    else
        Y = (Y.*sqrt(phi'))';
        C = exp(2*X*Y-((X.^2)*ones(size(phi))+ones(size(phi'))*(Y.^2)));
    end
end

