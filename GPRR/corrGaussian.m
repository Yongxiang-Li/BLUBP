function [ C ] = corrGaussian( theta, X, Y )
% Gaussian correlation between two points x and y with parameter theta
% x and y are row vectors
    X = X.*sqrt(theta');
    if nargin == 2
        XY = X*X';
        C = exp(2*XY-(diag(XY)+diag(XY)'));
        C = C + 1e-8*eye(size(C));
    else
        Y = (Y.*sqrt(theta'))';
        C = exp(2*X*Y-((X.^2)*ones(size(theta))+ones(size(theta'))*(Y.^2)));
    end
end