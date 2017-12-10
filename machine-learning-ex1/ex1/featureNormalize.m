function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% initialize
X_norm = X;
mu = zeros(1, size(X, 2)); % mu is a row vector, all 0s to start, with n+1 elements (size of X returns row, column, so size of X,2 returns the number of columns of X)
sigma = zeros(1, size(X, 2));

% perform update on X and set X_norm

for i=1:size(mu,2)
	mu(1,i) = mean(X(:,i)); % mean of column i
	sigma(1,i) = std(X(:,i)); % standard deviation of column i
	X_norm(:,i) = (X(:,i)-mu(1,i))/sigma(1,i); % for each column of X we just perform the normalization formula with mu and sigma of that column
end

end
