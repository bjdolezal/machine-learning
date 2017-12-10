function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % an empty vector to store our cost histories
thetaLen = length(theta); % this is how many thetas we need to update
tempVal = theta; % we will use this vector to temporarily update theta and then update the real theta all at once

for iter = 1:num_iters
	temp = (X*theta - y); % this is the diff of prediction and actual, will be included in sum
	
	for i=1:thetaLen
		tempVal(i,1) = sum(temp.*X(:,i)) % this is the key step
	end
	
	theta = theta - (alpha/m)*tempVal; % updating theta
	
	J_history(iter,1) = computeCost(X, y, theta);

end

end
