function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

doe1 = 0;
doe_grad = 0;
regularization = 0;
grad_adj = 0;

for i = 1:m,
            hypothesis = sigmoid(theta'*X(i,:)');
doe1 = doe1 - y(i)*log(hypothesis) - (1-y(i))*log(1-hypothesis);
doe_grad = doe_grad + (hypothesis - y(i))*X(i,:);
end;

grad(1) = (1/m) * doe_grad(1);

for j = 2:size(theta),
            regularization = regularization + theta(j)^2;
grad(j) = (1/m) * doe_grad(j) + (lambda/m) * theta(j);
end;

J = (1/m) * doe1 + (lambda/(2*m)) * regularization;
% =============================================================

end
