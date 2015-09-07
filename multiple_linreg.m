# Training set
x_1 = [2; 3; 5; 8]
x_2 = [89; 90; 88; 98]
y = [59; 61; 60; 71]

# Design Matrix
m = length(x_1)
X = [ones(m, 1) x_1 x_2]

# Parameter
alpha = 0.0002;
iterations = 1000;

# Initialize theta
fprintf('initialise theta\n');
theta = [1; 1; 1]

# Gradient Descent
for iter = 1:iterations

  temp = [
    (theta(1, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x_1 + theta(3, 1) * x_2 - y)' * X(:,1));
    (theta(2, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x_1 + theta(3, 1) * x_2 - y)' * X(:,2));
    (theta(3, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x_1 + theta(3, 1) * x_2 - y)' * X(:,3))
  ]
  theta = temp;

end

# Input
input = [1, 9, 88; 1, 11, 91]
# Predict
predict = input * theta