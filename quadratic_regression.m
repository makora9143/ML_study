# Training set
x = [1:10]'
y = [2; 4; 6; 28; 39; 64; 123; 213; 313; 424]


# Design Matrix
m = length(x)
X = [ones(m, 1) x x.^2]

# Parameter
alpha = 0.0005;
iterations = 1000;

# Initialize theta
fprintf('initialise theta\n');
theta = [1; 1; 1]

# Gradient Descent
for iter = 1:iterations

  temp = [
    (theta(1, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x + theta(3, 1) * x.^2 - y)' * X(:, 1));
    (theta(2, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x + theta(3, 1) * x.^2 - y)' * X(:, 2));
    (theta(3, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x + theta(3, 1) * x.^2 - y)' * X(:, 3))
  ]
  theta = temp;

  # Plot every line
  # plot(x, X * theta, 'b-');
  # hold on;
 
end

# Plot Dataset
plot(x, y, 'rx');

fprintf('Caliculated theta\n');
theta
hold on; 

# Plot Linear Regression
plot(X(:, 2), X * theta, 'r-', 'linewidth', 2.0)
hold off;

pause;
