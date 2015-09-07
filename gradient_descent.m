# Training set
x = [4.6; 0.0; 6.4; 6.5; 4.4; 1.1; 2.8; 5.1; 3.4; 5.8; 5.7; 5.5; 7.9; 3.0; 6.8; 6.2; 4.0; 8.6; 7.5; 1.3; 6.3; 3.1; 6.1; 5.3; 3.9; 5.8; 2.6; 4.8; 2.2; 5.3]
y = [5.5; 1.7; 7.2; 8.3; 5.7; 1.1; 4.1; 6.7; 5.0; 6.6; 6.3; 5.6; 8.7; 3.6; 8.2; 6.2; 5.0; 9.5; 8.9; 2.6; 7.4; 5.0; 8.2; 6.6; 5.1; 7.0; 3.5; 6.3; 2.9; 6.9]

# Design Matrix
m = length(x);
X = [ones(m, 1) x];

# Parameter
alpha = 0.005;
iterations = 100;

# Initialize theta
fprintf('initialise theta\n');
theta = [1; 1]

# Gradient Descent
for iter = 1:iterations

  temp = [
    (theta(1, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x - y)' * X(:,1));
    (theta(2, 1) - alpha/m * (theta(1, 1) + theta(2, 1) * x - y)' * X(:,2))
  ]
  theta = temp;

  # Plot every line
  # plot(x, X * theta, 'b-');
  # hold on;

end

# Plot Dataset
plot(x, y, 'rx', 'MarkerSize', 8);
xlabel('xlabel');
ylabel('ylabel');

pause;

fprintf('Caliculated theta\n');
theta
hold on;

# Plot Linear Regression
plot(X(:, 2), X * theta, 'b-', 'linewidth', 2.0)
legend('Training Data', 'Linear Regression')
hold off;

pause;
