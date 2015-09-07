# Training set
x = [4.6; 0.0; 6.4; 6.5; 4.4; 1.1; 2.8; 5.1; 3.4; 5.8; 5.7; 5.5; 7.9; 3.0; 6.8; 6.2; 4.0; 8.6; 7.5; 1.3; 6.3; 3.1; 6.1; 5.3; 3.9; 5.8; 2.6; 4.8; 2.2; 5.3]
y = [5.5; 1.7; 7.2; 8.3; 5.7; 1.1; 4.1; 6.7; 5.0; 6.6; 6.3; 5.6; 8.7; 3.6; 8.2; 6.2; 5.0; 9.5; 8.9; 2.6; 7.4; 5.0; 8.2; 6.6; 5.1; 7.0; 3.5; 6.3; 2.9; 6.9]

# Design Matrix
m = length(x);
X = [ones(m, 1) x];

# Normal Equation
theta = # write here!

plot(x, y, 'rx', 'MarkerSize', 8);
xlabel('xlabel');
ylabel('ylabel');
hold on;
pause;

plot(X(:, 2), X*theta, 'b-', 'linewidth', 2.0)
legend('Training data', 'Linear regression')
hold off;
pause;
