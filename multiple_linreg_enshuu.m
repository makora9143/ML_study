# Training set
x_1 = [2; 3; 5; 8]
x_2 = [89; 90; 88; 98]
y = [59; 61; 60; 71]

# Design Matrix
m = length(x_1)
X = # write here!

# Parameter
alpha = 0.0002;
iterations = 1000;

# Initialize theta
fprintf('initialise theta\n');
theta = [1; 1; 1]

# Gradient Descent
theta = # write here!

# Input
input = [1, 9, 88; 1, 11, 91]
# Predict
predict = input * theta