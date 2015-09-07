# Training set
x_1 = [4; 5; 6; 8]
x_2 = [97; 100; 98; 80]
y = [62; 46; 50; 55]

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
# write here!

# New data
newdata = [1, 7, 80; 1, 9, 80]

# Predict
pred = newdata * theta
