#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

x = np.array([4.6, 0.0, 6.4, 6.5, 4.4, 1.1, 2.8, 5.1, 3.4, 5.8, 5.7, 5.5, 7.9, 3.0, 6.8, 6.2, 4.0, 8.6, 7.5, 1.3, 6.3, 3.1, 6.1, 5.3, 3.9, 5.8, 2.6, 4.8, 2.2, 5.3])
y = np.array([5.5, 1.7, 7.2, 8.3, 5.7, 1.1, 4.1, 6.7, 5.0, 6.6, 6.3, 5.6, 8.7, 3.6, 8.2, 6.2, 5.0, 9.5, 8.9, 2.6, 7.4, 5.0, 8.2, 6.6, 5.1, 7.0, 3.5, 6.3, 2.9, 6.9])

m = x.size
X = np.c_[np.ones((m, 1)), x]

alpha = 0.005
iterations = 100

print 'initialize theta'
print theta
theta = np.array([[1.], [1.]])

for i in xrange(iterations):
    temp = np.array([
        [theta[0, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x - y).T.dot(X[:, 0])],
        [theta[1, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x - y).T.dot(X[:, 1])]
    ])
    theta = temp

print 'Calculated theta'
print theta

plt.plot(x, y, 'rx')
plt.plot(X[:, 1], X.dot(theta), 'b-')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.legend(('Training Data', 'Linear Regression'), loc='upper left')

plt.show()

# End of Line.
