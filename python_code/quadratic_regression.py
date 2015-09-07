#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(1, 11))
y = [2, 4, 6, 28, 39, 64, 123, 213, 313, 424]

m = x.size
# X = np.c_[np.c_[np.ones((m, 1)), x], x**2]
X = np.c_[np.ones((m, 1)), x, x**2]

alpha = 0.0005
iteration = 1000

print "Initialize theta"
theta = np.array([[1.], [1.], [1.]])

for i in xrange(iteration):
    temp = np.array([
        [theta[0, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x + theta[2, 0] * (x**2) - y).T.dot(X[:, 0])],
        [theta[1, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x + theta[2, 0] * (x**2) - y).T.dot(X[:, 1])],
        [theta[2, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x + theta[2, 0] * (x**2) - y).T.dot(X[:, 2])]
    ])
    theta = temp

plt.plot(x, y, 'rx')
print 'Calculated theta'
print theta

plt.plot(X[:, 1], X.dot(theta), 'r-')
plt.show()


# End of Line.
