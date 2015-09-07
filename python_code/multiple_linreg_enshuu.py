#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

x_1 = np.array([4, 5, 6, 8])
x_2 = np.array([97, 100, 98, 80])
y = np.array([62, 46, 50 ,55])

m = x_1.size
X = np.c_[np.ones((m, 1)), x_1, x_2]

alpha = 0.0002
iterations = 1000

theta = np.array([[1.], [1.], [1.]])

for _ in xrange(iterations):
    tmp = np.array([
        [theta[0, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x_1 + theta[2, 0] * x_2 - y).T.dot(X[:, 0])],
        [theta[1, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x_1 + theta[2, 0] * x_2 - y).T.dot(X[:, 1])],
        [theta[2, 0] - alpha / m * (theta[0, 0] + theta[1, 0] * x_1 + theta[2, 0] * x_2 - y).T.dot(X[:, 2])]
    ])
    theta = tmp

# End of Line.
