#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt


x = T.vector("x")
t = T.vector("t")

theta_0 = theano.shared(1., name="theta_0")
theta_1 = theano.shared(1., name="theta_1")
params = [theta_0, theta_1]

y = x * theta_1 + theta_0

cost = T.mean((y - t) ** 2)

gparams = T.grad(cost, params)
updates = [(param, param - 0.005 * gparam)
           for param, gparam in zip(params, gparams)]

train = theano.function(
    inputs=[x, t],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True
)

data_x = np.array([4.6, 0.0, 6.4, 6.5, 4.4, 1.1, 2.8, 5.1, 3.4, 5.8, 5.7, 5.5, 7.9, 3.0, 6.8, 6.2, 4.0, 8.6, 7.5, 1.3, 6.3, 3.1, 6.1, 5.3, 3.9, 5.8, 2.6, 4.8, 2.2, 5.3], dtype=np.float32)
data_y = np.array([5.5, 1.7, 7.2, 8.3, 5.7, 1.1, 4.1, 6.7, 5.0, 6.6, 6.3, 5.6, 8.7, 3.6, 8.2, 6.2, 5.0, 9.5, 8.9, 2.6, 7.4, 5.0, 8.2, 6.6, 5.1, 7.0, 3.5, 6.3, 2.9, 6.9], dtype=np.float32)
for i in range(100):
    err  = train(data_x, data_y)
    print 'epoch: %d, error: %f' % (i, err)


print "After training theta_0 and theta_1"
print theta_0.get_value(), theta_1.get_value()

plt.plot(data_x, data_y, 'rx')
plt.plot(data_x, theta_0.get_value() + theta_1.get_value() * data_x, 'b-')

plt.show()


# End of Line.
