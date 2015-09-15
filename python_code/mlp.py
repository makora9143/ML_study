#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


class Layer:
    def __init__(self, in_dim, out_dim, function):

        self.rng = np.random.RandomState(1234)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.function = function

        self.W = theano.shared(
            self.rng.uniform(
                low=-0.08,
                high=0.08,
                size=(in_dim, out_dim)
            ).astype("float32"),
            name="W"
        )

        self.b =  theano.shared(
            np.zeros(out_dim).astype("float32"),
            name="bias"
        )

        self.params = [ self.W, self.b ]

    def fprop(self, x):
        h = self.function(T.dot(x, self.W)+self.b)
        self.h = h
        return h

def plot_sample(x, axis):
    img = x.reshape(28, 28)
    axis.imshow(img, cmap='gray')


def main():
    # 入力
    x = T.fmatrix("x")
    # 出力
    t = T.ivector("t")
    activation = T.nnet.sigmoid #T.tanh

    layers = [
        Layer(784, 500, activation),
        # Layer(500, 500, activation),
        Layer(500, 500, activation),
        Layer(500, 10, T.nnet.softmax)
    ]

    ## Collect Parameters and Symbolic output
    params = []
    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.fprop(x)
        else:
            layer_out = layer.fprop(layer_out)


    ## Cost Function (Negative Log Likelihood)
    y = layers[-1].h
    cost = - T.mean((T.log(y))[T.arange(x.shape[0]), t])

    ## Gradient
    gparams = T.grad(cost, params)

    ## Defile Learning Rule, you can add Adagrad, Adadelta etc.
    learning_rate = np.float32(0.1)

    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]


    ## Compile
    print 'Compile'
    train = theano.function(
        inputs=[x,t],
        outputs=cost,
        updates=updates
    )
    test = theano.function(
        inputs=[x,t],
        outputs=[cost, T.argmax(y, axis=1)]
    )

    print 'load mnist data'

    mnist = fetch_mldata('MNIST original')

# mnist_x is a (n_sample, n_feature=784) matrix
    mnist_x = mnist.data.astype("float32")/255.0
    mnist_y = mnist.target.astype("int32")
    train_x, valid_x, train_y, valid_y = train_test_split(mnist_x, mnist_y, test_size=0.2, random_state=42)


    print 'start training'
    ## Iterate
    batch_size = 100
    nbatches = train_x.shape[0] / batch_size
    for epoch in range(50):
        train_x, train_y = shuffle(train_x, train_y)  # Shuffle Samples !!
        for i in range(nbatches):
                start = i * batch_size
                end = start + batch_size
                err = train(train_x[start:end], train_y[start:end])
        if (epoch + 1) % 10 == 0:
            print "EPOCH:: %i, cost: %.3f"%(epoch+1, err)

    fig = plt.figure(figsize=(6, 6))
    for i in range(36):
        ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
        plot_sample(mnist_x[numpy.random.randint(0,60000)], ax)


if __name__ == '__main__':
    main()

# End of Line.
