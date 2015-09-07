#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

# 入力の形（スカラーorベクトルor行列orテンソル）を決める
# いわゆる”xをベクトルとする．”というやつ
x = T.vector("x") # 入力
t = T.vector("t") # 正解

learning_rate = 0.005

# 今回必要となるパラメータ
theta_0 = theano.shared(1., name="theta_0") # θ_0: 切片
theta_1 = theano.shared(1., name="theta_1") # θ_1: 傾き
# まとめておくと便利
params = [theta_0, theta_1]

# 回帰したい数式
y = x * theta_1 + theta_0

# コスト関数．今回は最小二乗誤差を使用．
cost = T.mean((y - t) ** 2)

# ここが凄い．自動的に微分計算をしてくれる．
gparams = T.grad(cost, params)

# 各パラメータの更新
updates = [(param, param - learning_rate * gparam)
           for param, gparam in zip(params, gparams)]

# ここで関数を定義（コンパイル）
train = theano.function(
    inputs=[x, t], # 入力（引数）
    outputs=cost, # 出力
    updates=updates,
    allow_input_downcast=True
)

# ここまでで準備完了

# データを用意
data_x = np.array([4.6, 0.0, 6.4, 6.5, 4.4, 1.1, 2.8, 5.1, 3.4, 5.8, 5.7, 5.5, 7.9, 3.0, 6.8, 6.2, 4.0, 8.6, 7.5, 1.3, 6.3, 3.1, 6.1, 5.3, 3.9, 5.8, 2.6, 4.8, 2.2, 5.3], dtype=np.float32)
data_y = np.array([5.5, 1.7, 7.2, 8.3, 5.7, 1.1, 4.1, 6.7, 5.0, 6.6, 6.3, 5.6, 8.7, 3.6, 8.2, 6.2, 5.0, 9.5, 8.9, 2.6, 7.4, 5.0, 8.2, 6.6, 5.1, 7.0, 3.5, 6.3, 2.9, 6.9], dtype=np.float32)

# エポック数（試行回数）
for epoch in range(100):
    err  = train(data_x, data_y)
    print 'epoch: %d, error: %f' % (epoch, err)

print "After training, theta_0: %f, theta_1: %f" % (theta_0.get_value(), theta_1.get_value())

plt.plot(data_x, data_y, 'rx')
plt.plot(data_x, theta_0.get_value() + theta_1.get_value() * data_x, 'b-')

plt.show()


# End of Line.
