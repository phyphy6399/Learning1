# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

#基本的なパッケージ
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

#応用パッケージ
import tensorflow as tf

#TensorFlowサンプルコード
#入力Xから出力Yを予想する

X_train = np.arange(10).reshape((10,1))
Y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

#線形解禁モデルクラス
class TfLinreg(object): #継承クラスなし
    
    def __init__(self, x_dim, learning_rate = 0.01, random_seed = None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph();    #計算グラフ(データの流れを図化したもの)を作成
        
        # モデルを構築
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()
            
    def build(self):
        
        #入力データのプレースホルダを作成
        self.X = tf.placeholder(dtype = tf.float32, shape = (None, self.x_dim), name = 'x_input')
        self.Y = tf.placeholder(dtype = tf.float32, shape = (None), name = 'y_input')
        print(self.X)
        print(self.Y)
        
        w = tf.Variable(tf.zeros(shape = (1)), name = 'weight')
        b = tf.Variable(tf.zeros(shape = (1)), name = 'bias')
        print(w)
        print(b)
        
        self.z_net = tf.squeeze(w * self.X + b, name = 'z_net')
        print(self.z_net)
        
        sqr_errors = tf.square(self.Y - self.z_net, name = 'sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name = 'mean_cost')
        
        #optimizerを作成
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate, name = 'GrandDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)
 
#インスタンス作成       
lrmodel = TfLinreg(x_dim = X_train.shape[1], learning_rate = 0.01)

        
