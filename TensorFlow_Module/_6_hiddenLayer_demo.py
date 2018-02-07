#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-07 20:53
# @Abstract：定义神经网络hidden层

import tensorflow as tf

#定义添加隐藏层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))#in_size行out_size列的矩阵
    Biases=tf.Variable(tf.random_normal([1,out_size])) #1行out_size列
    wx_b=tf.matmul(inputs,Weights)+Biases
    if activation_function is None:#线性关系 不用加activation_function
        outputs=wx_b
    else:
        outputs=activation_function(wx_b)

    return outputs




