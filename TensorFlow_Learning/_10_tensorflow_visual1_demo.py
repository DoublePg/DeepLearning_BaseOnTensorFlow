#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-08 20:14
# @Abstract：可视化神经网络框架z
import tensorflow as tf
import numpy as np

x_data=np.linspace(-1,1,300)[:,np.newaxis]
y_noise=np.random.normal(0,0.05,x_data.shape) #y的噪音
y_data=np.square(x_data)-0.5+y_noise #y=x^2-0.5

#定义添加隐藏层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weigt'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='WEIGHT')#in_size行out_size列的矩阵
        with tf.name_scope('weigt'):
            Biases=tf.Variable(tf.random_normal([1,out_size]),name='BIASE') #1行out_size列
        with tf.name_scope('WEIGHT_BIASE'):
            wx_b=tf.matmul(inputs,Weights)+Biases
    if activation_function is None:#线性关系 不用加activation_function
        outputs=wx_b
    else:
        outputs=activation_function(wx_b)

    return outputs

#输入模块
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32, [None,1],name="x_input")
    ys=tf.placeholder(tf.float32, [None,1],name="y_input")

#隐藏层模块

#第一层：输入层1个神经元 隐藏层10个神经元 输出层10个神经元
layer_1=add_layer(xs,1,10,activation_function=tf.nn.relu)

#第二层：输入层10个神经元 隐藏层1个神经元 输出层1个神经元
prediction=add_layer(layer_1,10,1,activation_function=None)
with tf.name_scope('LOSS'):
    cost=tf.reduce_mean(
        tf.reduce_sum(tf.square(ys-prediction),
                       reduction_indices=[1])
    )

learning_rate=0.5
#优化器 更正误差
with tf.name_scope('TRAIN'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for epoch in range(200):
        sess.run(optimizer,feed_dict={xs:x_data,ys:y_data})
        if epoch%10==0:
            print(sess.run(cost,feed_dict={xs:x_data,ys:y_data}))