#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-07 19:59
# @Abstract：假设有直线y=1*x+3,我们搭建神经网络结构，训练weight和baise，使之经过tensorflow神经网络训练，weight慢慢接近1 biase慢慢接近3

import tensorflow as tf
import numpy as np

#数据
x_data=np.random.rand(100).astype(np.float32)
#真实的Y
y_data=x_data*1+3

#定义tensorflow Struct-------------begin--------------------
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))#权重 初始随机产生-1～1的数
Biases=tf.Variable(tf.zeros([1])) #Biases初始=0

#预测的Y
y=Weights*x_data+Biases

#预测的Y与实际的Y
cost=tf.reduce_mean(tf.square(y-y_data))

#优化器
learning_rate=0.5
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train=optimizer.minimize(cost)

init=tf.initialize_all_variables()
#定义tensorflow Struct-------------end--------------------


#神经网络参数训练---------------------------
with tf.Session() as sess:
    #激活
    sess.run(init)
    print("---------------train begin-------------------")
    for epoch in range(100):
        sess.run(train)
        if epoch%10==0: #每隔10步输出一次训练结果
            print("epoch:",epoch,"\tWeights:",sess.run(Weights),"\tBiases:",sess.run(Biases),)
    print("---------------train end-------------------")



'''
执行结果：
---------------train begin-------------------
epoch: 0 	Weights: [2.3087187] 	Biases: [3.1080055]
epoch: 10 	Weights: [1.5514023] 	Biases: [2.7151241]
epoch: 20 	Weights: [1.3074411] 	Biases: [2.8411634]
epoch: 30 	Weights: [1.1714177] 	Biases: [2.9114387]
epoch: 40 	Weights: [1.0955762] 	Biases: [2.9506216]
epoch: 50 	Weights: [1.0532894] 	Biases: [2.9724684]
epoch: 60 	Weights: [1.0297122] 	Biases: [2.9846494]
epoch: 70 	Weights: [1.0165664] 	Biases: [2.991441]
epoch: 80 	Weights: [1.0092368] 	Biases: [2.9952278]
epoch: 90 	Weights: [1.0051502] 	Biases: [2.9973392]
---------------train end-------------------
'''


