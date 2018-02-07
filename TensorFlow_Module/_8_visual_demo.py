#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-07 21:27
# @Abstract：画图可视化数据

import tensorflow as tf
import numpy as np
import _6_hiddenLayer_demo
import matplotlib.pyplot as plt


x_data=np.linspace(-1,1,300)[:,np.newaxis]
y_noise=np.random.normal(0,0.05,x_data.shape) #y的噪音
y_data=np.square(x_data)-0.5+y_noise #y=x^2-0.5

xs=tf.placeholder(tf.float32, [None,1])
ys=tf.placeholder(tf.float32, [None,1])

#第一层：输入层1个神经元 隐藏层10个神经元 输出层10个神经元
layer_1=_6_hiddenLayer_demo.add_layer(xs,1,10,activation_function=tf.nn.relu)

#第二层：输入层10个神经元 隐藏层1个神经元 输出层1个神经元
prediction=_6_hiddenLayer_demo.add_layer(layer_1,10,1,activation_function=None)

cost=tf.reduce_mean(
    tf.reduce_sum(tf.square(ys-prediction),
                   reduction_indices=[1])
)

learning_rate=0.5
#优化器 更正误差
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.initialize_all_variables()

fig=plt.figure()
ax=fig.add_subplot(1,1,1) #原始数据的图像
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1000):
        sess.run(optimizer,feed_dict={xs:x_data,ys:y_data})
        if epoch % 50==0:
            # print(sess.run(cost,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])  # 去除上一条
            except Exception: #因为第一次没有上一条
                pass
            prediction_value=sess.run(prediction,feed_dict={xs:x_data})
            lines=ax.plot(x_data,prediction_value,'r-',lw=5) #用曲线画上去
            plt.pause(1)





