#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-06 10:02
# @Abstract：传统神经网络通过tensorflow实现mnist手写数字识别

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("../MNIST_DATA/",one_hot=True)

import tensorflow as tf

#参数定义
learning_rate=0.1 #学习率
training_epochs=30 #训练轮数
batch_size=100 #每次选取训练集中的多少张
display_step=1 #训练多少轮显示一次

#神经网络参数定义
n_input=784 #输入层神经元个数 因为图像是28*28的
n_hidden1=256 #第一个隐藏层神经元个数
n_hidden2=512
n_output=10 #输出层0-9这10个数字的分类

#tensorflow输入
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_output])

#创建一个两层的传统神经网络
def multilayer_neuralNetwork_perceptron(x,weights,biases):
    # 隐藏层使用ReLU非线性激活函数
    layer1=tf.add(tf.matmul(x,weights['w1']),biases['b1'])
    layer1=tf.nn.relu(layer1)

    layer2=tf.add(tf.matmul(layer1,weights['w2']),biases['b2'])
    layer2=tf.nn.relu(layer2)

    outLayer=tf.matmul(layer2,weights['w_out'])+biases['b_out']

    return outLayer

#正态分布随机初始化权重和偏向矩阵
weights={
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden1])),
    'w2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    'w_out':tf.Variable(tf.random_normal([n_hidden2,n_output]))
}

biases={
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'b_out': tf.Variable(tf.random_normal([n_output]))
}

pred=multilayer_neuralNetwork_perceptron(x,weights,biases)

#调用softmax_cross_entropy计算损失
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#优化程序,求最小的cost
optimizer=tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print("train start.........................")
    #训练training_epochs轮
    for epoch in range(training_epochs):
        avg_cost=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)

            _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})

            avg_cost+=c/total_batch

        correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        if epoch%display_step==0:
            print("Epoch:",'%04d' % (epoch+1),"\tcost=","{:.9f}".format(avg_cost),"\taccuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("train fineshed.........................")

    correct_prediction=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("last accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))








