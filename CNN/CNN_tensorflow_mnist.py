#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-08 16:06
# @Abstract：卷积神经网络实现手写数字识别

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("../MNIST_DATA/",one_hot=True)

#参数定义
learning_rate=0.1 #学习率
training_epochs=30 #训练轮数
batch_size=100 #每次选取训练集中的多少张
display_step=1 #训练多少轮显示一次

#神经网络参数定义
n_input=784 #输入层神经元个数 因为图像是28*28的
n_output=10 #输出层0-9这10个数字的分类

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

#创建卷积Convolution
def conv2d(x,W):
    #卷积核移动步长为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#创建池化Polling
def max_pool_2x2(x):
    #取对应的值中的最大值作为结果 ksize表示pool大小 strides，表示在height和width维度上的步长都为2
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#创建一个卷积神经网络
def multilayer_neuralNetwork_perceptron(x):
    #把输入x变成4维的x_image -1表示这个维度不考虑，其他三个维度
    x_image=tf.reshape(x,[-1,28,28,1])
    #第一层
    conv_w1=tf.Variable(tf.random_normal([5, 5, 1, 32])),  # 表示卷积核大小为5*5，第一层输入神经元为1，输出神经元为32
    conv_b1=tf.Variable(tf.random_normal([32])),
    #把x_image和权重进行卷积运算，加上偏向，应用ReLU激活函数
    h_conv_1=tf.nn.relu(conv2d(x_image,conv_w1)+conv_b1)
    # 进行max_pooling
    h_pool_1=max_pool_2x2(h_conv_1)

    #第二层
    conv_w2=tf.Variable(tf.random_normal([5, 5, 32, 64])),  # 表示卷积核大小为5*5，第二层输入神经元为32，输出神经元为64
    conv_b2=tf.Variable(tf.random_normal([64])),
    h_conv_2 = tf.nn.relu(conv2d(h_pool_1, conv_w2) + conv_b2)
    h_pool_2 = max_pool_2x2(h_conv_2)

    #第三层：全连接层
    fc1_w=tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),  # 7*7是第二层输出的size，64是第二层输出神经元个数
    fc1_b=tf.Variable(tf.random_normal([1024])),
    #把第2层的输出reshape成[batch, 7*7*64]的向量
    h_pool_2_flat=tf.reshape(h_pool_2,[-1,7*7*64])
    h_fcl=tf.nn.relu(tf.matmul(h_pool_2_flat,fc1_w)+fc1_b)

    #输出层
    out_w=tf.Variable(tf.random_normal([1024, n_output]))
    out_b=tf.Variable(tf.random_normal([n_output]))
    out_layer=tf.matmul(h_fcl,out_w)+out_b
    return out_layer


pred=multilayer_neuralNetwork_perceptron(x)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
# 使用优化器来做梯度下降
optimizer=tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("train begin------------------------------")
    for epoch in range(training_epochs):
        avg_cost=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cross_entropy],feed_dict={x:batch_x,y:batch_y})
            avg_cost+=c/total_batch

        if epoch%display_step==0:
            print("Epoch:",(epoch+1),"\tcost=",avg_cost)

    print("train finish------------------------------")

    # argmax给出某个tensor对象在某一维上数据最大值的索引 返回的索引是数值为1的位置
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.arg_max(y, 1))
    # 使用cast()把布尔值转换成浮点数，然后用reduce_mean()求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("last accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

'''
train begin------------------------------
Epoch: 1 	cost= 19435.68523615058
Epoch: 2 	cost= 4755.046649058948
Epoch: 3 	cost= 2901.9184252929713
Epoch: 4 	cost= 2161.063936545633
Epoch: 5 	cost= 1764.4960428133866
Epoch: 6 	cost= 1504.1513279030537
Epoch: 7 	cost= 1313.4847816051126
Epoch: 8 	cost= 1171.9040069025218
Epoch: 9 	cost= 1063.1205186323682
Epoch: 10 	cost= 969.3205413649305
Epoch: 11 	cost= 891.447065859708
Epoch: 12 	cost= 826.0648938265708
Epoch: 13 	cost= 767.97002353148
Epoch: 14 	cost= 719.8034816499193
Epoch: 15 	cost= 675.7560679505094
Epoch: 16 	cost= 636.566652365597
Epoch: 17 	cost= 601.9109007245851
Epoch: 18 	cost= 569.9593252788889
Epoch: 19 	cost= 541.7019673122062
Epoch: 20 	cost= 514.3408602558487
Epoch: 21 	cost= 489.6141661982103
Epoch: 22 	cost= 467.86595938595855
Epoch: 23 	cost= 447.9648810820148
Epoch: 24 	cost= 431.22644985708314
Epoch: 25 	cost= 412.8984012551739
Epoch: 26 	cost= 395.74465026248566
Epoch: 27 	cost= 379.78681266838885
Epoch: 28 	cost= 367.00043094786747
Epoch: 29 	cost= 351.9804138103084
Epoch: 30 	cost= 341.81561215563255
train finish------------------------------
'''