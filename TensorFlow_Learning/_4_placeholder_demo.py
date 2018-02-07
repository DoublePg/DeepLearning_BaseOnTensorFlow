#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-07 20:40
# @Abstract：tensorflow placeholder 用的时候传入值

import tensorflow as tf

input_1=tf.placeholder(tf.float32) # tensorflow大部分情况下只能处理float32形式
input_1_2=tf.placeholder(tf.float32,[2,2]) #规定只能两行两列
input_2=tf.placeholder(tf.float32)

output=tf.multiply(input_1,input_2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input_1:[5.2],input_2:[2]}))


'''
-------执行结果：------
[10.4]
'''


