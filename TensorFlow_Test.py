#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-05 18:23
# @Abstract：测试tensorflow导入情况

# 引入 tensorflow 模块

import tensorflow as tf
# # 创建一个整型常量，即0阶 Tensor
# t0 = tf.constant(3, dtype=tf.int32)
#
# # 创建一个浮点数的一维数组，即1阶 Tensor
# t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)
#
# # 创建一个字符串的2x2数组，即2阶Tensor
# t2 = tf.constant([['1', 'YEN'], ['2', 'LMC']], dtype=tf.string)
#
# # 打印上面创建的几个 Tensor
# print("t0:",t0)
# print("t1:",t1)
# print("t2:",t2)

#官方教程代码测试：
#Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
#Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#Runs the op.
print(sess.run(c))
