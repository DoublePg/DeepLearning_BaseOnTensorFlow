#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-05 18:23
# @Abstract：测试tensorflow导入情况

# 引入 tensorflow 模块

import tensorflow as tf

# 创建一个整型常量，即0阶 Tensor
t0 = tf.constant(3, dtype=tf.int32)

# 创建一个浮点数的一维数组，即1阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)

# 创建一个字符串的2x2数组，即2阶Tensor
t2 = tf.constant([['1', 'YEN'], ['2', 'LMC']], dtype=tf.string)

# 打印上面创建的几个 Tensor
print("t0:",t0)
print("t1:",t1)
print("t2:",t2)
