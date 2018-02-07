#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-07 20:31
# @Abstract：使用tensorflow定义变量


import tensorflow as tf

#定义变量
var1=tf.Variable

var2=tf.Variable(0,name='count')
print(var2.name)

#定义常量
var3=tf.constant(5)
new_value=tf.add(var2,var3)
update=tf.assign(var2,new_value)
print(var2)

init=tf.initialize_all_variables() # 初试化所有的变量才会被激活

with tf.Session() as sess:
    sess.run(init)

    for x in range(3):
        sess.run(update)
        print(sess.run(var2))

'''
---执行结果----
5
10
15
'''