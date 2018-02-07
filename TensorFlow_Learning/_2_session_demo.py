#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-07 20:21
# @Abstract：Session打开方式

import tensorflow as tf


matrix1=tf.constant([[3,6]]) #1行2列
matrix2=tf.constant([[2],
                     [4]]) #2行1列


product=tf.multiply(matrix1,matrix2)

print(product)

#method one:
sess1=tf.Session()
result1=sess1.run(product)
print(result1)
sess1.close()


#method two:
with tf.Session() as sess2:
    result2 = sess2.run(product)
    print(result2)

'''
----执行结果----
[[ 6 12]
 [12 24]]
'''