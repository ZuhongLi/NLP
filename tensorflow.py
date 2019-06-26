'''
@author:Lee
@file:tensorflow.py
@Time: 2019/3/28 14:38
'''
import tensorflow as tf
import numpy as np
import utils
import os
import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# a  = tf.constant(2,name = 'a')
# b = tf.constant(3,name='b')
# x = tf.add(a,b,name='add')
# Writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
# with tf.Session() as sess:
#     print(sess.run(x))
# Writer.close()

#创建向量
# a = tf.constant(2,name = 'a')
# b = tf.constant(3,name='b')
# t0 = 19
# vec = tf.zeros_like(t0,np.float32)
# fill = tf.fill([2,3],6,)
# range = tf.range(3,9,1)
# num = tf.add_n([a,b,b])
# with tf.Session() as sess:
#     print(sess.run(vec))


#创建变量
# x = tf.get_variable('scalar',initializer=tf.constant(2))
# m = tf.get_variable('matrix',initializer=tf.constant([[0,1],[2,3]]))
# b = tf.get_variable('big_matrix',shape=(100,10),initializer=tf.truncated_normal_initializer())
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(m))

#变量赋值
# a = tf.get_variable('scalar', initializer=tf.constant(2))
# a_times_two = a.assign(a * 2)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(a_times_two)) # >> 4
#     print(sess.run(a_times_two)) # >> 8
#     print( sess.run(a_times_two)) # >> 16

# 创建交互会话
# sess = tf.InteractiveSession()
# a = tf.constant(5.0)
# b = tf.constant(6.0)
# c = a * b
# print(c.eval()) # we can use 'c.eval()' without explicitly stating a session
# sess.close()

#placeholder和feeddict
# tf.placeholder(shape=None,name=None)  # 当shape=none的时候， 表名可以接受任意shape的张量tensors
# a = tf.placeholder(tf.int32,shape=[3],name='name')
# b = tf.constant([5,5,5],tf.int32)
# c = a+b
# with tf.Session() as sess:
#     print(sess.run(c,feed_dict={a:[1,2,3]}))


# 使用线性回归预测人均寿命
data_file = './datasets/birth_life_2010.txt'

# step1:read datasets
data,n_samples = utils.read_birth_life_data(data_file)

# step2:create placeholder for X(birth_rate) and Y(life_expectancy)
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

# step3:create weight and bias ,initialized to 0
w = tf.get_variable('Weight',initializer=tf.constant(0.0))
b = tf.get_variable('Bias',initializer=tf.constant(0.0))

# step4:construct model for prediction
predict_y = w*X+b

# step5:use the square error to be loss function
loss = tf.square(Y-predict_y,name='loss')

# step6:use gradient decent with learning rate 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # step7:initialize global variables
    sess.run(tf.global_variables_initializer())

    #step8:train model
    for i in range(100):
        for x,y in data:
            sess.run(optimizer,feed_dict = {X:x,Y:y})

    #step9: output w and b
    out_w,out_b = sess.run([w,b])

# step10: draw the plot
plt.plot(data[:,0],data[:,1],'bo',label='Real data')
plt.plot(data[:,0],data[:,0]*out_w+out_b,'r',label = 'Predict data')
plt.legend()
plt.show()


# 显示日志
# exp1:简单创建log writer
# a = tf.constant(2,tf.int32,name='a')
# b = tf.constant(3,tf.int32,name = 'b')
# c = tf.add(a,b,name = 'add')
# writer = tf.summary.FileWriter('./log',tf.get_default_graph())
# with tf.Session() as sess:
#     sess.run(c)
# writer.close()

# div的用法
# a = tf.constant([2,2],name = 'a')
# b = tf.constant([[0,1],[2,3]],name = 'b')
# with tf.Session() as sess:
#     print(sess.run(tf.div(b,a)))  #对应元素相除，取商
#     print(sess.run(tf.divide(b,a))) #对应元素相除
#     print(sess.run(tf.truediv(b,a))) #对应元素相除
#     print(sess.run(tf.floordiv(b,a))) #结果向下取整，输出结果与输入一致
#     print(sess.run(tf.truncatediv(b,a))) #结果截断除，取余
#     print(sess.run(tf.floor_div(b,a))) #结果向下取整，输出结果与输入一致

# 例子3：乘法
# a = tf.constant([10, 20], name='a')
# b = tf.constant([2, 3], name='b')
# with tf.Session() as sess:
#     print(sess.run(tf.multiply(a, b))) #各元素相乘，结果仍是矩阵
#     print(sess.run(tf.tensordot(a, b, 1))) #点乘，各元素相乘，然后相加，结果为整数
#
# 例子4：Python 基础数据类型
# t_0 = 19
# x = tf.zeros_like(t_0)
# y = tf.ones_like(t_0)
# print(x)
# print(y)
#
# t_1 = ['apple', 'peach', 'banana']
# x = tf.zeros_like(t_1)                  # ==> ['' '' '']
# y = tf.ones_like(t_1)                           # ==> TypeError:

# t_2 = [[True, False, False],
#        [False, False, True],
#        [False, True, False]]
# x = tf.zeros_like(t_2)                  # ==> 3x3 tensor, all elements are False
# y = tf.ones_like(t_2)                   # ==> 3x3 tensor, all elements are True
#
# print(tf.int32.as_numpy_dtype())
#
# # Example 5: printing your graph's definition
# my_const = tf.constant([1.0, 2.0], name='my_const')
# print(tf.get_default_graph().as_graph_def())

#Placeholder和feeddict_的用法
# a = tf.placeholder(tf.int32,shape=[3],name='a')
# b = tf.constant([5,5,5],tf.int32,name='b')
# c = tf.add(a,b,name = 'c')
# write = tf.summary.FileWriter('./log',graph=tf.get_default_graph())
# with tf.Session() as sess:
#     print(sess.run(c,feed_dict={a:[1,2,3]}))
# write.close()

# exa2
# 如果在run中传入feed_dict，run会优先读取feed_dict的值
# a = tf.add(2,3)
# b = tf.multiply(a,5)
# with tf.Session() as sess:
#     print(sess.run(b))
#     print(sess.run(b,feed_dict={a:10}))

# Variable example
# s = tf.Variable(2,name = 'scalar')
# m = tf.Variable([[0,1],[2,3]],name='matrix')
# w = tf.Variable([748,10],name='Big_matrix')
# v = tf.Variable(tf.truncated_normal([748,10]),name='normal_matrix')

## 2种语句效果是一样的
# s = tf.get_variable(name='scalar',initializer=tf.constant(2))
# m = tf.get_variable(name='matrix',initializer=tf.constant([[0,1],[2,3]]))
# w = tf.get_variable(name='Big_matrix',shape=(748,10),initializer=tf.zeros_initializer())
# v = tf.get_variable(name='normal_matrix',shape=(748,10),initializer=tf.truncated_normal_initializer())
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(v.eval())


# exp2 assign values to variables
# w = tf.Variable(10)
# w.assign(100)
# with tf.Session() as sess:
#     # sess.run(W)
#     sess.run(w.initializer)
#     print(sess.run(w))
#     print(sess.run(w.assign_add(10)))
#     print(sess.run(w.assign_sub(2)))



# logistic regression
# learning_rate =0.01
# batch_size = 128  #每批数据量的大小
# n_epochs = 30
#
# # read datasets
# mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
# X_batch,Y_batch = mnist.train.next_batch(batch_size)
#
# # 创建placeholder
# # 每张图片是28 * 28 = 784，因此一张图片用1x784的tensor标示
# # 每张图片属于10类之一，0-9十个数字,每个标签是一个one hot 向量,比如图片“1”，[0,1,0,0,0,0,0,0,0,0]
# X = tf.placeholder(tf.float32,[batch_size,784],name='image')
# Y = tf.placeholder(tf.float32,[batch_size,10],name='label')
#
#
# # 创建权重和偏置
# # w为随机变量，服从平均值为0，标准方差(stddev)为0.01的正态分布
# # b初始化为0
# # w的shape取决于X和Y  Y = tf.matmul(X, w)
# # b的shape取决于Y
# # Y=Xw+b  [1,10]=[1,784][784,10]+[1,10]
# w = tf.get_variable(name='weight',shape=(784,10),
#                     initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
# b = tf.get_variable(name='b',shape=(1,10),initializer=tf.zeros_initializer())
#
#
# # 创建模型
# logist = tf.multiply(X,w)+b
#
# # 定义损失函数
# entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logist,labels='Y',name='loss')
# loss = tf.reduce_mean(entropy)  # 计算一个batch下的平均loss值
#
# # 定义训练的optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# # 计算测试集的准确度
# pred = tf.nn.softmax(logist)
# correct_rate = tf.equal(tf.argmax(pred,1),tf.arg_max(Y,1))
# accuracy = tf.reduce_sum(tf.cast(correct_rate,tf.float32))
# writer = tf.summary.FileWriter('./graphs/logistic',graph=tf.get_default_graph())
# with tf.Session() as sess:
#     start_time = time.time()
#     sess.run(tf.global_variables_initializer())
#     n_batches = int(mnist.train.num_examples / batch_size)
#     print(n_batches)
#
#     #训练模型
#     for i in range(n_epochs):
#         total_loss = 0
#         for j in range(n_batches):
#             X_batch,Y_batch = mnist.train.next_batch(batch_size)
#             _,loss_batch = sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
#             total_loss = total_loss+loss_batch
#         print('average loss {0}:{1}'.format(i,total_loss/n_batches))
#     print('total time:{0}'.format(time.time()-start_time))
#
#     # 测试模型
#     n_batches = int(mnist.test.num_examples / batch_size)
#     total_correct_preds = 0
#
#     for i in range(n_batches):
#         X_batch, Y_batch = mnist.test.next_batch(batch_size)
#         accuracy_batch = sess.run(accuracy, {X: X_batch, Y: Y_batch})
#         total_correct_preds += accuracy_batch
#     print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
# writer.close()


# argmax返回vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，
# 那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

# A = np.arange(1,8,2).reshape(1,4)
# print('A:',A)
# B = np.arange(1,7).reshape(2,3)
# print('B:',B)
# with tf.Session() as sess:
#     print('A沿Y轴的最大值：',sess.run(tf.arg_max(A,1)))
#     print('B沿Y轴的最大值：', sess.run(tf.arg_max(B, 1)))
#     print('A沿X轴的最大值：', sess.run(tf.arg_max(A, 0)))
#     print('B沿X轴的最大值：', sess.run(tf.arg_max(B, 0)))


# a = tf.get_variable(name = 'a',shape=[3,4],dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
# b = tf.arg_max(input=a,dimension=1)
# c = tf.arg_max(input=a,dimension=0)
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# print(sess.run(a))
# print(sess.run(b))
# print(sess.run(c))

# a = np.arange(0,6).reshape(2,3)
# print(a)
# with tf.Session() as sess:
#     print('a中所有元素之和为：',sess.run(tf.reduce_sum(a)))
#     print('a中所有元素沿x轴平均值为：',sess.run(tf.reduce_mean(a,axis=1)))
#     print('a中所有元素沿Y轴平均值为：',sess.run(tf.reduce_mean(a,axis=0)))

# tf.equal(real, prediction)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，
# 否则返回False，返回的值的矩阵维度和real是一样的，我们会在求准确率的时候经常用到它
# real = [1,2,3,4]
# pred = [1,2,3,5]
# with tf.Session() as sess:
#     corr_pred = tf.equal(real,pred)
#     print(sess.run(corr_pred))
#     correct_pred_num = tf.cast(corr_pred,tf.float32)
#     print(sess.run(correct_pred_num))
#     accuracy = tf.reduce_mean(correct_pred_num)
#     print(sess.run(accuracy))


#tf.nn.embedding_lookup()就是根据input_ids中的id， 寻找embeddings中的第id行。
# 比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回。
# input_ids = tf.placeholder(dtype=tf.int32,shape=None,name='input_dis')
# embeding = tf.Variable(np.identity(5,dtype=np.int32))
# input_embeding = tf.nn.embedding_lookup(embeding,input_ids)
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# print('embedings:\n',embeding.eval())
# print('input_embedings:\n',sess.run(input_embeding,feed_dict={input_ids:[1,2,3,0,3,2,1]}))

