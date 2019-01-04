import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
import numpy as np
tf.reset_default_graph()

#placeholder
X = tf.placeholder(tf.float32,[None, 784])
X_img = tf.reshape(X,[-1,28,28,1])
Y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#layer1 var
layer1_filter_num = 32
layer1_filter_size = 4
layer1_conv2d_stride = 1
layer1_ksize = 2
layer1_pool_stride = 2
#layer2 var
layer2_height_and_width = int(28/(layer1_conv2d_stride*layer1_pool_stride))
layer2_filter_num = 64
layer2_filter_size = 4
layer2_conv2d_stride = 1
layer2_ksize = 2
layer2_pool_stride = 2
#fully-connected layer
elements_num = int(layer2_height_and_width*layer2_height_and_width/layer2_conv2d_stride/layer2_conv2d_stride/layer2_pool_stride/layer2_pool_stride*layer2_filter_num)
#train
sess = tf.Session()
learning_rate = 0.001
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

#layer1
with tf.variable_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([layer1_filter_size,layer1_filter_size,1,layer1_filter_num],dtype=tf.float32), name='W1')
    conv2d1 = tf.nn.conv2d(X_img,W1,strides=[1,layer1_conv2d_stride,layer1_conv2d_stride,1],padding="SAME", name='conv2d1')
    relu1 = tf.nn.relu(conv2d1,name='relu1')
#     relu1 = tf.nn.dropout(relu1,keep_prob)
    pool1 = tf.nn.max_pool(relu1, ksize=[1,layer1_ksize,layer1_ksize,1],strides=[1,layer1_pool_stride,layer1_pool_stride,1],padding='SAME',name='pool1')
    pool1 = tf.nn.dropout(pool1,keep_prob)
#layer2
with tf.variable_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([layer2_filter_size,layer2_filter_size,layer1_filter_num,layer2_filter_num],dtype=tf.float32), name='W2')
    conv2d2 = tf.nn.conv2d(pool1,W2,strides=[1,layer2_conv2d_stride,layer2_conv2d_stride,1],padding="SAME", name='conv2d2')
    relu2 = tf.nn.relu(conv2d2,name='relu2')
#     relu2 = tf.nn.dropout(relu2,keep_prob)
    pool2 = tf.nn.max_pool(relu2, ksize=[1,layer2_ksize,layer2_ksize,1],strides=[1,layer2_pool_stride,layer2_pool_stride,1],padding='SAME',name='pool2')
    pool2 = tf.nn.dropout(pool2,keep_prob)
    #fully-connected layer
with tf.variable_scope('fully-connected_layer'):
    X_affine = tf.reshape(pool2, [-1,elements_num])
    W_affine = tf.Variable(tf.random_normal([elements_num,10],dtype=tf.float32),name='W_affine')
    b_affine = tf.Variable(tf.random_normal([10],dtype=tf.float32) , name='b_affine')
    model = tf.matmul(X_affine,W_affine)+b_affine

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#train
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(70):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.8})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch+1), 'Avg. cost =', '{:.3f}'.format(sess.run(tf.reduce_sum(total_cost))/total_batch))
print('최적화 완료!')
#결과 확인
is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도: ',sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))