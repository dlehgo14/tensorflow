import tensorflow as tf
import numpy as np

data = np.array([[0,0,1,0,0],[1,0,0,1,0],[1,1,0,0,1],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,0,1]])
data = np.transpose(data)
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.),name='W1')
    L1 = tf.nn.relu(tf.matmul(X,W1))
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1,W2))
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.), name='W3')
    model = tf.matmul(L2,W3)

global_step = tf.Variable(0, trainable = False, name='global_step')
with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost,global_step = global_step)

tf.summary.scalar('cost',cost)
sess= tf.Session()
saver = tf.train.Saver(tf.global_variables() , name='restore')

ckpt = tf.train.get_checkpoint_state('./modelLDH')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs',sess.graph)

feed_dict = {X: x_data, Y: y_data}
for step in range(2):
    sess.run(train_op, feed_dict=feed_dict)
    print('Step: %d, '%sess.run(global_step), 'cost: %.3f'%sess.run(cost,feed_dict))

summary = sess.run(merged,feed_dict)
writer.add_summary(summary, global_step=sess.run(global_step))

saver.save(sess, './modelLDH/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)
print('예측값: ',sess.run(prediction,feed_dict))
print('실제값: ',sess.run(target, feed_dict))
is_correct = tf.equal(prediction,target)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도: ',sess.run(accuracy,feed_dict))

sess.close()