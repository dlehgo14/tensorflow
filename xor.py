import tensorflow as tf
tf.reset_default_graph()
x_data = [[1.,1.],[1.,0.],[0.,1.],[0.,0.]]
y_data = [[0.],[1.],[1.],[0.]]

X = tf.placeholder(tf.float32, [None,2],name='X')
Y = tf.placeholder(tf.float32, [None,1],name='Y')
with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_uniform([2,10],0,0.1), name='W1')
    b1 = tf.Variable(tf.random_uniform([10],0,0.1), name='b1')
    h1 = tf.sigmoid(tf.matmul(X,W1)+b1)
    
    w1_hist = tf.summary.histogram("weights1",W1)
    b1_hist = tf.summary.histogram("bias1",b1)
    layer1_hist = tf.summary.histogram('layer1',h1)
with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_uniform([10,1],0,0.1), name='W2')
    b2 = tf.Variable(tf.random_uniform([1],0,0.1), name='b2')
    model = tf.sigmoid(tf.add(tf.matmul(h1,W2),b2))
    
    w2_hist = tf.summary.histogram("weights2",W1)
    b2_hist = tf.summary.histogram("bias2",b1)
    layer2_hist = tf.summary.histogram('layer2',model)

#cost = tf.reduce_mean(tf.square(Y-model))
cost = -tf.reduce_mean(Y*tf.log(model) + (1-Y)*tf.log(1-model))
optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)
predicted = tf.cast((model>0.5),tf.float32)

cost_summ = tf.summary.scalar("cost",cost)

summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs')


sess = tf.Session()
writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())
for step in range(1000):
    m,s,_,_cost= sess.run([model,summary,optimizer,cost],feed_dict={X: x_data, Y: y_data})
    print(step,_cost)
    print(sess.run(model,{X:x_data,Y:y_data}))
    writer.add_summary(s,global_step=step)

print(sess.run(predicted,{X:x_data}))

#이상하게 sigmoid를 쓰면 되고, relu를 쓰면 안된다.