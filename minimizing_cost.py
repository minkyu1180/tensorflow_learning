import tensorflow as tf
x_train = tf.placeholder(tf.float32, shape = [None])
# 변수 담아드려요~~
y_train = tf.placeholder(tf.float32, shape = [None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = W*x_train + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)
#train이란 노드 밑 코스트 밑 가설 밑 W, b.... 쭉쭉 노드 주렁주렁 트리

sess = tf.Session() # 괄호 주의!
sess.run(tf.global_variables_initializer())

for step in range (2001):
    cost_val, W_val, b_val, _ =\
        sess.run([cost,W,b,train],feed_dict = {x_train :[1,2,3,4,5], y_train :[1.1,2.2,3.3,4.4,5.5] })
# 이런 식으로 한꺼번에 할당가능
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
