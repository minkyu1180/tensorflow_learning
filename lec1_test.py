import tensorflow as tf
#tensorflow is defined as nodes and its relatives.
hello = tf.constant("Hello, Tensorflow!") #constant node
sess = tf.Session() #Session node
#Like this...

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
#print("node3:", node3) prints about just node 3. To activate this, we should run.

sess = tf.Session()
print("sess.run(node1, node2):", sess.run([node1, node2]) )
print("sees.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, feed_dict ={a: 3, b: [4.5, 5]}))
# we can put the values using placeholder.
