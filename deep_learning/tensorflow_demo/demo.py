# Import MNIST data
import input_data
mnist = input_data.read_data_sets("/home/weizhuozhang/workspace/dataset/mnist", one_hot=True)

import tensorflow as tf

# Set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

# TF graph input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Add summary ops to collect data
w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y * tf.log(model))
    # Create a summary to monitor the cost function
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries
merged_summary_op = tf.merge_all_symmaries()

