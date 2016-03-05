import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf 

f_train = '../inputs/train.csv'
f_test = '../inputs/test.csv'
f_out = '../submissions/submission_1.csv'

training = pd.read_csv(f_train, index_col=0)
test = pd.read_csv(f_test, index_col=0)

print("Training shape: " + str(training.shape))
print("Test shape: " + str(test.shape))

X_train = training.iloc[:,:-1]
y_train = training.TARGET

num_cols = X_train.shape[1]

# Find out the types of values in each of the columns
print("dtype vals: " + X_train.dtypes.value_counts())


assert (num_cols == 369)
###########################################
# Let's build a tensorflow graph
###########################################
x = tf.placeholder(tf.float32, [None, 369])

W = tf.Variable(tf.zeros([369, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# y = predicted value (y-hat), y_ = true value
y_ = tf.placeholder(tf.float32, [None, 1])

# cross entropy = -\sum(y * log(y-hat))
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = m
