import tensorflow as tf
import numpy as np


@tf.function
def test(x):
    def my_numpy_func(x):
        # x will be a numpy array with the contents of the input to the
        # tf.function
        # print(x)
        x[:, 1] = 0
        return x
    x = tf.numpy_function(my_numpy_func, [x], tf.float32)
    if len(tf.shape(x)) == 2:
        tf.print('len(tf.shape(x)) == 2')
    tf.print(tf.math.reduce_prod(tf.shape(x)))
    tf.print(len(x))
    tf.print(x)


# x = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
# test(x)
x = tf.constant([0,0,0,1,0])
x = tf.reshape(x,(1,5,1))
print(tf.nn.max_pool1d(x, ksize=3, strides=1, padding='SAME'))