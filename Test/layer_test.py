import tensorflow as tf

x = tf.ones((1,416,416,1))
# x = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(x)
# print(x.shape)
x = tf.keras.layers.Conv2D(32,(3,3),strides=(2,2),padding='same')(x)
print(x.shape)