import tensorflow as tf
from typing import Tuple, Union

class AttentionConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters:int, 
                 kernel_size:Union[int,Tuple[int,int]], 
                 padding:str='same', 
                 use_bias:bool=False, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.use_bias=use_bias

    def build(self, input_shape):
        self.W1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same')
        self.V1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same')
        self.W2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same')
        self.V2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same')
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding, use_bias=self.use_bias)

    def call(self, x):
        # 平面注意力
        o1=self.W1(x)
        o1=tf.math.tanh(o1)
        o1=self.V1(o1)
        o1=tf.math.exp(o1) / tf.math.reduce_sum(tf.math.exp(o1), axis=[1,2], keepdims=True)
        # 通道注意力
        o2=self.W2(x)
        o2=tf.math.tanh(o2)
        o2=self.V2(o2)
        o2=tf.nn.softmax(o2,axis=-1)
        # 输入保留率
        o=self.conv1(x)
        o=self.bn1(o)
        o=tf.nn.swish(o)
        o=o*o1+o*o2
        o=self.conv2(o)
        return o

