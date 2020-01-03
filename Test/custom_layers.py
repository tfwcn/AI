import tensorflow as tf
import numpy as np
import os
import sys

# 根目录
ROOT_DIR = os.path.abspath("./")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.image_loader import ImageLoader

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, **args):
        super(MyDenseLayer, self).__init__(**args)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        num_inputs = input_shape[-1]
        self.kernel = self.add_weight('kernel',
                                      shape=[num_inputs,
                                             self.num_outputs],
                                      initializer=tf.keras.initializers.he_normal()
                                      )
        # self.bias = self.add_weight("bias",
        #                             shape=[self.num_outputs],
        #                             initializer=tf.keras.initializers.Zeros
        #                             )

    @tf.function
    def call(self, input):
        # tf.print('input', tf.shape(input))
        # num_batch = tf.shape(input)[0]
        # num_inputs = tf.shape(input)[-1]
        # (num_inputs, 10)
        # output = tf.zeros([num_batch, self.num_outputs], dtype=self.dtype)
        # for i in tf.range(num_inputs):
        #     reverse_index = i % self.num_outputs
        #     kernel_one = tf.matmul(tf.reshape(input[:,i],[num_batch, 1]), tf.reshape(self.kernel[1,:],[1, self.num_outputs]))
        #     kernel_one = tf.concat([kernel_one[:, reverse_index:],
        #                             kernel_one[:, :reverse_index]], axis=1)
        #     # print('kernel_one', kernel_one)
        #     output = tf.add(output, kernel_one)
        # print('output', output.shape)
        # tf.print('output', tf.shape(output))
        output = tf.matmul(input, self.kernel)
        # output = tf.add(output, tf.math.multiply(0.01, tf.math.reduce_sum(self.kernel)))
        return output
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def main():
    # layer = tf.keras.layers.Dense(10, input_shape=(None, 5), dtype=tf.float16)
    # layer = MyDenseLayer(12, dtype=tf.float16)
    # print('layer:', layer)
    # print('variables:', layer.variables)
    # print('dtype:', layer.dtype)
    # value = layer(tf.ones([2, 23]))
    # print('value:', value)
    # print('variables:', layer.variables)
    # print('kernel:', layer.kernel)
    # print('bias:', layer.bias)
    # print('trainable_variables', [
    #       var.name for var in layer.trainable_variables])
    # a = tf.constant([[1, 2, 3, 4, 5],[6, 2, 3, 4, 5]])
    # print('reverse',tf.reverse(a, [1]))
    # tf.keras.backend.set_floatx('float16')

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.expand_dims(x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    # y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)
    image_loader = ImageLoader()
    image_loader.show_image_gray(x_train[0,:,:,0])

    # 全连接实现图片分类
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     MyDenseLayer(128),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation(activation=tf.keras.activations.relu),
    #     # tf.keras.layers.Dropout(0.2),
    #     MyDenseLayer(10),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation(activation=tf.keras.activations.softmax),
    # ])

    # 卷积实现图片分类
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(10, (7, 7), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.softmax),
        tf.keras.layers.Flatten(),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                #   loss=tf.keras.losses.CategoricalCrossentropy(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                #   loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
