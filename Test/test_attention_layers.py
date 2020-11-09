import tensorflow as tf
import numpy as np
import os
import sys

# 根目录
ROOT_DIR = os.path.abspath("./")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.image_loader import ImageLoader
from CNN.attention_conv import AttentionConv2D

def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.expand_dims(x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)
    image_loader = ImageLoader()
    image_loader.show_image_gray(x_train[0,:,:,0])

    # 卷积实现图片分类
    model = tf.keras.models.Sequential([
        AttentionConv2D(32, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        tf.keras.layers.AveragePooling2D((2,2)),
        AttentionConv2D(128, (3, 3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        tf.keras.layers.AveragePooling2D((2,2)),
        AttentionConv2D(10, (7, 7), padding='valid', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.softmax),
        tf.keras.layers.Flatten(),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                #   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                #   loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
