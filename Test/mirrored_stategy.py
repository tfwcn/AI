import tensorflow as tf
import numpy as np
import os
import sys

# 根目录
# ROOT_DIR = os.path.abspath("./")

# 导入WiderFaceLoader
# sys.path.append(ROOT_DIR)
# from DataLoader.image_loader import ImageLoader

tf.debugging.set_log_device_placement(True)

# 获取所有的物理GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus))
# 设置GPU可见，由于只有1个GPU，因此选择gpus[0]，一般一个物理GPU对应一个逻辑GPU
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# 将物理GPU划分逻辑分区，这里将gpus[0]分为1个逻辑GPU，内存分别是2048，程序运行时占用内存 < 2048
tf.config.experimental.set_virtual_device_configuration(gpus[0], \
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024), \
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
# 获取所有的逻辑GPU
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))

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
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)
    # image_loader = ImageLoader()
    # image_loader.show_image_gray(x_train[0,:,:,0])

    # 多GPU并行策略
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
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

        def loss_fun(y_true, y_pred):
            return tf.math.abs(y_true-y_pred)

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    #   loss=tf.keras.losses.CategoricalCrossentropy(),
                    #   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    #   loss=tf.keras.losses.MeanSquaredError(),
                    loss=loss_fun,
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
