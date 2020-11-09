import tensorflow as tf
import numpy as np
import os
import sys

# 打印设备调用日志
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
    # 加载数据
    mnist = tf.keras.datasets.mnist
    # 测试集和训练集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # (28, 28)=>(28, 28, 1)
    x_train, x_test = tf.expand_dims(x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)

    # 多GPU并行策略
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy'])
    # 训练
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
