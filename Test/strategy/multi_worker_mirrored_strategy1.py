import tensorflow as tf
import os
import json
import numpy as np


# tf.debugging.set_log_device_placement(True)

# 设置全局变量，定义多个worker
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:12346"]
    },
    'task': {'type': 'worker', 'index': 0} # 这是第一个节点
})

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def main():
    # 设置GPU显存自适应
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # 多worker分布式训练
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    
    num_workers = 2
    per_worker_batch_size = 64
    # Here the batch size scales up by number of workers since 
    # `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
    # and now this becomes 128.
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_and_compile_cnn_model()
        
    # 将 `filepath` 参数替换为在文件系统中所有工作器都能访问的路径。
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='E:\MyFiles\git\AI\data\strategy')]
    # Keras' `model.fit()` trains the model with specified number of epochs and
    # number of steps per epoch. Note that the numbers here are for demonstration
    # purposes only and may not sufficiently produce a model with good quality.
    multi_worker_model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=60000//global_batch_size,
        callbacks=callbacks)


if __name__ == '__main__':
    main()
