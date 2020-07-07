import tensorflow as tf
import numpy as np

class CustomModel(tf.keras.Model):
    '''自定义模型'''

    def __init__(self):
        '''初始化模型层'''
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=1)
        self.fc3 = tf.keras.layers.Dense(units=1)

    def call(self, x):
        '''运算部分'''
        x = self.fc1(x)
        y1 = self.fc2(x)
        y2 = self.fc3(x)
        return y1, y2

class CustomLoss(tf.keras.losses.Loss):
    '''自定义Loss'''

    def __init__(
        self,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'CustomLoss'
        super(CustomLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.square(y_pred[0] - y_true[0]), axis=-1)+tf.math.reduce_mean(tf.math.square(y_pred[1] - y_true[1]), axis=-1)

def generator():
    while True:
        x = tf.random.uniform((3,3), dtype=tf.float32)
        y1 = tf.random.uniform((3,1), dtype=tf.float32)
        y2 = tf.random.uniform((3,1), dtype=tf.float32)
        yield x, (y1, y2)
dataset = tf.data.Dataset.from_generator(generator,(tf.float32, (tf.float32, tf.float32)),(tf.TensorShape([None,3]),(tf.TensorShape([None,1]),tf.TensorShape([None,1]))))

for x, y in dataset.take(2):
    print(x.shape, y[0].shape, y[1].shape)

model = CustomModel()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=CustomLoss())

model.fit(dataset, epochs=10, steps_per_epoch=100)


