import tensorflow as tf

class MyLayer1(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(10, (3,3), padding='same', use_bias=False)
        super().build(input_shape)

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        return x

class MyLayer2(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def build(self, input_shape):
        self.conv1 = MyLayer1()
        super().build(input_shape)

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        return x

class MyModel1(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)

    def build(self, input_shape):
        self.conv1 = MyLayer1()
        self.conv2 = MyLayer2()
        super().build(input_shape)

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

m = MyModel1()
# m.compile(run_eagerly=False)
m.build((None, None, None, 3))
# m(tf.random.normal((2, 512, 512, 3)))
for v in m.trainable_variables:
    tf.print('v:', v.name, tf.shape(v))