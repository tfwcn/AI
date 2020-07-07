import tensorflow as tf
from disout2_tf2 import Disout


class CustomLayer(tf.keras.layers.Layer):
    '''自定义层'''

    def __init__(self, units, activation, **args):
        super(CustomLayer, self).__init__(**args)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        '''初始化网络'''
        num_inputs = input_shape[-1]
        self.kernel = self.add_weight('kernel',
                                      shape=[num_inputs,
                                             self.units],
                                      initializer=tf.keras.initializers.he_normal()
                                      )
        self.bias = self.add_weight("bias",
                                    shape=[self.units],
                                    initializer=tf.keras.initializers.Zeros
                                    )
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    def call(self, x):
        '''运算部分'''
        print('call动态图会不断打印该信息，静态图只打印数次。', type(x), x.shape)
        output = tf.matmul(x, self.kernel) + self.bias
        output = self.activation_layer(output)
        return output

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return (input_shape[0], self.output_dim)


class CustomModel(tf.keras.Model):
    '''自定义模型'''

    def __init__(self, conv_size, hidden_size, dropout_rate):
        '''初始化模型层'''
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=conv_size, activation='relu')
        self.disout1 = Disout(0.09, block_size=5)
        # self.disout1.weight_behind=self.conv1.weights
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=conv_size, activation='relu')
        self.disout2 = Disout(0.09, block_size=3)
        # self.disout2.weight_behind=self.conv2.weights
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = CustomLayer(units=hidden_size, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.fc2 = CustomLayer(units=10, activation='softmax')

    def call(self, x):
        '''运算部分'''
        x = self.conv1(x)
        x = self.disout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.disout2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CustomCallback(tf.keras.callbacks.Callback):
    '''
    自定义回调
    具体事件参考：https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback?hl=en
    '''

    def __init__(
        self,
        **kwargs
    ):
        super(CustomCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        '''epoch结束时调用'''
        # 训练时的参数
        params = self.params
        # 模型，可中止训练
        model = self.model
        if 'val_acc' in logs:
            val_acc = logs['val_acc']
        else:
            val_acc = logs['val_accuracy']
        print('on_epoch_end')
        print('params:', params)
        print('model:', model)
        print('logs:', logs)


class CustomLoss(tf.keras.losses.Loss):
    '''自定义Loss'''

    def __init__(
        self,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'CustomLoss'
        super(CustomLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)


class CustomOptimizer(tf.keras.optimizers.Optimizer):
    '''自定义梯度下降器'''

    def __init__(
        self,
        learning_rate=0.1,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'CustomOptimizer'
        super(CustomOptimizer, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)

    def _create_slots(self, var_list):
        '''
        给变量创建关联变量，用于梯度计算
        var_list：可更新的变量列表
        '''
        tf.print('var_list:', type(var_list))
        # for var in var_list:
        #     self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        '''每层梯度更新的计算公式'''
        # 准备变量
        lr_t = self._get_hyper('learning_rate')
        # v = self.get_slot(var, 'v')
        # v = v.assign(v+0.4*grad)
        var_t = var.assign(var - lr_t * grad)
        return var_t

    def _resource_apply_dense(self, grad, var):
        '''每层梯度跟新都会调用该方法'''
        # tf.print('_resource_apply_dense')
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        '''每层梯度跟新都会调用该方法'''
        tf.print('_resource_apply_sparse')
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        tf.print('get_config')
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
        }
        base_config = super(CustomOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.expand_dims(
        x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)

    # 卷积实现图片分类
    model = CustomModel(5, 128, 0.65)

    model.compile(
        #   optimizer=CustomOptimizer(),
        optimizer=tf.keras.optimizers.Adam(),
        loss=CustomLoss(),
        metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
              callbacks=[CustomCallback()])


if __name__ == '__main__':
    # 设置GPU显存自适应
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
    main()
