import tensorflow as tf
import numpy as np
import os
import sys
import time

# 根目录
ROOT_DIR = os.path.abspath("./")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.image_loader import ImageLoader

# 设置GPU显存自适应
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')

class AdaX(tf.keras.optimizers.Optimizer):
    r"""Implements AdaX algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 1e-4))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-12)
        weight_decay (float, optional): L2 penalty (default: 5e-4)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.0001,
        epsilon=1e-6,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'AdaX_V2'
        super(AdaX, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon
        print('self._initial_decay:', self._initial_decay)

    def _create_slots(self, var_list):
        '''
        给变量创建关联变量，用于梯度计算
        var_list：可更新的变量列表
        '''
        tf.print('var_list:', type(var_list))
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        '''每层梯度更新的计算公式'''
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)

        # 更新公式
        if indices is None:
            m_t = m.assign(beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = v.assign((1 + beta_2_t) * v + beta_2_t * grad**2)
        else:
            mv_ops = [
                m.assign(beta_1_t * m),
                v.assign((1 + beta_2_t) * v)
            ]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(
                    m, indices, (1 - beta_1_t) * grad
                )
                v_t = self._resource_scatter_add(
                    v, indices, beta_2_t * grad**2)

        # 返回算子
        # tf.control_dependencies先执行前置操作，后执行内部代码
        with tf.control_dependencies([m_t, v_t]):
            v_t = v_t / (tf.pow(1.0 + beta_2_t, local_step) - 1.0)
            var_t = var.assign(var - lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
            return var_t

    def _resource_apply_dense(self, grad, var):
        '''每层梯度跟新都会调用该方法'''
        # tf.print('_resource_apply_dense')
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        '''每层梯度跟新都会调用该方法'''
        # tf.print('_resource_apply_sparse')
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        tf.print('get_config')
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
        }
        base_config = super(AdaX, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(
        self,
        learning_rate=0.01,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'MyOptimizer'
        super(MyOptimizer, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)

    def _create_slots(self, var_list):
        '''
        给变量创建关联变量，用于梯度计算
        var_list：可更新的变量列表
        '''
        tf.print('var_list:', type(var_list))
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        '''每层梯度更新的计算公式'''
        # tf.print('grad:', type(grad), tf.shape(grad))
        # tf.print('var:', type(var), tf.shape(var))
        # 准备变量
        lr_t = self._get_hyper('learning_rate')
        v = self.get_slot(var, 'v')
        v = v.assign(v+0.4*grad)
        var_t = var.assign(var - lr_t * v)
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
        base_config = super(MyOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.expand_dims(x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)
    # image_loader = ImageLoader()
    # image_loader.show_image_gray(x_train[0,:,:,0])

    # 卷积实现图片分类
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(10, (7, 7), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.keras.activations.softmax),
        tf.keras.layers.Flatten(),
    ])

    optimizer = AdaX()
    loss_fun = tf.keras.losses.MAE()

    # @tf.function
    # def loss_fun(y_true, y_pred):
    #     return tf.reduce_sum(tf.math.abs(y_true-y_pred))

    @tf.function
    def TrainStep(x, y):
        '''
        单步训练
        input_image:图片(416,416,3)
        target_data:y15+透视变换2位移、2旋转、1缩放(5个值)
        '''
        print('Tracing with TrainStep', type(x), type(y))
        print('Tracing with TrainStep', x.shape, y.shape)
        tmp_loss = 0.0
        with tf.GradientTape() as tape:
            # 预测
            output = model(x)
            # 计算损失
            tmp_loss = loss_fun(
                y_true=y, y_pred=output)
        trainable_variables = model.trainable_variables
        gradients = tape.gradient(tmp_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return tmp_loss

    # @tf.function
    def FitGenerator(x_train, y_train, epochs, validation_data=None):
        '''批量训练'''
        for epoch in range(1, epochs+1):
            start = time.process_time()
            epoch_loss = 0
            steps_per_epoch = len(x_train)
            steps = 0
            for steps in range(0, steps_per_epoch):
                x, y = x_train[steps:steps+1], y_train[steps:steps+1]
                # print('generator', x.shape, y.shape)
                tmp_g_loss = TrainStep(x, y)
                epoch_loss += tmp_g_loss
                # print('tmp_g_loss:',tmp_g_loss)
                # print('epoch_loss:',epoch_loss)
                if steps % 100 == 0:
                    tf.print('\rsteps:', steps+1, '/', steps_per_epoch,', epochs:',epoch,'/',epochs, 'epoch_loss_avg:',  epoch_loss / (steps+1), end='')
                # if tf.math.is_nan(tmp_g_loss):
                #     return
            val_true = 0
            if validation_data is not None:
                for steps in range(0, len(validation_data[0])):
                    x, y = validation_data[0][steps:steps+1], validation_data[1][steps:steps+1]
                    # 预测
                    output = model(x)
                    output = tf.argmax(output, axis=-1)
                    # tf.print('output:', output)
                    # tf.print('y:', y)
                    if output[0] == tf.argmax(y, axis=-1)[0]:
                        val_true += 1
            # 求平均
            # epoch_loss = epoch_loss / steps_per_epoch
            end = time.process_time()
            tf.print('\rsteps:',steps_per_epoch,'/',steps_per_epoch,', epochs:',epoch,'/',epochs,', ',(end - start),' S, epoch_loss_avg:',epoch_loss / len(x_train),'val:',val_true / len(validation_data[0]))
            

    # model.compile(optimizer=AdaX(),
    #               loss=loss_fun,
    #               metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    FitGenerator(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
