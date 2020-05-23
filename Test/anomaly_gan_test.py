import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

# 根目录
ROOT_DIR = os.path.abspath("./")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.image_loader import ImageLoader


class MyEncoderModel(tf.keras.Model):
    def __init__(self, output_dim, **args):
        super(MyEncoderModel, self).__init__(**args)
        self.output_dim = output_dim

    def build(self, input_shape):
        print('MyEncoderModel input_shape', input_shape)
        self.layer1_conv = self.convLayer(32, (3, 3), name='layer1_conv')
        self.layer2_maxPool = tf.keras.layers.MaxPooling2D((2,2), name='layer2_maxPool')
        self.layer3_conv = self.convLayer(128, (3, 3), name='layer3_conv')
        self.layer4_maxPool = tf.keras.layers.MaxPooling2D((2,2), name='layer4_maxPool')
        self.layer5_conv = self.convLayer(self.output_dim, (8, 8), name='layer5_conv', padding='valid', activation=None)

    def convLayer(self, filters, kernel_size, name, padding='same', activation=tf.keras.activations.relu):
        result = [
            tf.keras.layers.Conv2D(filters, kernel_size, padding=padding,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   name=name+'_conv'),
            tf.keras.layers.BatchNormalization(name=name+'_bn'),
            tf.keras.layers.Activation(activation=activation, name=name+'_relu'),
        ]
        return result

    def conv(self, input, convLayers):
        result = input
        for layer in convLayers:
            # print('MyEncoderModel conv ', type(layer))
            result = layer(result)
        return result

    # @tf.function
    def call(self, input):
        output = self.conv(input, self.layer1_conv)
        output = self.layer2_maxPool(output)
        output = self.conv(output, self.layer3_conv)
        output = self.layer4_maxPool(output)
        output = self.conv(output, self.layer5_conv)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, self.output_dim)


class MyDecoderModel(tf.keras.Model):
    def __init__(self, encoder_input_shape, **args):
        super(MyDecoderModel, self).__init__(**args)
        self.encoder_input_shape = encoder_input_shape

    def build(self, input_shape):
        print('MyDecoderModel input_shape', input_shape)
        self.layer1_convTranspose = self.convLayer(128, (8, 8), name='layer1_convTranspose')
        self.layer11_convTranspose = self.convLayer(128, (3, 3), padding='same', name='layer11_convTranspose')
        self.layer2_convTranspose = self.convLayer(32, (3, 3), strides=(2, 2), padding='same', name='layer2_convTranspose')
        self.layer21_convTranspose = self.convLayer(32, (3, 3), padding='same', name='layer21_convTranspose')
        self.layer3_convTranspose = self.convLayer(3, (3, 3), strides=(2, 2), padding='same', name='layer3_convTranspose', activation=tf.keras.activations.relu)

    def convLayer(self, filters, kernel_size, strides=(1,1), name='', padding='valid', activation=tf.keras.activations.relu):
        result = [
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   name=name+'_conv'),
            tf.keras.layers.BatchNormalization(name=name+'_bn'),
            tf.keras.layers.Activation(activation=activation, name=name+'_relu'),
        ]
        return result

    def conv(self, input, convLayers):
        result = input
        for layer in convLayers:
            # print('MyDecoderModel conv ', type(layer))
            result = layer(result)
        return result

    # @tf.function
    def call(self, input):
        output = self.conv(input, self.layer1_convTranspose)
        output = self.conv(output, self.layer11_convTranspose)
        output = self.conv(output, self.layer2_convTranspose)
        output = self.conv(output, self.layer21_convTranspose)
        output = self.conv(output, self.layer3_convTranspose)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 28, 28, 1)


class MyGanModel():
    def __init__(self, **args):
        super(MyGanModel, self).__init__(**args)
        self.build((28, 28, 1))

    def build(self, input_shape):
        print('MyGamModel input_shape', input_shape)
        self.encoder = MyEncoderModel(20)
        self.decoder = MyDecoderModel(encoder_input_shape=input_shape)
        # self.loss_fun=tf.keras.losses.MeanSquaredError()
        self.loss_fun=tf.keras.losses.CategoricalCrossentropy()
        self.generator_optimizer=tf.keras.optimizers.RMSprop()

    def fit(self, train_data, epochs, steps, validation_data=None):
        print('初始化fit')
        tf.print('初始化fit')
        # train_data = train_data.batch(batch_size)
        tf.print('初始化train_data')
        for epoch_i in range(epochs):
            start = time.process_time()
            sum_loss = 0.0
            step_i = 0
            for step_i in range(steps):
                x, y = next(train_data)
                loss = self.train_step(x)
                sum_loss += loss
                # tf.print('\repocn:', epoch_i+1, '/', epochs, ',step:', step_i+1, '/', len(train_data), ',loss:', loss, end='')
                print('\repocn: %d/%d , step: %d/%d , loss: %.4f' % (epoch_i+1, epochs, step_i+1, steps, loss), end='')
            end = time.process_time()
            print('\repocn: %d/%d , step: %d/%d , loss: %.4f , %0.4f S' % (epoch_i+1, epochs, step_i+1, steps, sum_loss, (end - start)))
            # tf.print('\repocn:', epoch_i+1, '/', epochs, ',step:', step_i+1, '/', len(train_data), ',loss:', sum_loss)

    
    @tf.function
    def train_step(self, image):
        # input = tf.expand_dims(image, axis=0)
        input = image
        x = input

        loss = 0.0
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            x = self.encoder(x)
            x = self.decoder(x)
            loss = self.loss_fun(y_true=input,y_pred=x)
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients_of_generator = gen_tape.gradient(loss, trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, trainable_variables))

        return loss

    @tf.function
    def predict(self, image):
        input = tf.expand_dims(image, axis=0)
        x = self.encoder(input)
        x = self.decoder(x)
        return tf.squeeze(x, axis=0)

def data_generator(x, y, batch_size, shuffle=True):
    data = tf.data.Dataset.from_tensor_slices((x, y)).prefetch(tf.data.experimental.AUTOTUNE)
    if shuffle:
        for x, y in data.shuffle(60000).repeat().batch(batch_size):
            yield x, y
    else:
        for x, y in data.batch(batch_size):
            yield x, y

def main():
    # x = tf.zeros((1, 1, 1, 10))
    # x = tf.keras.layers.Conv2DTranspose(128 ,(7, 7))(x)
    # x = tf.keras.layers.Conv2DTranspose(32 ,(3, 3), (2, 2), padding='same')(x)
    # print('x', x.shape)
    # return

    # mnist = tf.keras.datasets.mnist
    mnist = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_train, x_test = tf.expand_dims(
    #     x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    x_train, x_test = tf.cast(x_train, dtype=np.float32), tf.cast(x_test, dtype=np.float32)
    # y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)
    print('x_train, y_train', type(x_train))
    # image_loader = ImageLoader()
    # image_loader.show_image(x_train[0, :, :, :])
    
    # test(data_train, 10)
    # g = data_generator(x_train, y_train, batch_size=10)
    # x, y=next(g)
    # print('g', x)

    # return

    model = MyGanModel()

    model.fit(data_generator(x_train, y_train, batch_size=10), epochs=20, steps=6000, validation_data=data_generator(x_test, y_test, batch_size=10))

    for i in range(10):
        tmp_img = x_test[i]
        print('tmp_img', tmp_img.shape)
        predict_img = model.predict(tmp_img)
        print('predict_img', predict_img.shape)
        plt.figure("Image") # 图像窗口名称
        plt.subplot(1, 2, 1)
        # plt.imshow(tmp_img[:,:,0], cmap='gray')
        plt.imshow(tmp_img)
        plt.subplot(1, 2, 2)
        # plt.imshow(predict_img[:,:,0], cmap='gray')
        plt.imshow(predict_img)
        plt.axis('off') # 关掉坐标轴为 off
        # plt.title('image') # 图像题目
        plt.show()

if __name__ == '__main__':
    main()
