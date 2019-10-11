import tensorflow as tf
import numpy as np
import os
import sys

# 根目录
ROOT_DIR = os.path.abspath("../")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.wider_face_loader import WiderFaceLoader

class Encoder(tf.keras.Model):
    """编码器"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature_conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid', name='feature_conv1',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='feature_conv2',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='feature_conv3',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='feature_conv4',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='feature_conv5',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)

    def call(self, input_data):
        x = self.feature_conv1(input_data)
        x = self.feature_conv2(x)
        x = self.feature_conv3(x)
        x = self.feature_conv4(x)
        x = self.feature_conv5(x)
        return x

class Decoder(tf.keras.Model):
    """解码器"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.feature_conv1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='feature_conv1',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='feature_conv2',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='feature_conv3',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv4 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='feature_conv4',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv5 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name='feature_conv5',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        self.feature_conv6 = tf.keras.layers.Conv2D(3, (1, 1), padding='same', name='feature_conv6',
                                                    kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.linear)

    def call(self, input_data):
        x = self.feature_conv1(input_data)
        x = self.feature_conv2(x)
        x = self.feature_conv3(x)
        x = self.feature_conv4(x)
        x = self.feature_conv5(x)
        x = self.feature_conv6(x)
        return x

        
class AutoEncoder():
    """自编码器"""

    def __init__(self):
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.build()

    def build(self):
        """创建模型"""
        self.input = tf.keras.Input(shape=(None,None,3),dtype=tf.float32)
        x = self.encoder(self.input)
        self.output = x = self.decoder(x)
        self.model = tf.keras.Model(inputs=self.input,outputs=self.output)
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(),loss=tf.keras.losses.MeanSquaredError())

    def generate_arrays_from_file(self, features, batch_size):
        cnt = 0
        X = []
        Y = []
        while 1:
            for line in range(len(features)):
                img = tf.io.read_file(features[line])
                img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
                img = tf.divide(img, 255.0)
                img = tf.image.resize(img, size=(512,512))
                x = img.numpy().tolist()
                # print("x:", x.shape)
                y = x
                # print("x:", line, features[line])
                # print("y:", line, labels[line])
                X.append(x)
                Y.append(y)
                cnt += 1
                if cnt == batch_size:
                    cnt = 0
                    X = np.array(X)
                    Y = np.array(Y)
                    yield X, Y
                    X = []
                    Y = []
    def train(self, features):
        self.model.fit_generator(self.generate_arrays_from_file(features, 3), steps_per_epoch=100, epochs=30)

def main():
    # 加载数据
    wider_face_loader = WiderFaceLoader()
    train_data, train_label = wider_face_loader.load(
        'E:/MyFiles/人脸检测素材/wider_face_split/wider_face_train_bbx_gt.txt', 'E:/MyFiles/人脸检测素材/WIDER_train/images')
    auto_encoder = AutoEncoder()
    auto_encoder.train(train_data)

if __name__ == '__main__':
    main()