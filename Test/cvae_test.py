import tensorflow as tf
import tensorflow_datasets as tfds

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

# 加载数据集
# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
(train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
print('train_images', train_images.shape)

# 数据预处理
train_images = train_images.reshape(
    train_images.shape[0], 32, 32, 3).astype('float32')
test_images = test_images.reshape(
    test_images.shape[0], 32, 32, 3).astype('float32')

# 标准化图片到区间 [0., 1.] 内
train_images /= 255.
test_images /= 255.

# 二值化
# train_images[train_images >= .5] = 1.
# train_images[train_images < .5] = 0.
# test_images[test_images >= .5] = 1.
# test_images[test_images < .5] = 0.

# 变量
TRAIN_BUF = 60000
BATCH_SIZE = 100

TEST_BUF = 10000

# 转换成Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

# CVAE模型


class CVAE(tf.keras.Model):
    '''自编码器，可生成随机图片'''
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        # 潜在特征
        self.latent_dim = latent_dim
        # 识别器
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        # 生成器
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        '''识别'''
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        '''编码，提取特征与噪声比例'''
        # tf.split：将(batch_size, latent_dim * 2)拆成2份(batch_size, latent_dim)
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        '''
        随机噪声
        mean：原图特征
        logvar：噪声比例
        '''
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        '''
        生成图片
        z：经过reparameterize，特征与噪声混合的值
        '''
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


# 梯度下降器
optimizer = tf.keras.optimizers.Adam(1e-4)

# 定义loss


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, x):
    '''
    调用模型，计算loss
    model：CVAE模型
    x：输入图片(batch_size, 28, 28, 1)
    '''

    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    # 计算交叉熵，多分类用sigmoid(结果和不为1)，单分类用softmax(结果和为1)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    '''
    训练，计算梯度更新权重
    model：CVAE模型
    x：输入图片(batch_size, 28, 28, 1)
    optimizer：梯度下降
    '''
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 200
# latent_dim = 50
latent_dim = 100
num_examples_to_generate = 16

# 保持随机向量恒定以进行生成（预测），以便更易于看到改进。
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

# 显示并保存图片


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout 最小化两个子图之间的重叠
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    # 训练
    for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
    end_time = time.time()

    if epoch % 10 == 0:
        # 计算测试集loss
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
        # # 识别
        # generate_and_save_images(
        #     model, epoch, random_vector_for_generation)
# 识别
generate_and_save_images(
    model, epoch, random_vector_for_generation)

# 显示图片
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


plt.imshow(display_image(epochs))
plt.axis('off')  # 显示图片
