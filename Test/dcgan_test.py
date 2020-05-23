import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

# train_images = train_images.reshape(
#     train_images.shape[0], 28, 28, 1).astype('float32')
train_images = train_images.reshape(
    train_images.shape[0], 32, 32, 3).astype('float32')
# Normalize the images to [-1, 1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256
# BATCH_SIZE = 128

train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# def make_generator_model():
#     '''生成器'''
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((8, 8, 256)))
#     # Note: None is the batch size
#     assert model.output_shape == (None, 8, 8, 256)

#     model.add(layers.Conv2DTranspose(
#         128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     assert model.output_shape == (None, 8, 8, 128)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(
#         64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     assert model.output_shape == (None, 16, 16, 64)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
#                                      padding='same', use_bias=False, activation='tanh'))
#     assert model.output_shape == (None, 32, 32, 3)

#     return model


def make_generator_model():
    '''生成器'''
    inputs = tf.keras.Input((100,))
    x = tf.keras.layers.Dense(8*8*256, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Reshape((8, 8, 256))(x)
    # Note: None is the batch size
    assert tuple(x.shape) == (None, 8, 8, 256)
    x = tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    assert tuple(x.shape) == (None, 8, 8, 128)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    assert tuple(x.shape) == (None, 16, 16, 64)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh')(x)
    assert tuple(x.shape) == (None, 32, 32, 3)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model


generator = make_generator_model()

# 100个随机噪声
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

print('generated_image')
print(generated_image.shape)
print(np.max(generated_image), np.min(generated_image))

show_image = generated_image[0, :, :, :] * 127.5 + 127.5
show_image = tf.cast(show_image, dtype=np.int32)
# plt.imshow(generated_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
plt.imshow(show_image)
plt.show()

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    for i in range(num_blocks):
        y = tf.keras.layers.Conv2D(num_filters//2, (1, 1), padding='same', use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU()(y)
        x = Add()([x,y])
    return x

# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                             input_shape=[32, 32, 3]))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#     # model.add(layers.Dropout(0.3))

#     model.add(layers.Conv2D(128, (3, 3), padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#     # model.add(layers.Dropout(0.3))

#     model.add(layers.Flatten())
#     model.add(layers.Dense(1))

#     return model

def make_discriminator_model():
    inputs = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

# 识别器，识别真假
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print('识别真假')
print(decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_optimizer = tf.keras.optimizers.RMSprop()
discriminator_optimizer = tf.keras.optimizers.RMSprop()

checkpoint_dir = './data/dcgan_model/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# EPOCHS = 50
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16


# 我们将重复使用该种子（因此在动画 GIF 中更容易可视化进度）
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 注意 `tf.function` 的使用
# 该注解使函数被“编译”
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        batch_num = 0
        for image_batch in dataset:
            print('\rbatch:', batch_num, end='')
            batch_num += 1
            train_step(image_batch)

        # 继续进行时为 GIF 生成图像
        # display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # 每 15 个 epoch 保存一次模型
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))

    # 最后一个 epoch 结束后生成图片
    # display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


def generate_and_save_images(model, epoch, test_input):
    # 注意 training` 设定为 False
    # 因此，所有层都在推理模式下运行（batchnorm）。
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        show_image = predictions[i, :, :, :] * 127.5 + 127.5
        show_image = tf.cast(show_image, dtype=np.int32)
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.imshow(show_image)
        plt.axis('off')

    plt.savefig('./data/dcgan_model/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    
train(train_dataset, EPOCHS)