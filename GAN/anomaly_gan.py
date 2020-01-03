import tensorflow as tf
import os
import sys
import time
import cv2 as cv
import numpy as np

# 根目录
ROOT_DIR = os.path.abspath("./")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.image_loader import ImageLoader

def MarginConv2D(inputs, filters, kernel_size, strides, apply_dropout=False):
    """卷积合并"""
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    # skip = x
    # x = tf.keras.layers.Conv2D(int(filters / 2), (3, 3), padding='same')(x)
    # x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    if apply_dropout:
        x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Concatenate()([x, skip])
    return x

def MarginConv2DTranspose(inputs, filters, kernel_size, strides):
    """卷积合并"""
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    # skip = x
    # x = tf.keras.layers.Conv2D(int(filters / 2), (3, 3), padding='same')(x)
    # x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.Concatenate()([x, skip])
    return x

def make_generator_model():
    """生成器，随机生成假图片"""
    inputs = tf.keras.layers.Input(shape=(100,))
    x = tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Reshape((7, 7, 256))(x)
    # assert x.shape == (None, 7, 7, 512) # Note: None is the batch size

    x = MarginConv2DTranspose(x, 256, (5, 5), strides=(1, 1))
    # assert x.shape == (None, 7, 7, 512)

    x = MarginConv2DTranspose(x, 256, (5, 5), strides=(2, 2))
    # assert x.shape == (None, 14, 14, 512)

    x = MarginConv2DTranspose(x, 256, (5, 5), strides=(2, 2))
    # assert x.shape == (None, 28, 28, 256)

    x = MarginConv2DTranspose(x, 128, (5, 5), strides=(2, 2))
    # assert x.shape == (None, 56, 56, 128)

    x = MarginConv2DTranspose(x, 64, (5, 5), strides=(2, 2))
    # assert x.shape == (None, 112, 112, 64)

    x = MarginConv2DTranspose(x, 32, (5, 5), strides=(2, 2))
    # assert x.shape == (None, 224, 224, 32)

    x = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=tf.keras.activations.tanh)(x)
    # assert x.shape == (None, 448, 448, 3)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_discriminator_model():
    """鉴别器，区分真假，0.假，1.真"""
    inputs = tf.keras.layers.Input(shape=(448,448,3))
    # (448,448,3)=>(224,224,32)
    x = MarginConv2D(inputs, 32, (5, 5), strides=(2, 2), apply_dropout=False)

    # (224,224,32)=>(112,112,64)
    x = MarginConv2D(x, 64, (5, 5), strides=(2, 2), apply_dropout=False)

    # (112,112,64)=>(56,56,128)
    x = MarginConv2D(x, 128, (5, 5), strides=(2, 2), apply_dropout=True)
    
    # (56,56,128)=>(28,28,256)
    x = MarginConv2D(x, 256, (5, 5), strides=(2, 2), apply_dropout=True)

    # (28,28,256)=>(14,14,512)
    x = MarginConv2D(x, 256, (5, 5), strides=(2, 2), apply_dropout=True)

    # (14,14,512)=>(7,7,512)
    x = MarginConv2D(x, 256, (5, 5), strides=(2, 2), apply_dropout=True)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# 生成器，输入(None, 100)
generator = make_generator_model()
# 鉴别器，输入(None, 448, 448, 3)
discriminator = make_discriminator_model()


# 损失函数，0-1交叉熵
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 生成器梯度下降
generator_optimizer = tf.keras.optimizers.Adam(1e-5)
# 鉴别器梯度下降
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def generator_loss(fake_output):
    """生成器损失"""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    """鉴别器损失"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 权重保存
checkpoint_dir = './data/anomaly_gan_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 加载权重
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 6
BATCH_SIZE = 1

# 随机种子，用于展示
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    # 输入随机噪点
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        # print(images, fake_output)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


#%%
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        gen_loss_sum = 0
        disc_loss_sum = 0
        for image_batch in dataset:
            input_data = tf.io.read_file(image_batch)
            input_data = tf.io.decode_image(input_data, channels=3, dtype=tf.float32)
            input_data = tf.image.resize(input_data, size=(448, 448))
            input_data = tf.divide(input_data, 255.0)
            input_data = tf.expand_dims(input_data, 0)
            gen_loss, disc_loss = train_step(input_data)
            gen_loss_sum += gen_loss
            disc_loss_sum += disc_loss

        gen_loss_sum = gen_loss_sum / len(dataset)
        disc_loss_sum = disc_loss_sum / len(dataset)

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec. gen_loss {} disc_loss {}'.format(epoch + 1, time.time()-start, gen_loss_sum, disc_loss_sum))

    # Generate after the final epoch
    generate_and_save_images(generator,
                            epochs,
                            seed)
                           
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all tf.keras.layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        image = predictions[i, :, :, :] * 127.5 + 127.5
        frame = cv.cvtColor(image.numpy().astype(np.uint8), cv.COLOR_RGB2BGR)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.imwrite('image_at_epoch_{:04d}.jpg'.format(epoch),frame)

image_loader = ImageLoader()
image_files =image_loader.load(u'E:\MyFiles\labels\Switch03')

# 训练
        
train(image_files, EPOCHS)

cv.destroyAllWindows()