import tensorflow as tf
import tensorflow_datasets as tfds

mnist_train = tfds.load(name="lfw", split="train", data_dir='.\\data\\tfds')
print(mnist_train)
