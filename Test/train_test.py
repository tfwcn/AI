import tensorflow as tf
import numpy as np

inputs = tf.keras.Input((None, 4))
x = tf.keras.layers.Dense(10, activation=tf.keras.activations.relu)(inputs)
x = tf.keras.layers.Dense(2)(x)

model = tf.keras.Model(inputs=inputs, outputs=x)
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.RMSprop()

x_train = np.ones((1, 4), dtype=np.float32)
y_train = np.ones((1, 2), dtype=np.float32)
loss_val = 0
with tf.GradientTape() as tape:
    y = model(x_train)
    loss_val = loss(y_true=y_train,y_pred=y)

print('loss', loss_val)
gradients = tape.gradient(loss_val, model.trainable_variables)

optimizer.apply_gradients(zip(gradients, model.trainable_variables))