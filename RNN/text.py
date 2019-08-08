import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self,units):
        super(MyModel, self).__init__(self)
        self.dense=tf.keras.layers.Dense(units,activation=None)

    def call(self, input_data):
        output = self.dense(input_data)
        return output

print(tf.__version__)
my_model=MyModel(1)
losses = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        input_data = np.array([[1.,1.]])
        # print('input_data',input_data.shape)
        prediction = my_model(input_data)
        print('prediction1', prediction.numpy())
        loss = losses(prediction, [[2.]])
    gradients = tape.gradient(loss, my_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))

prediction = my_model(input_data)
print('prediction2', prediction.numpy())

