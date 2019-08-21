import tensorflow as tf
import numpy as np
import time
import shutil
import os

class MyModel(tf.keras.Model):

    def __init__(self,units):
        super(MyModel, self).__init__(self)
        self.dense=tf.keras.layers.Dense(units,activation=None)

    def call(self, input_data):
        # print('input_data',input_data)
        output = self.dense(input_data)
        return output

print(tf.__version__)
# 定义模型
my_model1=MyModel(3)
my_model2=MyModel(1)
losses = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1)

# 使用 @tf.function 标识，进行JIT编译，执行效率高
# 去掉 @tf.function 标识为即刻模式，可用于调试，执行效率较低
@tf.function
def train(input_data,target_data):
    with tf.GradientTape() as tape:
        # print('input_data',input_data.shape)
        x = my_model1(input_data)
        output_data = my_model2(x)
        # tf.print('prediction1', prediction)
        loss = losses(output_data, target_data)
        # 记录日志,会影响效率
        tf.summary.scalar('loss', loss, step=optimizer.iterations)
    variables = my_model1.trainable_variables + my_model2.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

@tf.function
def prediction(input_data):
    # print('input_data',input_data.shape)
    x = my_model1(input_data)
    output_data = my_model2(x)
    return output_data

# 记录日志,会影响效率
if os.path.exists('./tmp/summaries'):
    shutil.rmtree('./tmp/summaries')
summary_writer = tf.summary.create_file_writer('./tmp/summaries')
# 打印执行时间
start = time.process_time()
# 设置日志默认保存路径
# summary_writer.as_default()
for i in range(500):
    input_data = np.array([[1.,1.]])
    target_data = np.array([[2.]])
    train(input_data,target_data)
elapsed = (time.process_time() - start)
print("运行时间:",elapsed)
# 保存模型权重
my_model1.save_weights('./tmp/save_models1.h5')
my_model2.save_weights('./tmp/save_models2.h5')
# 加载模型权重
my_model1.load_weights('./tmp/save_models1.h5')
my_model2.load_weights('./tmp/save_models2.h5')
print('识别')
input_data = np.array([[1.,1.]])
output_data = prediction(input_data)
print('prediction1', output_data)
# 下面不在 @tf.function 标识的方法内，执行为即刻模式
x = my_model1(input_data)
output_data = my_model2(x)
print('prediction2', output_data.numpy())

