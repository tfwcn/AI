# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import sys
import numpy as np
import time

print("TensorFlow version: {}".format(tf.version.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

# 根目录
ROOT_DIR = os.path.abspath("./")

# GuPiaoLoader
sys.path.append(ROOT_DIR)
from DataLoader.gupiao_loader import GuPiaoLoader

class Encoder(tf.keras.Model):
    '''编码器'''

    def __init__(self, gru1_units, gru2_units):
        super(Encoder, self).__init__()

        # GRU
        self.gru1 = tf.keras.layers.GRU(gru1_units, return_sequences=True, return_state=True, activation=tf.keras.activations.relu, name='feature_gru1')
        self.gru2 = tf.keras.layers.GRU(gru2_units, return_state=True, activation=tf.keras.activations.relu, name='feature_gru2')

    def call(self, input_data):
        '''
        input_data:批量已知数据(None,history_size,15)
        '''
        x, gru_state1 = self.gru1(input_data)
        x, gru_state2 = self.gru2(x)
        return x, gru_state1, gru_state2


class BahdanauAttention(tf.keras.Model):
    '''注意力模块'''

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, name='feature_denseW1')
        self.W2 = tf.keras.layers.Dense(units, name='feature_denseW2')
        self.V = tf.keras.layers.Dense(1, name='feature_denseV')

    def call(self, query, values):
        '''
        query:状态(batch_size,hidden_size)
        values:编码器输出，记忆(batch_size,1,hidden_size)
        '''
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # 分数(batch_size, 1, 1)，通过上一状态与记忆计算分数，觉得保留多少记忆
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # 注意力权重，0-1范围(batch_size, 1, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        # 通过注意力权重决定保留多少记忆信息
        context_vector = attention_weights * values
        # 求和第二维度(batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    '''解码器'''

    def __init__(self, class_num, gru1_units, gru2_units):
        super(Decoder, self).__init__()

        # used for attention
        # 注意力模块
        self.attention1 = BahdanauAttention(gru1_units)
        self.attention2 = BahdanauAttention(gru2_units)
        # GRU
        self.gru1 = tf.keras.layers.GRU(gru1_units, return_sequences=True, return_state=True, activation=tf.keras.activations.relu, name='feature_gru1')
        self.gru2 = tf.keras.layers.GRU(gru2_units, return_state=True, activation=tf.keras.activations.relu, name='feature_gru2')
        # 输出
        self.dense1 = tf.keras.layers.Dense(class_num, name='feature_dense1')

    def call(self, input_data, gru_state1, gru_state2, encoder_output):
        '''
        input_data:单步预测数据(None,1,15)
        gru_state1:上一步的状态(None,128)
        gru_state2:上一步的状态(None,128)
        encoder_output:编码器最后状态，已知数据提取的特征(None,128)
        '''
        context_vector1, _ = self.attention1(
            gru_state1, encoder_output)
        context_vector2, _ = self.attention2(
            gru_state2, encoder_output)
        # print('context_vector', context_vector1.shape, context_vector2.shape)
        # tf.print('context_vector', tf.shape(context_vector1), tf.shape(context_vector2))
        x = tf.concat([context_vector1, context_vector2, input_data], axis=-1)
        x = tf.expand_dims(x, 1)
        x, gru_state1 = self.gru1(x, initial_state=gru_state1)
        x, gru_state2 = self.gru2(x, initial_state=gru_state2)
        x = self.dense1(x)

        return x, gru_state1, gru_state2


class GuPiaoModel():
    '''股票预测模型'''

    def __init__(self, output_num, model_path='./data/gupiao_model', is_load_model=True):
        # 预测数据维度
        self.output_num = output_num
        # 加载模型路径
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path
        # 建立模型
        self.build_model()
        # 加载模型
        if is_load_model:
            self.load_model()
    
    def build_model(self):
        '''建立模型'''
        '''@nni.variable(nni.choice(16 ,32, 64, 128, 256), name=gru1_units)'''
        gru1_units = 64
        '''@nni.variable(nni.choice(16 ,32, 64, 128, 256), name=gru2_units)'''
        gru2_units = 128
        self.encoder_model = Encoder(gru1_units, gru2_units)
        self.decoder_model = Decoder(self.output_num, gru1_units, gru2_units)
        # 优化器
        '''@nni.variable(nni.uniform(0.05, 0.0001), name=learning_rate)'''
        learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0, lr=learning_rate)
        # 损失函数
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
        # 保存模型
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoder_model,
                                        decoder=self.decoder_model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.model_path, max_to_keep=3)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None, 15), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 15), dtype=tf.float32),
    ))
    def train_step(self, input_data, target_data):
        '''
        训练
        input_data:(batch_size, history_size, 15) 
        target_data:(batch_size, target_size, 15) 
        '''
        print('Tracing with train_step', type(input_data), type(target_data))
        print('Tracing with train_step', input_data.shape, target_data.shape)
        loss = 0.0
        with tf.GradientTape() as tape:
            # 编码
            # encoder_output(history_size,128)
            # encoder_state1(history_size,128)
            # encoder_state2(history_size,128)
            encoder_output, encoder_state1, encoder_state2 = self.encoder_model(input_data)
            # print('Tracing with encoder_output', encoder_output, encoder_state1, encoder_state2)
            # tf.print('Action with encoder_output', tf.shape(encoder_output), tf.shape(encoder_state1), tf.shape(encoder_state2))
            decoder_state1 = encoder_state1
            decoder_state2 = encoder_state2
            decoder_input = input_data[:,-1,:]

            # 解码
            for target_index in tf.range(tf.shape(target_data)[1]):
                # 正确值
                true_target = target_data[:,target_index,3:]
                # tf.print('Action with true_target', tf.shape(true_target))
                # 解码
                decoder_output, decoder_state1, decoder_state2 = self.decoder_model(
                    decoder_input, decoder_state1, decoder_state2, encoder_output)
                # tf.print('Action with decoder_output', tf.shape(decoder_output), tf.shape(decoder_state1), tf.shape(decoder_state2))
                # 计算损失
                batch_loss = self.loss_object(y_true=true_target, y_pred=decoder_output)
                # tf.print('Action with batch_loss', batch_loss)
                loss += batch_loss
                decoder_input = target_data[:,target_index,:]

        total_loss = (loss / float(tf.shape(target_data)[1]))
        trainable_variables  = self.encoder_model.trainable_variables + self.decoder_model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables )
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def fit_generator(self, generator, steps_per_epoch, epochs, initial_epoch=1, auto_save=False):
        '''训练'''
        for epoch in range(initial_epoch, epochs+1):
            start = time.process_time()
            epoch_loss = 0
            for steps in range(1, steps_per_epoch+1):
                x, y = next(generator)
                # print('generator', x.shape, y.shape)
                loss, total_loss = self.train_step(x, y)
                epoch_loss += total_loss
                print('\rsteps:%d/%d, epochs:%d/%d, loss:%0.4f, total_loss:%0.4f' 
                    % (steps, steps_per_epoch, epoch, epochs, loss, total_loss), end='')
            end = time.process_time()
            print('\rsteps:%d/%d, epochs:%d/%d, %0.4f S, loss:%0.4f, total_loss:%0.4f, epoch_loss:%0.4f' 
                % (steps, steps_per_epoch, epoch, epochs, (end - start), loss, total_loss, epoch_loss))
            """@nni.report_intermediate_result(epoch_loss)"""
            if auto_save:
                self.save_model()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None, 15), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
    ))
    def predict_jit(self, input_data, time_step, output_size):
        '''
        预测(编译模式)
        input_data:(1, history_size,15)
        time_step:预测时间序列，(1,target_size,3)
        output_size:预测数量
        '''
        # print('Tracing with predict', type(input_data), type(output_size))
        # print('Tracing with predict', input_data.shape, output_size)
        predict_data = tf.TensorArray(dtype=tf.float32, size=output_size, dynamic_size=True)
        # 编码
        # encoder_output(history_size,128)
        # encoder_state1(history_size,128)
        # encoder_state2(history_size,128)
        encoder_output, encoder_state1, encoder_state2 = self.encoder_model(input_data)
        # print('Tracing with encoder_output', encoder_output, encoder_state1, encoder_state2)
        # tf.print('Action with encoder_output', tf.shape(encoder_output), tf.shape(encoder_state1), tf.shape(encoder_state2))
        decoder_state1 = encoder_state1
        decoder_state2 = encoder_state2
        decoder_input = input_data[:,-1,:]

        # 解码
        for i in tf.range(output_size):
            # 解码
            decoder_output, decoder_state1, decoder_state2 = self.decoder_model(
                decoder_input, decoder_state1, decoder_state2, encoder_output)
            # tf.print('Action with decoder_output', tf.shape(decoder_output), tf.shape(decoder_state1), tf.shape(decoder_state2))
            decoder_input = tf.concat([time_step[:,i,:], decoder_output], axis=1)
            # 记录预测值
            predict_data = predict_data.write(i, decoder_input)

        # 交换维度
        predict_data = predict_data.stack()
        predict_data = tf.transpose(predict_data, perm=[1, 0, 2])
        return predict_data
        
    def predict_eager(self, input_data, time_step, output_size):
        '''
        预测(即时模式)))
        input_data:(1, history_size,15)
        time_step:预测时间序列，(1,target_size,3)
        output_size:预测数量
        '''
        # print('Tracing with predict', type(input_data), type(output_size))
        # print('Tracing with predict', input_data.shape, output_size)
        predict_data = []
        input_data = tf.constant(input_data, dtype=tf.float32)
        # 编码
        # encoder_output(history_size,128)
        # encoder_state1(history_size,128)
        # encoder_state2(history_size,128)
        encoder_output, encoder_state1, encoder_state2 = self.encoder_model(input_data)
        # print('Tracing with encoder_output', encoder_output, encoder_state1, encoder_state2)
        # tf.print('Action with encoder_output', tf.shape(encoder_output), tf.shape(encoder_state1), tf.shape(encoder_state2))
        decoder_state1 = encoder_state1
        decoder_state2 = encoder_state2
        decoder_input = input_data[:,-1,:]

        # 解码
        for i in range(output_size):
            # 解码
            decoder_output, decoder_state1, decoder_state2 = self.decoder_model(
                decoder_input, decoder_state1, decoder_state2, encoder_output)
            decoder_input = tf.concat([time_step[:,i,:], decoder_output], axis=1)
            # 记录预测值
            predict_data.append(decoder_input.numpy())

        predict_data = np.array(predict_data)
        # 交换维度
        predict_data = predict_data.swapaxes(0,1)
        return predict_data
        
    def save_model(self):
        '''保存模型'''
        save_path = self.checkpoint_manager.save()
        print('保存模型 {}'.format(save_path))
        
    def load_model(self):
        '''加载模型'''
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('加载模型 {}'.format(self.checkpoint_manager.latest_checkpoint))


def main():
    # 输入数据维度,(年 月 日+(开盘 最高 最低 收盘)*3)
    input_num = 15
    # 预测数据维度
    output_num = 12
    """@nni.variable(nni.choice(1, 2, 5, 10, 20, 50, 100), name=batch_size)"""
    batch_size = 20
    """@nni.variable(nni.choice(5, 10, 20, 30, 50), name=history_size)"""
    history_size = 30
    """@nni.variable(nni.choice(5, 10, 20), name=target_size)"""
    target_size = 5
    # 创建模型
    print('创建模型')
    gupiao_model = GuPiaoModel(output_num, is_load_model=False)
    # 加载数据
    print('加载数据')
    gupiao_loader = GuPiaoLoader()
    df_sh_all = gupiao_loader.load_one('./data/gupiao_data/999999.SH.csv')
    df_sz_all = gupiao_loader.load_one('./data/gupiao_data/399001.SZ.csv')
    df_target_all = gupiao_loader.load_one('./data/gupiao_data/603106.SH.csv')
    print('df_target_all.shape', df_target_all.shape)
    # 预测数量
    predict_num = 20
    df_sh = df_sh_all.iloc[:-predict_num,:]
    df_sz = df_sz_all.iloc[:-predict_num,:]
    df_target = df_target_all.iloc[:-predict_num,:]
    x, y = gupiao_loader.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
    print(x.shape,y.shape)
    x, y = gupiao_loader.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
    print(x.shape,y.shape)
    x, y = gupiao_loader.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
    print(x.shape,y.shape)
    print('训练前预测')
    x, time_step = gupiao_loader.get_data_to_predict(df_sh, df_sz, df_target, history_size, predict_num)
    print('x', x.shape, 'time_step', time_step.shape)
    y = gupiao_model.predict_jit(x, time_step, predict_num)
    print('y', y.shape)
    # 显示预测值
    _, y2 = gupiao_loader.get_data_to_train(df_sh_all, df_sz_all, df_target_all, 1, history_size, predict_num, len(df_target_all)-predict_num)
    print('y2', y2.shape)
    # gupiao_loader.show_image(x[0,:,:], y[0,:,:], y2[0,:,:])
    # 开始训练
    print('开始训练')
    """@nni.variable(nni.choice(10), name=epochs)"""
    epochs=20
    gupiao_model.fit_generator(
        gupiao_loader.data_generator(df_sh, df_sz, df_target, batch_size, history_size, target_size),
        steps_per_epoch=int(len(df_target)/2),
        epochs=epochs, auto_save=False)
    # 预测
    print('预测')
    y = gupiao_model.predict_jit(x, time_step, predict_num)
    # 显示预测值
    # gupiao_loader.show_image(x[0,:,:], y[0,:,:], y2[0,:,:])
    # 损失
    loss = gupiao_model.loss_object(y_true=y2[0,:,:], y_pred=y[0,:,:])
    """@nni.report_final_result(loss)"""
        

if __name__ == '__main__':
    main()