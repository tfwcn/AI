import tensorflow as tf
import os
import sys
import numpy as np
import time
import os
import pandas
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

print("TensorFlow version: {}".format(tf.version.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

# 根目录
ROOT_DIR = os.path.abspath("./")

# GuPiaoLoader
sys.path.append(ROOT_DIR)

# 全局参数
model_input_num = 6 + 5
model_output_num = 6 + 5

class Encoder(tf.keras.Model):
    '''编码器'''

    def __init__(self):
        super(Encoder, self).__init__()

        # GRU
        self.gru1 = tf.keras.layers.GRU(128, return_sequences=True, return_state=True, activation=tf.keras.activations.relu, name='feature_gru1')
        self.gru2 = tf.keras.layers.GRU(128, return_state=True, activation=tf.keras.activations.relu, name='feature_gru2')

    def call(self, input_data):
        '''
        input_data:批量已知数据(None,history_size,model_input_num)
        '''
        x, gru_state1 = self.gru1(input_data)
        x, gru_state2 = self.gru2(x)
        return x, gru_state1, gru_state2


class BahdanauAttention(tf.keras.Model):
    '''记忆模块'''

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

    def __init__(self, class_num):
        super(Decoder, self).__init__()

        # used for attention
        # 注意力模块
        self.attention1 = BahdanauAttention(128)
        self.attention2 = BahdanauAttention(128)
        # GRU
        self.gru1 = tf.keras.layers.GRU(128, return_sequences=True, return_state=True, activation=tf.keras.activations.relu, name='feature_gru1')
        self.gru2 = tf.keras.layers.GRU(128, return_state=True, activation=tf.keras.activations.relu, name='feature_gru2')
        # 输出
        self.dense1 = tf.keras.layers.Dense(class_num, name='feature_dense1')

    def call(self, input_data, gru_state1, gru_state2, encoder_output):
        '''
        input_data:单步预测数据(None,1,input_num)
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

class DataPerdictModel():
    '''预测模型'''

    def __init__(self, input_num, output_num, model_path='./data/data_perdict_model'):
        # 输入数据维度
        self.input_num = input_num
        # 预测数据维度
        self.output_num = output_num
        # 加载模型路径
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path
        # 建立模型
        self.build_model()
        # 加载模型
        self.load_model()
    
    def build_model(self):
        '''建立模型'''
        self.encoder_model = Encoder()
        self.decoder_model = Decoder(self.output_num-6)
        # 优化器
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0, lr=0.001)
        # 损失函数
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
        # 保存模型
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoder_model,
                                        decoder=self.decoder_model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.model_path, max_to_keep=3)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None, model_input_num), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, model_output_num), dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ))
    def train_step(self, input_data, target_data, same_shape=False):
        '''
        训练
        input_data:(batch_size, history_size, input_num) 
        target_data:(batch_size, target_size, output_num) 
        same_shape:是否相同类数据预测
        '''
        # 如果输出多行，则图会不断重新构建，会导致异常
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
            if same_shape:
                decoder_input = input_data[:,-1,:]
            else:
                decoder_input = tf.concat([input_data[:,-1,:6], 
                                        tf.zeros(shape=(tf.shape(input_data)[0], self.output_num-6))],
                                        axis=1)

            # 解码
            for target_index in tf.range(tf.shape(target_data)[1]):
                # 正确值
                true_target = target_data[:,target_index,6:]
                # tf.print('Action with true_target', tf.shape(true_target))
                # 解码
                decoder_output, decoder_state1, decoder_state2 = self.decoder_model(
                    decoder_input, decoder_state1, decoder_state2, encoder_output)
                # tf.print('Action with decoder_output', tf.shape(decoder_output), 
                #          tf.shape(decoder_state1), tf.shape(decoder_state2))
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

    def fit_generator(self, generator, steps_per_epoch, epochs, initial_epoch=1, auto_save=False, same_shape=False):
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
            if auto_save:
                self.save_model()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None, model_input_num), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool),
    ))
    def predict_jit(self, input_data, time_step, output_size, same_shape=False):
        '''
        预测(JIT编译模式)
        input_data:(1, history_size, input_num)
        time_step:预测时间序列，(1,target_size,6)
        output_size:预测数量
        same_shape:是否相同类数据预测
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
        if same_shape:
            decoder_input = input_data[:,-1,:]
        else:
            decoder_input = tf.concat([input_data[:,-1,:6], 
                                      tf.zeros(shape=(tf.shape(input_data)[0], self.output_num-6))],
                                      axis=1)
        # decoder_input = tf.concat([decoder_input[:,:10], 
        #                               tf.zeros(shape=(tf.shape(input_data)[0], 1))*0.5],
        #                               axis=1)
        # 解码
        for i in tf.range(output_size):
            # 解码
            decoder_output, decoder_state1, decoder_state2 = self.decoder_model(
                decoder_input, decoder_state1, decoder_state2, encoder_output)
            # tf.print('Action with decoder_output', tf.shape(decoder_output), tf.shape(decoder_state1), tf.shape(decoder_state2))
            
            # decoder_output = tf.concat([decoder_output[:,:4],tf.zeros(shape=(tf.shape(input_data)[0], 1))*0.5],
            #                             axis=1)
            decoder_input = tf.concat([time_step[:,i,:], decoder_output], axis=1)
            # 记录预测值
            predict_data = predict_data.write(i, decoder_output)

        # 交换维度
        predict_data = predict_data.stack()
        predict_data = tf.transpose(predict_data, perm=[1, 0, 2])
        return predict_data
        
    def predict_eager(self, input_data, time_step, output_size, same_shape=False):
        '''
        预测(Eager模式)
        input_data:(1, history_size, input_num)
        time_step:预测时间序列，(1,target_size,6)
        output_size:预测数量
        same_shape:是否相同类数据预测
        '''
        # print('Tracing with predict', type(input_data), type(output_size))
        # print('Tracing with predict', input_data.shape, output_size)
        predict_data = []
        input_data = tf.constant(input_data, dtype=tf.float32)
        time_step = tf.constant(time_step, dtype=tf.float32)
        # 编码
        # encoder_output(history_size,128)
        # encoder_state1(history_size,128)
        # encoder_state2(history_size,128)
        encoder_output, encoder_state1, encoder_state2 = self.encoder_model(input_data)
        # print('Tracing with encoder_output', encoder_output, encoder_state1, encoder_state2)
        # tf.print('Action with encoder_output', tf.shape(encoder_output), tf.shape(encoder_state1), tf.shape(encoder_state2))
        decoder_state1 = encoder_state1
        decoder_state2 = encoder_state2
        if same_shape:
            decoder_input = input_data[:,-1,:]
        else:
            decoder_input = tf.concat([input_data[:,-1,:6], 
                                      tf.zeros(shape=(tf.shape(input_data)[0], self.output_num-6))],
                                      axis=1)

        # 解码
        for i in range(output_size):
            # 解码
            decoder_output, decoder_state1, decoder_state2 = self.decoder_model(
                decoder_input, decoder_state1, decoder_state2, encoder_output)
            decoder_input = tf.concat([time_step[:,i,:], decoder_output], axis=1)
            # 记录预测值
            predict_data.append(decoder_output.numpy())

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


class DataPerdictLoader():
    '''加载数据文件'''

    def __init__(self, input_colname, output_colname):
        self.input_colname = input_colname
        self.output_colname = output_colname

    def load_file(self, file_name):
        '''加载数据文件'''
        print('加载文件', file_name)
        df = pandas.read_csv(file_name)
        return df

    def get_random_data(self, df, history_size, target_size, start_index=None):
        '''根据数据窗口获取数据'''
        data = []
        labels = []
        # 随机取一段时间数据
        if start_index==None:
            start_index = random.randint(history_size, len(df)-target_size-1)
        tmp_df = df.iloc[start_index-history_size:start_index+target_size].copy()
        # 数据归一化
        tmp_df.loc[:, 'VALUE'] = tmp_df.apply(lambda x: x['VALUE'] / 500, axis=1)
        tmp_df.loc[:, 'VALUE_df2'] = tmp_df.apply(lambda x: x['VALUE_df2'] / 300, axis=1)
        tmp_df.loc[:, 'VALUE_df3'] = tmp_df.apply(lambda x: x['VALUE_df3'] / 100, axis=1)
        tmp_df.loc[:, 'VALUE_df4'] = tmp_df.apply(lambda x: x['VALUE_df4'] / 5000, axis=1)
        tmp_df.loc[:, 'VALUE_df5'] = tmp_df.apply(lambda x: x['VALUE_df5'] / 20000, axis=1)
        # 增加列
        col_name = tmp_df.columns.tolist()
        col_name.insert(1, 'year')  # 默认值为NaN
        col_name.insert(2, 'month')  # 默认值为NaN
        col_name.insert(3, 'day')  # 默认值为NaN
        col_name.insert(4, 'hour')  # 默认值为NaN
        col_name.insert(5, 'minute')  # 默认值为NaN
        col_name.insert(6, 'second')  # 默认值为NaN
        tmp_df = tmp_df.reindex(columns=col_name)
        # 日期数据归一化
        tmp_df.loc[:, 'year'] = tmp_df.apply(lambda x: (datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').year-2000) / 20, axis=1)
        tmp_df.loc[:, 'month'] = tmp_df.apply(lambda x: (datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').month) / 12, axis=1)
        tmp_df.loc[:, 'day'] = tmp_df.apply(lambda x: (datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').day) / 31, axis=1)
        tmp_df.loc[:, 'hour'] = tmp_df.apply(lambda x: (datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').hour) / 24, axis=1)
        tmp_df.loc[:, 'minute'] = tmp_df.apply(lambda x: (datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').minute) / 60, axis=1)
        tmp_df.loc[:, 'second'] = tmp_df.apply(lambda x: (datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').second) / 60, axis=1)
        # print('tmp_df_merge', len(tmp_df_merge))
        # print(tmp_df_merge.loc[:, ['data','年','月','日']])
        # 重置索引
        tmp_df = tmp_df.reset_index(drop=True)
        return tmp_df

    def get_data_to_train(self, df, batch_size, history_size, target_size, start_index=None):
        '''
        数据格式化用于训练
        batch_size:批次大小
        history_size:训练数据大小
        target_size:预测数据大小
        '''
        x = []
        y = []
        for _ in range(batch_size):
            tmp_df = self.get_random_data(df, history_size, target_size, start_index)
            x.append(tmp_df.iloc[:history_size,:].loc[:, self.input_colname].values.tolist())
            y.append(tmp_df.iloc[history_size:,:].loc[:, self.output_colname].values.tolist())
        x = np.array(x)
        y = np.array(y)
        # print('x', x.shape, 'y', y.shape)
        return x, y

    def get_data_to_predict(self, df, history_size, target_size, start_index=None, end=False):
        '''
        数据格式化用于训练
        batch_size:批次大小
        history_size:训练数据大小
        target_size:预测数据大小
        '''
        tmp_target_size = target_size
        if start_index==None:
            start_index = len(df)-target_size
        if end:
            start_index = len(df)
            tmp_target_size = 0
        tmp_df = self.get_random_data(df, history_size, tmp_target_size, start_index)
        # print('predict tmp_df', tmp_df)
        # print('predict y_history', tmp_df.iloc[:history_size,:])
        # print('predict y_target', tmp_df.iloc[history_size:,:])
        x = tmp_df.iloc[:history_size,:].loc[:, self.input_colname].values
        x = np.expand_dims(x, axis=0)
        y_history = tmp_df.iloc[:history_size,:].loc[:, self.output_colname].values
        y_history = np.expand_dims(y_history, axis=0)
        y_target = tmp_df.iloc[history_size:,:].loc[:, self.output_colname].values
        y_target = np.expand_dims(y_target, axis=0)
        time_step = self.create_time(tmp_df.iloc[history_size-1,:].loc['CREATE_TIME'], target_size, minutes=5)
        time_step = np.expand_dims(time_step, axis=0)
        return x, y_history, y_target, time_step
    
    def data_generator(self, df, batch_size, history_size, target_size):
        '''循环生成数据'''
        while True:
            x, y = self.get_data_to_train(df, batch_size, history_size, target_size)
            yield x, y
    
    def create_time(self, start_time, target_size, days=0, hours=0, minutes=0, seconds=0):
        '''
        创建预测时序
        target_size:预测数据大小
        '''
        # print('开始时间', start_time)
        tmp_start_time = datetime.datetime.strptime(start_time,'%Y/%m/%d %H:%M:%S')
        result = []
        for i in range(target_size):
            if days!=0:
                tmp_start_time = tmp_start_time + datetime.timedelta(days=days)
            if hours!=0:
                tmp_start_time = tmp_start_time + datetime.timedelta(hours=hours)
            if minutes!=0:
                tmp_start_time = tmp_start_time + datetime.timedelta(minutes=minutes)
            if seconds!=0:
                tmp_start_time = tmp_start_time + datetime.timedelta(seconds=seconds)
            tmp_year = (tmp_start_time.year - 2000) / 20
            tmp_month = tmp_start_time.month / 12
            tmp_day = tmp_start_time.day / 31
            tmp_hour = tmp_start_time.hour / 24
            tmp_minute = tmp_start_time.minute / 60
            tmp_second = tmp_start_time.second / 60
            result.append([tmp_year, tmp_month, tmp_day, tmp_hour, tmp_minute, tmp_second])
        
        result = np.array(result)
        # print('时间序列', result)
        return result

    def save_csv(self, file_path, time_step, perdict_data):
        '''保存csv'''
        new_df = pandas.DataFrame(np.append(time_step, perdict_data, axis=1),
                                columns=['year','month','day','hour','minute','second','VALUE','VALUE_df2','VALUE_df3','VALUE_df4','VALUE_df5'])
        
        # 增加列
        col_name = new_df.columns.tolist()
        col_name.insert(0, 'CREATE_TIME')  # 默认值为NaN
        new_df = new_df.reindex(columns=col_name)
        new_df.loc[:, 'CREATE_TIME'] = new_df.apply(lambda x: datetime.datetime(
            int(x['year'] * 20 + 2000), int(x['month'] * 12), int(x['day'] * 31), 
            int(x['hour'] * 24), int(x['minute'] * 60), int(x['second'] * 60)).strftime('%Y/%m/%d %H:%M:%S'), axis=1)
        new_df.loc[:, 'VALUE'] = new_df.apply(lambda x: x['VALUE'] * 500, axis=1)
        new_df.loc[:, 'VALUE_df2'] = new_df.apply(lambda x: x['VALUE_df2'] * 300, axis=1)
        new_df.loc[:, 'VALUE_df3'] = new_df.apply(lambda x: x['VALUE_df3'] * 100, axis=1)
        new_df.loc[:, 'VALUE_df4'] = new_df.apply(lambda x: x['VALUE_df4'] * 5000, axis=1)
        new_df.loc[:, 'VALUE_df5'] = new_df.apply(lambda x: x['VALUE_df5'] * 20000, axis=1)
        new_df.loc[:, ['CREATE_TIME','VALUE','VALUE_df2','VALUE_df3','VALUE_df4','VALUE_df5']].to_csv(file_path, index=0) #不保存行索引

    def show_image(self, history_data, target_data, perdict_data=None):
        '''
        显示K线图
        history_data:(None,2)
        target_data:(None,2)
        '''
        # print(show_history_data)
        # 创建一个子图 
        fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5))
        fig.subplots_adjust(bottom=0.2)
        # 设置X轴刻度为日期时间
        # ax.xaxis_date()
        # # X轴刻度文字倾斜45度
        # plt.xticks(rotation=45)
        plt.title("曲线图")
        plt.xlabel("时间")
        plt.ylabel("数值")
        for i in range(history_data.shape[1]):
            plt.plot(range(history_data.shape[0]+target_data.shape[0]), np.append(history_data[:,i], target_data[:,i], axis=0) , 'b')
            if perdict_data is not None:
                plt.plot(range(history_data.shape[0],history_data.shape[0]+perdict_data.shape[0]), perdict_data[:,i], 'r')
        plt.show()

def main():
    # 输入数据维度
    input_num = model_input_num
    # 预测数据维度
    output_num = model_output_num
    batch_size = 5
    history_size = 100
    target_size = 25
    predict_target_size = 25
    # 输入输出字段
    # input_colname = ['year','month','day','hour','minute','second','col1','col2']
    # input_colname = ['year','month','day','hour','minute','second','col4']
    # output_colname = ['year','month','day','hour','minute','second','col4']
    input_colname = ['year','month','day','hour','minute','second','VALUE','VALUE_df2','VALUE_df3','VALUE_df4','VALUE_df5']
    output_colname = ['year','month','day','hour','minute','second','VALUE','VALUE_df2','VALUE_df3','VALUE_df4','VALUE_df5']
    # 创建模型
    print('创建模型')
    data_perdict_model = DataPerdictModel(input_num, output_num)
    # 加载数据
    print('加载数据')
    data_perdict_loader = DataPerdictLoader(input_colname, output_colname)
    df = data_perdict_loader.load_file('./data/bdzdata_merge.csv')
    print(df)
    # 训练数据集
    # df_train = df.iloc[:-100,:].copy()
    df_train = df.copy()
    # 测试数据集
    df_test = df.iloc[-100:,:].copy()
    print('训练前预测')
    x, y_history = data_perdict_loader.get_data_to_train(df_train, batch_size, history_size, target_size)
    print('x', x[:,-5:])
    print('y_history', y_history[:,-5:])
    x, y_history, y_target, time_step = data_perdict_loader.get_data_to_predict(df_test, history_size, predict_target_size, end=True)
    print('x', x.shape, 'y_history', y_history.shape, 'y_target', y_target.shape, 'time_step', time_step.shape)
    # print('y_target', y_target)
    y2 = data_perdict_model.predict_jit(x, time_step, predict_target_size, same_shape=True)
    print('y2', y2.shape)
    print('y2', y2)
    # 保存csv
    data_perdict_loader.save_csv('./data/bdzdata_new.csv', time_step[0,:,:], y2[0,:,:])
    # 显示预测值
    data_perdict_loader.show_image(y_history[0,:,6:], y_target[0,:,6:], y2[0,:,:])
    # 开始训练
    print('开始训练')
    data_perdict_model.fit_generator(
        data_perdict_loader.data_generator(df_train, batch_size, history_size, target_size),
        steps_per_epoch=int(len(df_train)/history_size),
        epochs=50, auto_save=True, same_shape=True)
    # 预测
    print('预测')
    y2 = data_perdict_model.predict_jit(x, time_step, predict_target_size, same_shape=True)
    # 显示预测值
    data_perdict_loader.show_image(y_history[0,:,6:], y_target[0,:,6:], y2[0,:,:])
    # 保存csv
    data_perdict_loader.save_csv('bdzdata_new.csv', time_step[0,:,:], y2[0,:,:])



if __name__ == '__main__':
    main()