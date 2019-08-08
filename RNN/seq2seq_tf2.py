import tensorflow as tf
import numpy as np
import math
import os
import random


class Encoder(tf.keras.Model):
    """编码器"""
    def __init__(self, units, num_tokens):
        super(Encoder, self).__init__(self)
        # return_state返回状态，用于状态保持
        self.units = units
        self.num_tokens = num_tokens
        # 编码器，将num_tokens个字符，编码成(num_tokens,units)的二维数组
        self.embedding = tf.keras.layers.Embedding(num_tokens, units)
        self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True,
                                       return_state=True, activation=tf.keras.activations.tanh)
        self.lstm2 = tf.keras.layers.LSTM(units, return_sequences=False,
                                        return_state=True, activation=tf.keras.activations.tanh)

    def call(self, input_data):
        # input_data shape=(None, num_tokens)
        input_data=self.embedding(input_data)
        encoder_outputs, state_h, state_c = self.lstm1(input_data)
        # encoder_outputs2编码器最后输出
        encoder_outputs, state_h2, state_c2 = self.lstm2(encoder_outputs)
        return state_h, state_c, state_h2, state_c2
    
    def initialize_hidden_state(self):
        return tf.zeros((self.num_tokens, self.units))

class Decoder(tf.keras.Model):
    """解码器"""
    def __init__(self, units, num_tokens):
        super(Decoder, self).__init__(self)
        self.units = units
        self.num_tokens = num_tokens
        # 编码器，将num_tokens个字符，编码成(num_tokens,units)的二维数组
        self.embedding = tf.keras.layers.Embedding(num_tokens, units)
        self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True,
                                            return_state=True, activation=tf.keras.activations.tanh)
        self.lstm2 = tf.keras.layers.LSTM(units, return_sequences=True,
                                             return_state=True, activation=tf.keras.activations.tanh)
        self.dense = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.linear)

    def call(self, input_data, state_h, state_c, state_h2, state_c2):
        # shape(1,1)=>(1,1,units)
        input_data=self.embedding(input_data)
        # shape(1,1,units)
        decoder_outputs, state_h, state_c = self.lstm1(
            input_data, initial_state=[state_h, state_c])
        decoder_outputs, state_h2, state_c2 = self.lstm2(
            decoder_outputs, initial_state=[state_h2, state_c2])
        # shape(1,1,units)=>(1,units)
        decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
        # 输出值，真正答案
        # shape(1,units)=>(1,num_tokens)
        decoder_outputs = self.dense(decoder_outputs)
        return decoder_outputs, state_h, state_c, state_h2, state_c2

class MyRobot:
    """聊天机器人"""

    def __init__(self):
        self.batch_size = 1  # Batch size for training. 训练批次大小
        self.epochs = 1  # Number of epochs to train for. 训练多少回
        self.latent_dim = 128  # 隐藏神经元数量\字符编码器特征数
        self.num_samples = 0  # Number of samples to train on. 训练数量
        self.max_length = 256  # 句子最大长度
        self.word_all = ''  # 中文字符集
        self.alphabet = ''  # 字符集
        self.word_file_path = 'word_all.txt'  # 字符集文件路径
        self.train_file_path = 'ai.txt'  # 训练集文件路径
        self.char_to_int = None  # 字符转序号
        self.int_to_char = None  # 序号转字符
        self.question_texts = []  # 问题集合
        self.answer_texts = []  # 答案集合
        self.num_tokens = 0  # 字符位数，加载训练集后计算
        self.encoder_input_data = None  # 训练时，编码器输入，one hot
        self.decoder_input_data = None  # 训练时，解码器输入，one hot
        self.decoder_target_data = None  # 训练时，解码器输出，one hot

    def load_word_file(self):
        """加载字库"""
        print('正在加载字库...')
        with open(self.word_file_path, 'r', encoding='UTF-8') as word_file:
            self.word_all = word_file.read()  # 两万多个字符
            self.alphabet = '\0\t\n'  # \0：填充字符，\t：开头标志，\n：结束标志
            self.alphabet += self.word_all
            word_file.close()
        print('正在加载字库完成!')

    def add_word(self, char):
        """新增字符"""
        f2 = open(self.word_file_path, 'w', encoding='utf-8')
        f2.truncate()  # 清空文件
        self.word_all += char
        self.alphabet = '\0\t\n'  # \0：填充字符，\t：开头标志，\n：结束标志
        self.alphabet += self.word_all
        f2.write(self.word_all)
        f2.close()

    def load_train_file(self):
        """加载训练素材"""
        print('正在加载训练素材...')
        # 训练数据集
        with open(self.train_file_path, 'r', encoding='UTF-8') as train_file:
            sentences = train_file.read().split('\n')
            train_file.close()
        self.question_texts = []
        self.answer_texts = []
        for senterce in sentences:
            if len(senterce) == 0:
                continue
            # 补全缺失文字
            for char in senterce:
                if self.alphabet.find(char) == -1:
                    self.add_word(char)
            print('senterce', senterce.split('\t'))
            question_text, answer_text = senterce.split('\t')
            # \t 作为开头标识
            # \n 作为结尾标识
            question_text = '\t' + question_text + '\n'
            answer_text = '\t' + answer_text + '\n'
            self.question_texts.append(question_text)
            self.answer_texts.append(answer_text)
        # print('question_texts', question_texts)
        # print('answer_texts', answer_texts)


        # 字典字符数量
        self.num_tokens = len(self.alphabet)
        self.num_samples = len(self.question_texts)
        print('加载训练素材完成!')
        print('字符数：', self.num_tokens)
        print('素材数：', self.num_samples)

    def text_to_indexs(self, text):
        """训练素材句子转序号"""
        result=[]
        for char in text:
            result.append(self.char_to_int[char])
        return result

    def indexs_to_text(self, indexs):
        """序号转训练素材句子"""
        result=''
        for index in indexs:
            result += self.int_to_char[index]
        return result


    def create_index(self):
        """训练素材字符转序号"""
        # 字符与序号对应的字典
        self.char_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.alphabet))
        # print('char_to_int', char_to_int)
        # print('int_to_char', int_to_char)

        # 输入
        self.encoder_input_data = np.zeros(
            (self.num_samples, self.max_length),
            dtype='float32')
        # 输出
        self.decoder_input_data = np.zeros(
            (self.num_samples, self.max_length),
            dtype='float32')
        # 下一个时间点的输出
        self.decoder_target_data = np.zeros(
            (self.num_samples, self.max_length),
            dtype='float32')

        # enumerate返回下标与元素，zip把两个列表打包成一个个元组组成的列表
        # 循环句子
        for i, (input_text, target_text) in enumerate(zip(self.question_texts, self.answer_texts)):
            # print('input_text', input_text)
            # print('target_text', target_text)
            # 循环字符
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t] = self.char_to_int[char]
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t] = self.char_to_int[char]
                # 翻译时下一个时间点的输入数据
                if t > 0:
                    # 最终输出结果，与结果输入错开一位，模拟识别时的循环输入方式
                    self.decoder_target_data[i, t-1] = self.char_to_int[char]
        # print('encoder_input_data', len(encoder_input_data))
        # print('decoder_input_data', len(decoder_input_data))

    def build_model(self):
        """创建模型"""
        # ==================编码器=====================
        # 用于提取句子特征，返回最后时间点的特征数组与隐藏层状态
        self.encoder = Encoder(self.latent_dim, self.num_tokens)
        # ==================编码器 end=====================

        # ==================解码器=====================
        # 解码输入，以\t开头，用上一次输出的字符作为下一次的输入，训练时预测
        self.decoder = Decoder(self.latent_dim, self.num_tokens)
        # ==================解码器 end=====================

        # 编码					解码
        # \t	h	i	\n		\t	你	好	\n
        # LSTM					LSTM
        # 					    你	好	\n
        # 构建训练模型
        # encoder_inputs = tf.keras.Input(shape=(self.max_length))
        # decoder_inputs = tf.keras.Input(shape=(self.max_length))
        # state_h, state_c, state_h2, state_c2=self.encoder(encoder_inputs)
        # decoder_outputs, state_h, state_c, state_h2, state_c2=self.decoder(decoder_inputs,state_h, state_c, state_h2, state_c2)
        # self.train_model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # 定义损失函数
        # self.loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.loss=tf.keras.losses.CategoricalCrossentropy(reduction='none')
        # 定义使用的梯度下降方式
        self.optimizer=tf.keras.optimizers.Adadelta(lr=0.1)
    
    @tf.function
    def train_one(self, encoder_input_data_one, decoder_input_data_one):
        """训练（单个句子）"""
        # Run training
        # 训练
        # encoder_input_data：输入要翻译的语句
        # decoder_input_data：输入解码器的结果\t开头
        # decoder_target_data：真正的翻译结果
        print('encoder_input_data_one(句子长度)',encoder_input_data_one.shape)
        print('decoder_input_data_one(句子长度)',decoder_input_data_one.shape)
        encoder_input_data_one=tf.reshape(encoder_input_data_one,(1,encoder_input_data_one.shape[0]))
        decoder_input_data_one=tf.reshape(decoder_input_data_one,(1,decoder_input_data_one.shape[0]))
        print('encoder_input_data_one(1,句子长度)',encoder_input_data_one.shape)
        print('decoder_input_data_one(1,句子长度)',decoder_input_data_one.shape)
        with tf.GradientTape() as tape:
            # 编码
            state_h, state_c, state_h2, state_c2 = self.encoder(encoder_input_data_one)
            # 解码开始符号
            target_seq = np.array([[self.char_to_int['\t']]])
            # print('target_seq',target_seq.shape)

            # 循环解码
            loss = 0
            for t in range(1, decoder_input_data_one.shape[1]):
                print('解码', t)
                # 解码
                output_tokens, h, c, h2, c2 = self.decoder(target_seq, state_h, state_c, state_h2, state_c2)

                # 累计损失，不是句子末尾空白符合时
                if decoder_input_data_one[:,t]!=0:
                    # shape(1,1),(1,num_tokens)
                    loss += self.loss(decoder_input_data_one[:,t], output_tokens)
                # print('loss', loss)
                # 当前字符，传递到下一次预测
                target_seq = tf.reshape(decoder_input_data_one[:,t],(1,1))

                # Update states
                # 当前状态，传递到下一次预测
                state_h, state_c, state_h2, state_c2 = [h, c, h2, c2]
            print('开始更新权重，loss', loss)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        print('variables', variables)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def save_model(self):
        # Save model
        # self.train_model.save_weights('s2s.h5')
        pass

    @tf.function
    def predict(self, input_data_one):
        """识别（单个句子）"""
        input_data_one=tf.reshape(input_data_one,(1,input_data_one.shape[0]))
        # 编码
        state_h, state_c, state_h2, state_c2 = self.encoder(input_data_one)
        # 解码开始符号
        target_seq = np.array([[self.char_to_int['\t']]])

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        char_num = 0
        while not stop_condition:
            # output_tokens.shape(1,num_tokens)
            output_tokens, h, c, h2, c2 = self.decoder(target_seq, state_h, state_c, state_h2, state_c2)

            # print('output_tokens', output_tokens)
            # 对应字符下标，把预测出的字符拼成字符串
            # Sample a token
            sampled_token_index = np.maximum(output_tokens.numpy()[0])
            sampled_char = self.int_to_char[sampled_token_index]
            decoded_sentence += sampled_char
            char_num += 1

            # 句子结束
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    char_num > self.max_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            # 当前字符，传递到下一次预测
            target_seq = tf.reshape([sampled_token_index],(1,1))

            # Update states
            # 当前状态，传递到下一次预测
            state_h, state_c, state_h2, state_c2 = [h, c, h2, c2]

        return decoded_sentence


def main():
    my_robot = MyRobot()
    # 加载字符集
    my_robot.load_word_file()
    # 加载训练素材
    my_robot.load_train_file()
    # 训练素材字符转序号
    my_robot.create_index()
    # 生成模型
    my_robot.build_model()

    i = 0
    while i != len(my_robot.encoder_input_data):
        i = 0
        # my_robot.train(my_robot.encoder_input_data,
        #                my_robot.decoder_input_data, my_robot.decoder_target_data)
        for (encoder_input_data_one, decoder_input_data_one, decoder_target_data_one) in zip(my_robot.encoder_input_data, my_robot.decoder_input_data, my_robot.decoder_target_data):
            # 问题
            input_sentence = my_robot.indexs_to_text(encoder_input_data_one)
            # 机器人回答
            decoded_sentence = ''
            # 正确答案
            output_sentence = my_robot.indexs_to_text(decoder_target_data_one)
            # 答案有误时50%重复训练
            # while decoded_sentence != output_sentence and random.random() < 0.5:
            while decoded_sentence != output_sentence:
                i += 1
                # 训练
                my_robot.train_one(encoder_input_data_one, decoder_input_data_one)
                print('训练完')
                # 识别
                decoded_sentence = my_robot.predict(encoder_input_data_one)
                print('Input sentence:', input_sentence)
                print('Output sentence:', output_sentence)
                print('Decoded sentence:', decoded_sentence)
        # 保存模型
        my_robot.save_model()

    for seq_index in range(50):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = my_robot.encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = my_robot.predict(input_seq)
        print('-')
        print('Input sentence:', my_robot.question_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

    # print('请输入问题：')
    # question = input()
    # while question != 'q':
    #     input_seq = my_robot.one_hot_encoder(question)
    #     input_seq = input_seq.reshape(
    #         (1, input_seq.shape[0], input_seq.shape[1]))
    #     decoded_sentence = my_robot.predict(input_seq)
    #     print('-')
    #     print('Input sentence:', question)
    #     print('Decoded sentence:', decoded_sentence)
    #     question = input()


if __name__ == '__main__':
    main()
