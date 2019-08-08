import keras as K
import numpy as np
import math
import os
import random


class MyRobot:
    """聊天机器人"""

    def __init__(self):
        self.batch_size = 1  # Batch size for training. 训练批次大小
        self.epochs = 1  # Number of epochs to train for. 训练多少回
        self.latent_dim = 128  # Latent dimensionality of the encoding space. 隐藏神经元数量
        self.num_samples = 0  # Number of samples to train on. 训练数量
        self.max_encoder_seq_length = 256  # 句子最大长度
        self.word_all = ''  # 中文字符集
        self.alphabet = ''  # 字符集
        self.word_file_path = 'word_all.txt'  # 字符集文件路径
        self.train_file_path = 'ai.txt'  # 训练集文件路径
        self.char_to_int = None  # 字符转序号
        self.int_to_char = None  # 序号转字符
        self.question_texts = []  # 问题集合
        self.answer_texts = []  # 答案集合
        self.num_tokens = 48  # 字符位数
        self.encoder_input_data = None  # 训练时，编码器输入，one hot
        self.decoder_input_data = None  # 训练时，解码器输入，one hot
        self.decoder_target_data = None  # 训练时，解码器输出，one hot

    def load_word_file(self):
        """加载字库"""
        print('正在加载字库...')
        with open(self.word_file_path, 'r', encoding='UTF-8') as word_file:
            self.word_all = word_file.read()  # 2500
            self.alphabet = self.word_all
            self.alphabet += '\t\n'  # 开头结束标志
            word_file.close()
        print('正在加载字库完成!')

    def add_word(self, char):
        """新增字符"""
        f2 = open(self.word_file_path, 'w', encoding='utf-8')
        f2.truncate()  # 清空文件
        self.word_all += char
        f2.write(self.word_all)
        f2.close()
        self.alphabet = self.word_all
        self.alphabet += '\t\n'  # 开头结束标志

    def int2bit(self, num):
        bit = []
        while True:
            if num == 0:
                break
            num, rem = divmod(num, 2)  # 10进制转换2进制，使用除2取余
            bit.append(rem)  # 默认加入列表是filo（先进后出）方式
        return bit[::-1]

    def bit2int(self, num):
        return_num = 0
        num = num[::-1]
        for i in range(len(num)):
            if num[i] < 0.5:
                return_num += math.pow(2, i)*0
            else:
                return_num += math.pow(2, i)*1
        return int(return_num)

    def char2utf8_bit(self, char):
        num = char.encode('utf-8')
        word_one_hot = []
        for i in num:
            word_one_hot += self.int2bit(i)
        word_one_hot = [0]*(48-len(word_one_hot))+word_one_hot
        return word_one_hot

    def utf8_bit2char(self, bit):
        bit = np.array(bit, dtype='float32').reshape((6, 8))
        bits = []
        for i in range(6):
            num = self.bit2int(bit[i])
            if num != 0:
                bits.append(num)
        return bytes(bits).decode('utf-8')

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
            # # 补全缺失文字
            # for char in senterce:
            #     if self.alphabet.find(char) == -1:
            #         self.add_word(char)
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

        # # 字符与序号对应的字典
        # self.char_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        # self.int_to_char = dict((i, c) for i, c in enumerate(self.alphabet))
        # print('char_to_int', char_to_int)
        # print('int_to_char', int_to_char)

        # # 字典字符数量
        # self.num_tokens = len(self.alphabet)
        self.num_samples = len(self.question_texts)
        print('正在加载训练素材完成!')
        # print('字符数：', self.num_tokens)
        print('素材数：', self.num_samples)

    def one_hot(self):
        """训练素材转one hot"""
        # 输入
        self.encoder_input_data = np.zeros(
            (len(self.question_texts), self.max_encoder_seq_length, self.num_tokens),
            dtype='float32')
        # 输出
        self.decoder_input_data = np.zeros(
            (len(self.question_texts), self.max_encoder_seq_length, self.num_tokens),
            dtype='float32')
        # 下一个时间点的输出
        self.decoder_target_data = np.zeros(
            (len(self.question_texts), self.max_encoder_seq_length, self.num_tokens),
            dtype='float32')

        # enumerate返回下标与元素，zip把两个列表打包成一个个元组组成的列表
        # 下面循环生成训练数据，转one hot
        for i, (input_text, target_text) in enumerate(zip(self.question_texts, self.answer_texts)):
            # print('input_text', input_text)
            # print('target_text', target_text)
            for t, char in enumerate(input_text):
                char_bit = self.char2utf8_bit(char)
                for i2 in range(self.num_tokens):
                    self.encoder_input_data[i, t, i2] = char_bit[i2]
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                char_bit = self.char2utf8_bit(char)
                for i2 in range(self.num_tokens):
                    self.decoder_input_data[i, t, i2] = char_bit[i2]
                # 翻译时下一个时间点的输入数据
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    for i2 in range(self.num_tokens):
                        self.decoder_target_data[i, t-1, i2] = char_bit[i2]
        # print('encoder_input_data', len(encoder_input_data))
        # print('decoder_input_data', len(decoder_input_data))

    def one_hot_encoder(self, sentence):
        """句子转one hot"""
        # 输入
        encoder_input_data_one = np.zeros(
            (self.max_encoder_seq_length, self.num_tokens),
            dtype='float32')

        for t, char in enumerate(sentence):
            char_bit = self.char2utf8_bit(char)
            for i2 in range(self.num_tokens):
                encoder_input_data_one[t, i2] = char_bit[i2]

        return encoder_input_data_one

    def one_hot_decoder(self, sentence):
        """one hot反编译"""
        decoded_sentence = ''
        for char in sentence:
            sampled_char = self.utf8_bit2char(char)
            decoded_sentence += sampled_char
            if sampled_char == '\n':
                break
        return decoded_sentence

    def build_model(self):
        """创建模型"""
        # ==================编码器=====================
        # 用于提取句子特征，返回最后时间点的特征数组与隐藏层状态
        # 输入一句话
        encoder_inputs = K.Input(shape=(None, self.num_tokens))
        # return_state返回状态，用于状态保持
        encoder = K.layers.LSTM(self.latent_dim, return_sequences=True,
                                return_state=True, activation=K.activations.tanh)
        encoder2 = K.layers.LSTM(self.latent_dim, return_sequences=False,
                                 return_state=True, activation=K.activations.tanh)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # encoder_outputs2编码器最后输出
        encoder_outputs, state_h2, state_c2 = encoder2(encoder_outputs)
        # 编码器最后状态，包含两层LSTM的状态
        encoder_states = [state_h, state_c]
        encoder_states2 = [state_h2, state_c2]
        # ==================编码器 end=====================

        # ==================解码器=====================
        # 解码输入，以\t开头，用上一次输出的字符作为下一次的输入，训练时预测
        decoder_inputs = K.Input(shape=(None, self.num_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        # return_sequences返回完整序列
        decoder_lstm = K.layers.LSTM(self.latent_dim, return_sequences=True,
                                     return_state=True, activation=K.activations.tanh)
        decoder_lstm2 = K.layers.LSTM(self.latent_dim, return_sequences=True,
                                      return_state=True, activation=K.activations.tanh)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=encoder_states)
        decoder_outputs, _, _ = decoder_lstm2(
            decoder_outputs, initial_state=encoder_states2)
        decoder_dense = K.layers.Dense(
            self.num_tokens, activation=K.activations.softmax)
        # 输出值，真正答案
        decoder_outputs = decoder_dense(decoder_outputs)
        # ==================解码器 end=====================

        # 编码					解码
        # \t	h	i	\n		\t	你	好	\n
        # LSTM					LSTM
        # 					    你	好	\n
        # 构造训练模型
        self.train_model = K.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs)
        if os.path.exists('s2s.h5'):
            print('加载模型')
            self.train_model.load_weights('s2s.h5')
        # 编译模型，定义损失函数
        self.train_model.compile(K.optimizers.Adadelta(lr=0.1), loss=[
                                 K.losses.mean_absolute_error])

        # ==================识别模型=====================
        # 编码模型,encoder_states
        self.predict_encoder_model = K.Model(
            encoder_inputs, encoder_states + encoder_states2)

        # 解码模型
        # 状态输入
        decoder_state_input_h = K.Input(shape=(self.latent_dim,))
        decoder_state_input_c = K.Input(shape=(self.latent_dim,))
        decoder_state_input_h2 = K.Input(shape=(self.latent_dim,))
        decoder_state_input_c2 = K.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_states_inputs2 = [
            decoder_state_input_h2, decoder_state_input_c2]
        # 训练后的LSTM,
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs2, state_h2, state_c2 = decoder_lstm2(
            decoder_outputs, initial_state=decoder_states_inputs2)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs = decoder_dense(decoder_outputs2)
        # 输入[decoder_inputs, decoder_state_input_h, decoder_state_input_c]
        # 输出[decoder_outputs, state_h, state_c]
        self.predict_decoder_model = K.Model(
            [decoder_inputs] + decoder_states_inputs + decoder_states_inputs2,
            [decoder_outputs] + decoder_states + decoder_states2)
        # ==================识别模型 end=====================

    def train(self, encoder_input_data_one, decoder_input_data_one, decoder_target_data_one):
        """训练（单个句子）"""
        # Run training
        # 训练
        # encoder_input_data：输入要翻译的语句
        # decoder_input_data：输入解码器的结果\t开头
        # decoder_target_data：真正的翻译结果
        # self.train_model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
        #         batch_size=self.batch_size,epochs=self.epochs)
        self.train_model.fit(
            [encoder_input_data_one, decoder_input_data_one], decoder_target_data_one)

    def save_model(self):
        # Save model
        self.train_model.save_weights('s2s.h5')

    def predict(self, input_data_one):
        """识别（单个句子）"""
        # 编码，抽象概念
        states_value = self.predict_encoder_model.predict(input_data_one)

        # # Generate empty target sequence of length 1.
        # target_seq = np.zeros((1, 1, self.num_tokens))
        # # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, self.char_to_int['\t']] = 1.
        target_seq = np.array(self.char2utf8_bit(
            '\t')).reshape((1, 1, self.num_tokens))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        char_num = 0
        while not stop_condition:
            output_tokens, h, c, h2, c2 = self.predict_decoder_model.predict(
                [target_seq] + states_value)

            # print('output_tokens', output_tokens)
            # 对应字符下标，把预测出的字符拼成字符串
            # Sample a token
            # sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.utf8_bit2char(output_tokens[0, 0, :])
            decoded_sentence += sampled_char
            char_num += 1

            # 句子结束
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    char_num > self.max_encoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            # 当前字符，传递到下一次预测
            target_seq = np.zeros((1, 1, self.num_tokens))
            for i, num in enumerate(output_tokens[0, -1, :]):
                if num < 0.5:
                    target_seq[0, 0, i] = 0
                else:
                    target_seq[0, 0, i] = 1

            # Update states
            # 当前状态，传递到下一次预测
            states_value = [h, c, h2, c2]

        return decoded_sentence


def main():
    my_robot = MyRobot()
    # 加载字符集
    my_robot.load_word_file()
    # 加载训练素材
    my_robot.load_train_file()
    # 训练素材转one hot
    my_robot.one_hot()
    # 生成模型
    my_robot.build_model()

    i = 0
    while i != len(my_robot.encoder_input_data):
        i = 0
        my_robot.train(my_robot.encoder_input_data,
                       my_robot.decoder_input_data, my_robot.decoder_target_data)
        # for (encoder_input_data_one, decoder_input_data_one, decoder_target_data_one) in zip(my_robot.encoder_input_data, my_robot.decoder_input_data, my_robot.decoder_target_data):
        #     # 问题
        #     input_sentence = my_robot.one_hot_decoder(encoder_input_data_one)
        #     # 机器人回答
        #     decoded_sentence = ''
        #     # 正确答案
        #     output_sentence = my_robot.one_hot_decoder(decoder_target_data_one)
        #     # 答案有误时50%重复训练
        #     # while decoded_sentence != output_sentence and random.random() < 0.5:
        #     while decoded_sentence != output_sentence:
        #         i += 1
        #         # 训练
        #         my_robot.train(encoder_input_data_one.reshape((1, encoder_input_data_one.shape[0], encoder_input_data_one.shape[1])),
        #                        decoder_input_data_one.reshape(
        #             (1, decoder_input_data_one.shape[0], decoder_input_data_one.shape[1])),
        #             decoder_target_data_one.reshape((1, decoder_target_data_one.shape[0], decoder_target_data_one.shape[1])))
        #         # 识别
        #         decoded_sentence = my_robot.predict(encoder_input_data_one.reshape(
        #             (1, encoder_input_data_one.shape[0], encoder_input_data_one.shape[1])))
        #         print('Input sentence:', input_sentence)
        #         print('Output sentence:', output_sentence)
        #         print('Decoded sentence:', decoded_sentence)
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
