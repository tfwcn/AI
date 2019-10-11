import tensorflow as tf
import numpy as np
import math
import os
import sys
import random

# 跟目录
ROOT_DIR = os.path.abspath("./")

# 导入Seq2SeqLoader
sys.path.append(ROOT_DIR)
from DataLoader.seq2seq_loader import Seq2SeqLoader


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
        # LSTM输出state_h, state_c，decoder_outputs=state_h
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
        # 输出序号
        self.dense = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.linear)
    
    def call(self, input_data, state_h, state_c, state_h2, state_c2):
        # shape(1,1)=>(1,1,units)
        input_data=self.embedding(input_data)
        # shape(1,1,units)
        # LSTM输出state_h, state_c，decoder_outputs=state_h
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
        self.char_to_int = None  # 字符转序号
        self.int_to_char = None  # 序号转字符
        self.max_length = 256  # 句子最大长度

    def build_model(self,latent_dim,num_tokens,char_to_int,int_to_char):
        """创建模型"""
        self.char_to_int = char_to_int  # 字符转序号
        self.int_to_char = int_to_char  # 序号转字符
        # ==================编码器=====================
        # 用于提取句子特征，返回最后时间点的特征数组与隐藏层状态
        self.encoder = Encoder(latent_dim, num_tokens)
        # ==================编码器 end=====================

        # ==================解码器=====================
        # 解码输入，以\t开头，用上一次输出的字符作为下一次的输入，训练时预测
        self.decoder = Decoder(latent_dim, num_tokens)
        # ==================解码器 end=====================

        # 编码					解码
        # \t	h	i	\n		\t	你	好	\n
        # LSTM					LSTM
        # 					    你	好	\n

        # 定义损失函数
        # self.loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.loss=tf.keras.losses.CategoricalCrossentropy(reduction='none')
        # 定义使用的梯度下降方式
        self.optimizer=tf.keras.optimizers.Adadelta(lr=1)

    @tf.function
    def encoder_setp(self, encoder_input_data_one):
        """一句话编码"""
        state_h, state_c, state_h2, state_c2 = self.encoder(encoder_input_data_one)
        return state_h, state_c, state_h2, state_c2

    @tf.function
    def decoder_setp(self, target_seq, state_h, state_c, state_h2, state_c2):
        """单个字解码"""
        output_tokens, h, c, h2, c2 = self.decoder(target_seq, state_h, state_c, state_h2, state_c2)
        return output_tokens, h, c, h2, c2

    
    # def train_one(self, encoder_input_data_one, decoder_input_data_one):
    #     """训练（单个句子）"""
    #     # Run training
    #     # 训练
    #     # encoder_input_data：输入要翻译的语句
    #     # decoder_input_data：输入解码器的结果\t开头
    #     # decoder_target_data：真正的翻译结果
    #     # print('encoder_input_data_one(句子长度)',encoder_input_data_one.shape)
    #     # print('decoder_input_data_one(句子长度)',decoder_input_data_one.shape)
    #     encoder_input_data_one=np.reshape(encoder_input_data_one,(1,encoder_input_data_one.shape[0]))
    #     decoder_input_data_one=np.reshape(decoder_input_data_one,(1,decoder_input_data_one.shape[0]))
    #     # print('encoder_input_data_one(1,句子长度)',encoder_input_data_one.shape)
    #     # print('decoder_input_data_one(1,句子长度)',decoder_input_data_one.shape)
    #     with tf.GradientTape() as tape:
    #         # print('训练',encoder_input_data_one)
    #         # 编码
    #         state_h, state_c, state_h2, state_c2 = self.encoder_setp(encoder_input_data_one)
    #         # 解码开始符号
    #         target_seq = np.array([[self.char_to_int['\t']]])
    #         # print('target_seq',target_seq.shape)

    #         # 循环解码
    #         loss = 0
    #         for t in range(1, decoder_input_data_one.shape[1]):
    #             # print('解码', t)
    #             # 解码
    #             output_tokens, h, c, h2, c2 = self.decoder_setp(target_seq, state_h, state_c, state_h2, state_c2)

    #             # 累计损失，是句子末尾空白符合时,跳出循环
    #             if decoder_input_data_one[:,t]==0:
    #                 break
    #             # 累计损失
    #             # shape(1,1),(1,num_tokens)
    #             loss += self.loss([decoder_input_data_one[:,t]], output_tokens)
    #             # print('loss', loss)
    #             # 当前字符，传递到下一次预测
    #             target_seq = np.reshape(decoder_input_data_one[:,t],(1,1))

    #             # Update states
    #             # 当前状态，传递到下一次预测
    #             state_h, state_c, state_h2, state_c2 = [h, c, h2, c2]
    #         print('开始更新权重，loss', loss)
    #     variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    #     # print('variables', variables)
    #     gradients = tape.gradient(loss, variables)
    #     self.optimizer.apply_gradients(zip(gradients, variables))
    #     print('更新权重完成')

    def train_one(self, encoder_input_data_one, decoder_input_data_one, decoder_target_data_one):
        """训练（单个句子）"""
        # Run training
        # 训练
        # encoder_input_data：输入要翻译的语句
        # decoder_input_data：输入解码器的结果\t开头
        # decoder_target_data：真正的翻译结果
        # print('encoder_input_data_one(句子长度)',encoder_input_data_one.shape)
        # print('decoder_input_data_one(句子长度)',decoder_input_data_one.shape)
        encoder_input_data_one=np.reshape(encoder_input_data_one,(1,encoder_input_data_one.shape[0]))
        decoder_input_data_one=np.reshape(decoder_input_data_one,(1,decoder_input_data_one.shape[0]))
        decoder_target_data_one=np.reshape(decoder_target_data_one,(1,decoder_target_data_one.shape[0]))
        # print('encoder_input_data_one(1,句子长度)',encoder_input_data_one.shape)
        # print('decoder_input_data_one(1,句子长度)',decoder_input_data_one.shape)
        with tf.GradientTape() as tape:
            # print('训练',encoder_input_data_one)
            # 编码
            state_h, state_c, state_h2, state_c2 = self.encoder_setp(encoder_input_data_one)
            # 解码
            output_tokens, h, c, h2, c2 = self.decoder_setp(decoder_input_data_one, state_h, state_c, state_h2, state_c2)
            
            loss = self.loss(decoder_target_data_one, output_tokens)
            print('开始更新权重，loss', loss)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        # print('variables', variables)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        print('更新权重完成')

    def save_model(self):
        # Save model
        # self.train_model.save_weights('s2s.h5')
        pass
    
    def predict(self, input_data_one):
        """识别（单个句子）"""
        input_data_one=np.reshape(input_data_one,(1,input_data_one.shape[0]))
        # print('识别',input_data_one)
        # 编码
        state_h, state_c, state_h2, state_c2 = self.encoder_setp(input_data_one)
        # 解码开始符号
        target_seq = np.array([[self.char_to_int['\t']]])

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        char_num = 0
        while not stop_condition:
            # output_tokens.shape(1,num_tokens)
            output_tokens, h, c, h2, c2 = self.decoder_setp(target_seq, state_h, state_c, state_h2, state_c2)

            # print('output_tokens', output_tokens)
            # 对应字符下标，把预测出的字符拼成字符串
            # Sample a token
            sampled_token_index = int(output_tokens.numpy()[0,0])
            # print('sampled_token_index',sampled_token_index)
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
            target_seq = np.array([[sampled_token_index]])

            # Update states
            # 当前状态，传递到下一次预测
            state_h, state_c, state_h2, state_c2 = [h, c, h2, c2]

        return decoded_sentence


def main():
    seq2seq_loader = Seq2SeqLoader()
    # 读取字库
    alphabet=seq2seq_loader.load_word('./RNN/word_all.txt')
    # 读取素材
    question_texts,answer_texts=seq2seq_loader.load_file('./RNN/ai.txt')
    
    # 转序号
    encoder_input_data,decoder_input_data,decoder_target_data,char_to_int,int_to_char=seq2seq_loader.create_index(alphabet,question_texts,answer_texts)

    my_robot = MyRobot()
    # 生成模型
    my_robot.build_model(128,seq2seq_loader.num_tokens,char_to_int,int_to_char)

    i = 0
    while i != len(encoder_input_data):
        i = 0
        # my_robot.train(my_robot.encoder_input_data,
        #                my_robot.decoder_input_data, my_robot.decoder_target_data)
        for (encoder_input_data_one, decoder_input_data_one, decoder_target_data_one) in zip(encoder_input_data, decoder_input_data, decoder_target_data):
            # 问题
            input_sentence = seq2seq_loader.indexs_to_text(encoder_input_data_one)
            # 机器人回答
            decoded_sentence = ''
            # 正确答案
            output_sentence = seq2seq_loader.indexs_to_text(decoder_target_data_one)
            # i += 1
            # # 训练
            # my_robot.train_one(encoder_input_data_one, decoder_input_data_one, decoder_target_data_one)
            # 答案有误时50%重复训练
            # while decoded_sentence != output_sentence and random.random() < 0.5:
            while decoded_sentence != output_sentence:
                i += 1
                # 训练
                my_robot.train_one(encoder_input_data_one, decoder_input_data_one, decoder_target_data_one)
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
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = my_robot.predict(input_seq)
        print('-')
        print('Input sentence:', question_texts[seq_index])
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
