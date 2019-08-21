import numpy as np

class Seq2SeqLoader():
    """读取问答素材
    返回：问题，答案
    """
    def __init__(self):
        self.word_all = ''  # 中文字符集
        self.alphabet = ''  # 字符集
        self.num_tokens = 0  # 字符数
        self.question_texts = []  # 问题集合
        self.answer_texts = []  # 答案集合
        self.max_length = 256  # 句子最大长度
        self.num_samples = 0  # 素材数
        self.word_file_path = ''  # 字库文件路径
        self.encoder_input_data = None  # 训练时，编码器输入，(数据集长度，句子最大长度，1：字符序号)
        self.decoder_input_data = None  # 训练时，解码器输入，(数据集长度，句子最大长度，1：字符序号)
        self.decoder_target_data = None  # 训练时，解码器输出，(数据集长度，句子最大长度，1：字符序号)

    def load_word(self,word_file_path):
        """加载字库"""
        print('正在加载字库...')
        self.word_file_path=word_file_path
        with open(word_file_path, 'r', encoding='UTF-8') as word_file:
            self.word_all = word_file.read()  # 两万多个字符
            self.alphabet = '\0\t\n'  # \0：填充字符，\t：开头标志，\n：结束标志
            self.alphabet += self.word_all
            word_file.close()
        # 字典字符数量
        self.num_tokens = len(self.alphabet)
        print('字符数：', self.num_tokens)
        print('加载字库完成!')
        return self.alphabet

    def add_word(self, char):
        """新增字符"""
        f2 = open(self.word_file_path, 'w', encoding='utf-8')
        f2.truncate()  # 清空文件
        self.word_all += char
        self.alphabet = '\0\t\n'  # \0：填充字符，\t：开头标志，\n：结束标志
        self.alphabet += self.word_all
        f2.write(self.word_all)
        f2.close()

    def load_file(self,label_file_path):
        """加载训练素材"""
        print('正在加载训练素材...')
        # 训练数据集
        with open(label_file_path, 'r', encoding='UTF-8') as train_file:
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
            # print('senterce', senterce.split('\t'))
            question_text, answer_text = senterce.split('\t')
            # \t 作为开头标识
            # \n 作为结尾标识
            question_text = '\t' + question_text + '\n'
            answer_text = '\t' + answer_text + '\n'
            self.question_texts.append(question_text)
            self.answer_texts.append(answer_text)
        # print('question_texts', question_texts)
        # print('answer_texts', answer_texts)

        # 素材数量
        self.num_samples = len(self.question_texts)
        print('加载训练素材完成!')
        print('素材数：', self.num_samples)
        return self.question_texts,self.answer_texts

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


    def create_index(self,alphabet,question_texts,answer_texts):
        """训练素材字符转序号"""
        # 字符与序号对应的字典
        self.char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        self.int_to_char = dict((i, c) for i, c in enumerate(alphabet))
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
        for i, (input_text, target_text) in enumerate(zip(question_texts, answer_texts)):
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
        return self.encoder_input_data,self.decoder_input_data,self.decoder_target_data,self.char_to_int,self.int_to_char


def main():
    seq2seq_loader = Seq2SeqLoader()
    # 读取字库
    alphabet=seq2seq_loader.load_word('../RNN/word_all.txt')
    print('alphabet',alphabet)
    # 读取素材
    question_texts,answer_texts=seq2seq_loader.load_file(u'../RNN/ai.txt')
    print('question_texts',question_texts[0])
    print('answer_texts',answer_texts[0])
    # 转序号
    encoder_input_data,decoder_input_data,decoder_target_data,char_to_int,int_to_char=seq2seq_loader.create_index(alphabet,question_texts,answer_texts)
    print('encoder_input_data',encoder_input_data[0])
    print('decoder_input_data',decoder_input_data[0])
    print('decoder_target_data',decoder_target_data[0])
    # print('char_to_int',seq2seq_loader.char_to_int)
    # print('int_to_char',seq2seq_loader.int_to_char)


if __name__ == '__main__':
    main()