import tensorflow as tf
import numpy as np
import os
import time


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

    def __init__(self, vocab_size, embedding_dim, units):
        '''
        vocab_size:字符数
        embedding_dim:字符编码特征数
        units:隐藏层神经元数量
        '''
        super(Decoder, self).__init__()
        #
        self.embedding1 = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # 注意力模块
        self.attention1 = BahdanauAttention(units)
        self.attention2 = BahdanauAttention(units)
        # GRU
        self.gru1 = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True, activation=tf.keras.activations.tanh, recurrent_initializer=tf.keras.initializers.GlorotUniform(), name='feature_gru1')
        self.gru2 = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True, activation=tf.keras.activations.tanh, recurrent_initializer=tf.keras.initializers.GlorotUniform(), name='feature_gru2')
        # 输出
        self.dense1 = tf.keras.layers.Dense(vocab_size, name='feature_dense1')

    def call(self, input_data, gru_state1=None, gru_state2=None):
        '''
        input_data:单步预测数据(None,1)
        gru_state1:上一步的状态(None,units)
        gru_state2:上一步的状态(None,units)
        '''
        # print('decoder', input_data)
        x = self.embedding1(input_data)
        # context_vector1, _ = self.attention1(
        #     gru_state1, x)
        # context_vector2, _ = self.attention2(
        #     gru_state2, x)
        # context_vector1 = tf.expand_dims(context_vector1, 1)
        # context_vector2 = tf.expand_dims(context_vector2, 1)
        # x = tf.concat([context_vector1, context_vector2, x], axis=-1)
        x, gru_state1 = self.gru1(x, initial_state=gru_state1)
        x, gru_state2 = self.gru2(x, initial_state=gru_state2)
        x = self.dense1(x)

        return x, gru_state1, gru_state2
        # return x, gru_state1


class Decoder2(tf.keras.Model):
    '''解码器'''

    def __init__(self, vocab_size, embedding_dim, units):
        '''
        vocab_size:字符数
        embedding_dim:字符编码特征数
        units:隐藏层神经元数量
        '''
        super(Decoder2, self).__init__()
        #
        self.embedding1 = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # GRU
        self.gru1 = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True, recurrent_initializer=tf.keras.initializers.GlorotUniform(), name='feature_gru1')
        # 输出
        self.dense1 = tf.keras.layers.Dense(vocab_size, name='feature_dense1')

    def call(self, input_data, gru_state1=None, gru_state2=None):
        '''
        input_data:单步预测数据(None,1)
        gru_state1:上一步的状态(None,units)
        gru_state2:上一步的状态(None,units)
        '''
        # print('decoder', input_data)
        x = self.embedding1(input_data)
        x, gru_state1 = self.gru1(x, initial_state=gru_state1)
        x = self.dense1(x)
        return x, gru_state1


class RobotModel():
    def __init__(self, vocab_size, embedding_dim, units, model_path):
        super().__init__()
        # 加载模型路径
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        # 建立模型
        self.build_model()
        # 加载模型
        self.load_model()

    def build_model(self):
        '''
        构建模型
        input_num：输入字符宽度
        '''
        self.model = Decoder2(self.vocab_size, self.embedding_dim, self.units)
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # 损失函数
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # 保存模型
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              decoder=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.model_path, max_to_keep=3)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ))
    def train_step(self, input_data, target_data):
        '''
        训练
        input_data:(batch_size, input_size) 
        target_data:(batch_size, target_size) 
        '''
        print('Tracing with train_step', type(input_data), type(target_data))
        print('Tracing with train_step', input_data.shape, target_data.shape)
        input_shape = tf.shape(input_data)
        # tf.print('input_shape', input_shape)
        gru_state1 = tf.zeros((input_shape[0],self.units))
        gru_state2 = tf.zeros((input_shape[0],self.units))
        loss = 0.0
        with tf.GradientTape() as tape:
            # 解码
            input_value = input_data
            predictions, gru_state1, gru_state2 = self.model(input_value, gru_state1, gru_state2)
            # tf.print('output', predictions.shape, gru_state1.shape, gru_state2.shape)
            loss = self.loss_object(y_true=target_data, y_pred=predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss


    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ))
    def train_step2(self, input_data, target_data):
        '''
        训练
        input_data:(batch_size, input_size) 
        target_data:(batch_size, target_size) 
        '''
        print('Tracing with train_step', type(input_data), type(target_data))
        print('Tracing with train_step', input_data.shape, target_data.shape)
        input_shape = tf.shape(input_data)
        # tf.print('input_shape', input_shape)
        gru_state1 = tf.zeros((input_shape[0],self.units))
        loss = 0.0
        with tf.GradientTape() as tape:
            # 解码
            input_value = input_data
            predictions, gru_state1 = self.model(input_value, gru_state1)
            loss = self.loss_object(y_true=target_data, y_pred=predictions)
        # print('loss', loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def fit(self, dataset, epochs, steps_per_epoch):
        '''训练'''
        for epoch in range(epochs):
            start_time = time.time()

            for (batch_n, (inp, target)) in enumerate(dataset.take(steps_per_epoch)):
                # print('batch data:', inp, target)
                loss = self.train_step2(inp, target)
                template = '\rEpoch {} Batch {} Loss {}'
                print(template.format(epoch+1, batch_n, loss), end='')

                if batch_n % 100 == 0:
                    template = 'Epoch {} Batch {} Loss {}'
                    print(template.format(epoch+1, batch_n, loss))

            # 每 5 个训练周期，保存（检查点）1 次模型
            if (epoch + 1) % 5 == 0:
                self.save_model()

            print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
            print('Time taken for 1 epoch {} sec\n'.format(
                time.time() - start_time))

        self.save_model()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
    ))
    def predict(self, input_data, output_size):
        '''
        预测(编译模式)
        input_data:(1, history_size)
        output_size:预测数量
        '''
        # print('Tracing with predict', type(input_data), type(output_size))
        # print('Tracing with predict', input_data.shape, output_size)
        predict_data = tf.TensorArray(
            dtype=tf.int32, size=output_size, dynamic_size=True)
        input_shape = tf.shape(input_data)
        gru_state1 = tf.zeros((input_shape[0],self.units))
        gru_state2 = tf.zeros((input_shape[0],self.units))
        # 编码
        input_value = input_data
        input_value, gru_state1, gru_state2 = self.model(input_value, gru_state1, gru_state2)
        input_value = tf.math.argmax(input_value, 2, output_type=tf.int32)

        # 低温度会生成更可预测的文本
        # 较高温度会生成更令人惊讶的文本
        # 可以通过试验以找到最好的设定
        temperature = 1.0

        # 解码
        for i in tf.range(output_size):
            # 解码
            input_value, gru_state1, gru_state2 = self.model(input_value, gru_state1, gru_state2)
            # input_value = tf.math.argmax(input_value, 2, output_type=tf.int32)
            # 删除批次的维度
            input_value = tf.squeeze(input_value, 0)

            # 用分类分布预测模型返回的字符
            input_value = input_value / temperature
            # 按概率随机取其中一个作为输入
            predicted_id = tf.random.categorical(
                input_value, num_samples=1)[-1, 0]

            # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
            input_value = tf.expand_dims([predicted_id], 0)
            input_value = tf.cast(input_value, dtype=tf.int32)

            # 记录预测值
            predict_data = predict_data.write(i, input_value)

        # 交换维度
        predict_data = predict_data.stack()
        predict_data = tf.transpose(predict_data, perm=[1, 0, 2])
        return predict_data

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
    ))
    def predict2(self, input_data, output_size):
        '''
        预测(编译模式)
        input_data:(1, history_size)
        output_size:预测数量
        '''
        # print('Tracing with predict', type(input_data), type(output_size))
        # print('Tracing with predict', input_data.shape, output_size)
        predict_data = tf.TensorArray(
            dtype=tf.int32, size=output_size, dynamic_size=True)
        input_shape = tf.shape(input_data)
        gru_state1 = tf.zeros((input_shape[0],self.units))
        # 编码
        input_value = input_data
        input_value, gru_state1 = self.model(input_value, gru_state1)
        input_value = tf.math.argmax(input_value, 2, output_type=tf.int32)
        # tf.print('input_value', input_value)

        # 低温度会生成更可预测的文本
        # 较高温度会生成更令人惊讶的文本
        # 可以通过试验以找到最好的设定
        temperature = 1.0

        # 解码
        for i in tf.range(output_size):
            # 解码
            input_value, gru_state1 = self.model(input_value, gru_state1)
            # input_value = tf.math.argmax(input_value, 2, output_type=tf.int32)
            # 删除批次的维度
            input_value = tf.squeeze(input_value, 0)

            # 用分类分布预测模型返回的字符
            input_value = input_value / temperature
            # 按概率随机取其中一个作为输入
            predicted_id = tf.random.categorical(
                input_value, num_samples=1)[-1, 0]

            # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
            input_value = tf.expand_dims([predicted_id], 0)
            input_value = tf.cast(input_value, dtype=tf.int32)

            # 记录预测值
            predict_data = predict_data.write(i, input_value)

        # 交换维度
        predict_data = predict_data.stack()
        predict_data = tf.transpose(predict_data, perm=[1, 0, 2])
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


def TestDemo():
    # path_to_file = tf.keras.utils.get_file(
    #     'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    path_to_file = './data/哈利波特全集 1_7部.txt'
    # 读取并为 py2 compat 解码
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # 文本长度是指文本中的字符个数
    print('Length of text: {} characters'.format(len(text)))

    # 文本中的非重复字符
    vocab = sorted(set(text))
    # word_all = open('./data/word_all.txt', 'rb').read().decode(encoding='utf-8')
    # word_all += '\n\r\t\0'
    # vocab = sorted(set(word_all))
    print('{} unique characters'.format(len(vocab)))

    # 创建从非重复字符到索引的映射
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    print('{')
    for char, _ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')

    # 显示文本首 13 个字符的整数映射
    print(
        '{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # 设定每个输入句子长度的最大值
    seq_length = 100
    examples_per_epoch = len(text)//seq_length

    # 创建训练样本 / 目标
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True).repeat()

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(
            input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(
            target_idx, repr(idx2char[target_idx])))

    # 批大小
    BATCH_SIZE = 64

    # 设定缓冲区大小，以重新排列数据集
    # （TF 数据被设计为可以处理可能是无限的序列，
    # 所以它不会试图在内存中重新排列整个序列。相反，
    # 它维持一个缓冲区，在缓冲区重新排列元素。）
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True).repeat()

    print('dataset', dataset)

    # 词集的长度
    vocab_size = len(vocab)

    # 嵌入的维度
    embedding_dim = 256

    # RNN 的单元数量
    rnn_units = 1024

    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    # 检查点保存至的目录
    checkpoint_dir = './data/robot_model'

    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape,
              "# (batch_size, sequence_length, vocab_size)")

    model.summary()

    sampled_indices = tf.random.categorical(
        example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print('sampled_indices', sampled_indices)

    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr(
        "".join(idx2char[sampled_indices])))

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape,
          " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)

    EPOCHS = 20

    # 用模型训练
    # history = model.fit(dataset, epochs=EPOCHS,
    #                     callbacks=[checkpoint_callback], steps_per_epoch=1000)

    # 自定义训练过程
    # 梯度下降方法
    optimizer = tf.keras.optimizers.Adam()

    # 单步训练方法
    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            predictions = model(inp)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    target, predictions, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    # 训练步骤
    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()

        # 在每个训练周期开始时，初始化隐藏状态
        # 隐藏状态最初为 None
        hidden = model.reset_states()
        print('dataset', len(dataset))
        for (batch_n, (inp, target)) in enumerate(dataset):
            loss = train_step(inp, target)

            if batch_n % 100 == 0:
                template = 'Epoch {} Batch {} Loss {}'
                print(template.format(epoch+1, batch_n, loss))

        # 每 5 个训练周期，保存（检查点）1 次模型
        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # 保存权重
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print('最后权重文件路径：', tf.train.latest_checkpoint(checkpoint_dir))

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    model.summary()

    def generate_text(model, start_string):
        # 评估步骤（用学习过的模型生成文本）

        # 要生成的字符个数
        num_generate = 2000

        # 将起始字符串转换为数字（向量化）
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # 空字符串用于存储结果
        text_generated = []

        # 低温度会生成更可预测的文本
        # 较高温度会生成更令人惊讶的文本
        # 可以通过试验以找到最好的设定
        temperature = 1.0

        # 这里批大小为 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # 删除批次的维度
            predictions = tf.squeeze(predictions, 0)

            # 用分类分布预测模型返回的字符
            predictions = predictions / temperature
            # 按概率随机取其中一个作为输入
            predicted_id = tf.random.categorical(
                predictions, num_samples=1)[-1, 0].numpy()

            # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    print(generate_text(model, start_string=u"一天"))


def TrainRobot():
    # path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    path_to_file = './data/哈利波特全集 1_7部.txt'
    # 读取并为 py2 compat 解码
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # 文本长度是指文本中的字符个数
    print ('Length of text: {} characters'.format(len(text)))
    
    # 文本中的非重复字符
    vocab = sorted(set(text))
    print ('{} unique characters'.format(len(vocab)))

    # 创建从非重复字符到索引的映射
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    # 设定每个输入句子长度的最大值
    seq_length = 200
    examples_per_epoch = len(text)//seq_length

    # 创建训练样本 / 目标
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True).repeat()

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # 批大小
    BATCH_SIZE = 32

    # 设定缓冲区大小，以重新排列数据集
    # （TF 数据被设计为可以处理可能是无限的序列，
    # 所以它不会试图在内存中重新排列整个序列。相反，
    # 它维持一个缓冲区，在缓冲区重新排列元素。） 
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

    # 词集的长度
    vocab_size = len(vocab)

    # 嵌入的维度
    embedding_dim = 256

    # RNN 的单元数量
    rnn_units = 1024

    model = RobotModel(vocab_size, embedding_dim, rnn_units, './data/robot_model')

    # 训练步骤
    EPOCHS = 10

    model.fit(dataset, EPOCHS, 100)

    # 要生成的字符个数
    num_generate = 1000

    start_string = '一天'

    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    print('input_eval',input_eval.shape)
    predict_value = model.predict2(input_eval, num_generate)
    print('predict_value',predict_value.shape)
    for i in range(predict_value.shape[1]):
        print(idx2char[predict_value[0,i,0]], end='')

def main():
    TrainRobot()
    # x = tf.random.uniform([10,3],
    #     minval=0,
    #     maxval=10,
    #     dtype=tf.dtypes.int32)
    # print('x', x)
    # y = tf.keras.layers.Embedding(10, 4)(x)
    # print('y', y.shape)


if __name__ == '__main__':
    main()
