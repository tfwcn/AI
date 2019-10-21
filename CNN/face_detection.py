import tensorflow as tf
import os
import sys
import cv2 as cv
import numpy as np
from absl import flags, app


# tf.compat.v1.enable_eager_execution()

print("TensorFlow version: {}".format(tf.version.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

# 根目录
ROOT_DIR = os.path.abspath("../")

# 导入WiderFaceLoader
sys.path.append(ROOT_DIR)
from DataLoader.wider_face_loader import WiderFaceLoader

# 输入参数
FLAGS = flags.FLAGS
flags.DEFINE_string('label_path', None, '标签路径')
flags.DEFINE_string('image_path', None, '图片路径')

class Encoder(tf.keras.Model):
    """编码器"""

    def __init__(self):
        super(Encoder, self).__init__()
        # 特征提取
        # self.feature_conv1 = tf.keras.layers.Conv2D(32, (7, 7), strides=(2, 2), padding='valid', name='feature_conv1',
        #                                             kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        # self.feature_conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='feature_conv2',
        #                                             kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        # self.feature_conv3 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='valid', name='feature_conv3',
        #                                             kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        # self.feature_conv4 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', name='feature_conv4',
        #                                             kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)
        # self.feature_conv5 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='feature_conv5',
        #                                             kernel_initializer=tf.keras.initializers.he_normal(), activation=tf.keras.activations.tanh)

        # 使用ResNet50做特征提取
        self.feature_inceptionV3 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        # 减少维度
        self.feature_conv = tf.keras.layers.Conv2D(16, (3, 3), padding='same', name='feature_conv',
                                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                                    activation=tf.keras.activations.tanh)
        # 二维数据转一维
        self.feature_flatten1 = tf.keras.layers.Flatten(
            name='feature_flatten1')

        # 全连接层1，长度128
        self.feature_dense1 = tf.keras.layers.Dense(
            128, activation=tf.keras.activations.sigmoid, name='feature_dense1')

    def call(self, input_data):
        # x = self.feature_conv1(input_data)
        # x = self.feature_conv2(x)
        # x = self.feature_conv3(x)
        # x = self.feature_conv4(x)
        # x = self.feature_conv5(x)
        x = tf.image.resize(input_data, size=(512, 512))
        x = self.feature_inceptionV3(x)
        x = self.feature_conv(x)
        x = tf.image.resize(x, size=(40, 40))
        x = self.feature_flatten1(x)
        x = self.feature_dense1(x)
        return x


class BahdanauAttention(tf.keras.Model):
    """记忆模块"""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, name='feature_denseW1')
        self.W2 = tf.keras.layers.Dense(units, name='feature_denseW2')
        self.V = tf.keras.layers.Dense(1, name='feature_denseV')

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """解码器"""

    def __init__(self, class_num):
        super(Decoder, self).__init__()
        # GRU
        self.gru1 = tf.keras.layers.GRU(128, return_sequences=True,
                                        return_state=True, activation=tf.keras.activations.tanh, name='feature_gru')
        # 可输出门
        self.dense_door = tf.keras.layers.Dense(
            128, activation=tf.keras.activations.sigmoid, name='feature_dense_door')
        # 输出类别（起始标识，结束标识，类别）
        self.dense_class = tf.keras.layers.Dense(
            class_num, activation=tf.keras.activations.softmax, name='feature_dense_class')
        # 输出类别（x，y，w, h）
        self.dense_box = tf.keras.layers.Dense(
            4, activation=tf.keras.activations.linear, name='feature_dense_box')

        # used for attention
        self.attention = BahdanauAttention(128)

    def call(self, input_class, input_box, input_state, encoder_output):
        x = input_all = tf.concat([input_class, input_box], axis=2)
        context_vector, attention_weights = self.attention(
            input_state, encoder_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, hidden = self.gru1(x, initial_state=input_state)
        x *= self.dense_door(input_all)
        x_classes = self.dense_class(x)
        x_boxs = self.dense_box(x)

        return x_classes, x_boxs, hidden, attention_weights


# 标签
class_indexes = {'<pad>': 0, '<begin>': 1, '<end>': 2, 'person': 3}
class_names = {0: '<pad>', 1: '<begin>', 2: '<end>', 3: 'person'}
class_num = len(class_indexes)

# 定义损失函数（交叉熵）
loss_class_func = tf.keras.losses.CategoricalCrossentropy()
loss_box_func = tf.keras.losses.MeanSquaredError()
# 定义使用的梯度下降方式
optimizer = tf.keras.optimizers.Adadelta(lr=0.001)

# 编码器
print('编码器')
encoder = Encoder()

# 解码器
print('解码器')
decoder = Decoder(class_num)

checkpoint_dir = './data'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
# 加载权重
print('加载权重:', checkpoint_dir)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


@tf.function(input_signature=(
    tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
    tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
))
def train_step(input_data, label_classes, label_boxs):
    """训练"""
    print('Tracing with train_step', type(input_data),
          type(label_classes), type(label_boxs))
    print('Tracing with train_step', input_data.shape,
          label_classes.shape, label_boxs.shape)
    loss_class_sum = 0.0
    loss_box_sum = 0.0
    with tf.GradientTape() as tape:
        # 编码
        input_data = tf.expand_dims(input_data, 0)
        encoder_output = encoder(input_data)
        label_class_onehot = tf.one_hot(
            label_classes[0], class_num, dtype=tf.float32)
        input_class = tf.expand_dims(tf.expand_dims(label_class_onehot, 0), 0)
        input_box = tf.expand_dims(tf.expand_dims(label_boxs[0], 0), 0)
        input_state = encoder_output

        # 解码
        for j in tf.range(1, tf.shape(label_classes)[0]):
            # 正确值
            label_class_onehot = tf.one_hot(
                label_classes[j], class_num, dtype=tf.float32)
            true_label_class = tf.expand_dims(
                tf.expand_dims(label_class_onehot, 0), 0)
            true_label_box = tf.expand_dims(
                tf.expand_dims(label_boxs[j], 0), 0)
            # 解码
            decoder_output_class, decoder_output_box, decoder_output_state, _ = decoder(
                input_class, input_box, input_state, encoder_output)
            # 计算损失
            loss_class = loss_class_func(
                y_true=true_label_class, y_pred=decoder_output_class)
            loss_box = loss_box_func(
                y_true=true_label_box, y_pred=decoder_output_box)
            loss_class_sum += loss_class
            loss_box_sum += loss_box
            # 下一个输入
            input_class = true_label_class
            input_box = true_label_box
            input_state = decoder_output_state

    tf.print('Action with loss_class_sum', loss_class_sum,
             'Action with loss_box_sum', loss_box_sum, end='\r')
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient([loss_class_sum, loss_box_sum], variables)
    optimizer.apply_gradients(zip(gradients, variables))


def predict(input_data):
    """识别"""
    result_classes = []
    result_boxes = []
    label_class = tf.constant(0, dtype=tf.int32)
    label_box = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    # 编码
    input_data = tf.expand_dims(input_data, 0)
    encoder_output = encoder(input_data)
    label_class_onehot = tf.one_hot(label_class, class_num, dtype=tf.float32)
    input_class = tf.expand_dims(tf.expand_dims(label_class_onehot, 0), 0)
    input_box = tf.expand_dims(tf.expand_dims(label_box, 0), 0)
    input_state = encoder_output

    # 解码
    for _ in tf.range(500):
        # 解码
        decoder_output_class, decoder_output_box, decoder_output_state, _ = decoder(
            input_class, input_box, input_state, encoder_output)
        # 下一个输入
        input_class = decoder_output_class
        input_box = decoder_output_box
        input_state = decoder_output_state

        predicted_id = tf.math.argmax(decoder_output_class[0,0]).numpy()
        if predicted_id == class_indexes['<end>']:
            return result_classes, result_boxes
        result_classes.append(predicted_id)
        result_boxes.append(decoder_output_box[0,0].numpy().tolist())

    return result_classes, result_boxes


def get_bacth_data(label_data, img_width, img_height):
    """解析训练数据"""
    label_classes = []
    label_boxs = []
    # 开始
    label_class = class_indexes['<begin>']
    label_classes.append(label_class)
    label_box = [0, 0, 0, 0]
    label_boxs.append(label_box)
    # 标签
    for i in range(len(label_data)):
        label_class = class_indexes['person']
        label_classes.append(label_class)
        label_box = [(label_data[i][0]+label_data[i][2]*0.5)/img_width, (label_data[i][1]+label_data[i][3]*0.5)/img_height,
                     label_data[i][2]/img_width, label_data[i][3]/img_height]
        label_boxs.append(label_box)
    # 结束
    label_class = class_indexes['<end>']
    label_classes.append(label_class)
    label_box = [0, 0, 0, 0]
    label_boxs.append(label_box)

    label_classes = np.array(label_classes, dtype=np.int32)
    label_boxs = np.array(label_boxs, dtype=np.float32)
    # print('label_data', len(label_data))
    # print('label_classes', label_classes.shape)
    # print('label_boxs', label_boxs.shape)
    return label_classes, label_boxs


def main(argv):
    # 加载数据
    wider_face_loader = WiderFaceLoader()
    # train_data,图片路径list
    # train_label(素材数量,目标框数量,4)
    train_data, train_label = wider_face_loader.load(
        FLAGS.label_path, FLAGS.image_path)
    # print('train_data', train_data[2])
    # print('train_label', train_label[2])
    # 读取图片
    img = cv.imdecode(np.fromfile(train_data[2], dtype=np.uint8), -1)
    # print('img',type(img))
    print('img', img.shape)
    # 显示图片
    cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('input_image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print('开始训练')
    # 转换成tensor
    train_data = tf.constant(train_data, dtype=tf.string)

    # 下标，用于随机顺序
    train_index = range(train_data.shape[0])
    for _ in range(10):
        # 随机下标
        train_index = tf.random.shuffle(train_index)
        for i in train_index:
        # for i in range(10):
            img = tf.io.read_file(train_data[i])
            img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
            img = tf.divide(img, 255.0)
            label_classes, label_boxs = get_bacth_data(
                train_label[i], img.shape[1], img.shape[0])
            img = tf.constant(img, dtype=tf.float32)
            label_classes = tf.constant(label_classes, dtype=tf.int32)
            label_boxs = tf.constant(label_boxs, dtype=tf.float32)
            train_step(img, label_classes, label_boxs)

        # 保存权重
        print('\n保存权重')
        checkpoint.save(file_prefix=checkpoint_prefix)

    # 识别
    print('开始识别')
    predict_path = os.path.join(FLAGS.image_path, '1--Handshaking/1_Handshaking_Handshaking_1_164.jpg')
    predict_data = tf.constant(predict_path, dtype=tf.string)
    img = tf.io.read_file(predict_data)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.divide(img, 255.0)
    result_classes, result_boxes = predict(img)
    print('result_classes', result_classes)
    print('result_boxes', result_boxes)
    # 读取图片
    img = img.numpy()
    img = img * 255
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    print('img', img.shape)
    # 画框
    for i in range(len(result_boxes)):
        box_x1 = (result_boxes[i][0]-result_boxes[i][2]/2)*img.shape[1]
        box_y1 = (result_boxes[i][1]-result_boxes[i][3]/2)*img.shape[0]
        box_x2 = (result_boxes[i][0]+result_boxes[i][2]/2)*img.shape[1]
        box_y2 = (result_boxes[i][1]+result_boxes[i][3]/2)*img.shape[0]
        cv.rectangle(img,(int(box_x1),int(box_y1)),(int(box_x2),int(box_y2)),(0,255,0),3)
        if (box_y1 > 10):
            cv.putText(img, class_names[result_classes[i]], (int(box_x1),int(box_y1-6)), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0) )
        else:
            cv.putText(img, class_names[result_classes[i]], (int(box_x1),int(box_y1+15)), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0) )
    # 显示图片
    cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('input_image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    flags.mark_flag_as_required('label_path')
    flags.mark_flag_as_required('image_path')
    app.run(main)
