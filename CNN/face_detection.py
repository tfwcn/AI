import tensorflow as tf

class FaceDetection(tf.keras.Model):
    """人脸检测类"""
    def __init__(self):
        # 卷积
        self.feature_conv1=tf.keras.layers.Conv2D(32, (7,7), strides=(2,2), padding='same', name='feature_conv1',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_bn1=tf.keras.layers.BatchNormalization(name='feature_bn1')
        self.feature_act1=tf.keras.layers.Activation(tf.keras.activations.relu, name='feature_act1')
        # 卷积
        self.feature_conv2=tf.keras.layers.Conv2D(64, (3,3), padding='same', name='feature_conv2',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_bn2=tf.keras.layers.BatchNormalization(name='feature_bn2')
        self.feature_act2=tf.keras.layers.Activation(tf.keras.activations.relu, name='feature_act2')
        # 卷积
        self.feature_conv3=tf.keras.layers.Conv2D(128, (1,1), padding='same', name='feature_conv3',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_bn3=tf.keras.layers.BatchNormalization(name='feature_bn3')
        self.feature_act3=tf.keras.layers.Activation(tf.keras.activations.relu, name='feature_act3')
        # 输入源缩小一半
        self.feature_pool1=tf.keras.layers.MaxPool2D((2,2), name='feature_pool1')
        # 取8像素区域
        self.feature_pool2=tf.keras.layers.MaxPool2D((8,8), name='feature_pool2')
        # 卷积,分类是否前景
        self.feature_anchor2_conv1=tf.keras.layers.Conv2D(2, (1,1), padding='same', name='feature_anchor2_conv1',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_anchor2_bn1=tf.keras.layers.BatchNormalization(name='feature_anchor2_bn1')
        self.feature_anchor2_act1=tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='feature_anchor2_act1')
        # 卷积,框坐标回归
        self.feature_anchor2_conv2=tf.keras.layers.Conv2D(4, (1,1), padding='same', name='feature_anchor2_conv2',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_anchor2_bn2=tf.keras.layers.BatchNormalization(name='feature_anchor2_bn2')
        self.feature_anchor2_act2=tf.keras.layers.Activation(tf.keras.activations.linear, name='feature_anchor2_act2')
        # 取12像素区域
        self.feature_pool3=tf.keras.layers.MaxPool2D((12,12), name='feature_pool3')
        # 卷积,分类是否前景
        self.feature_anchor3_conv1=tf.keras.layers.Conv2D(2, (1,1), padding='same', name='feature_anchor3_conv1',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_anchor3_bn1=tf.keras.layers.BatchNormalization(name='feature_anchor3_bn1')
        self.feature_anchor3_act1=tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='feature_anchor3_act1')
        # 卷积,框坐标回归
        self.feature_anchor3_conv2=tf.keras.layers.Conv2D(4, (1,1), padding='same', name='feature_anchor3_conv2',
                                                 kernel_initializer=tf.keras.initializers.he_normal)
        self.feature_anchor3_bn2=tf.keras.layers.BatchNormalization(name='feature_anchor3_bn2')
        self.feature_anchor3_act2=tf.keras.layers.Activation(tf.keras.activations.linear, name='feature_anchor3_act2')
        # 取24像素区域
        self.feature_pool4=tf.keras.layers.MaxPool2D((24,24), name='feature_pool4')
        # 取48像素区域
        self.feature_pool5=tf.keras.layers.MaxPool2D((48,48), name='feature_pool5')
        # 取64像素区域
        self.feature_pool6=tf.keras.layers.MaxPool2D((64,64), name='feature_pool6')
        # 取128像素区域
        self.feature_pool7=tf.keras.layers.MaxPool2D((128,128), name='feature_pool7')

    def call(self, input_data):
        # 提取特征
        x = self.feature_conv1(input_data)
        x = self.feature_bn1(x)
        x = self.feature_act1(x)
        x = self.feature_conv2(input_data)
        x = self.feature_bn2(x)
        x = self.feature_act2(x)
        x = self.feature_conv3(input_data)
        x = self.feature_bn3(x)
        x = self.feature_act3(x)
        x = self.feature_pool1(x)
        # 生成候选区域
        C2 = self.feature_pool2(x)
        C2_foreground = self.feature_anchor2_conv1(C2)
        C2_foreground = self.feature_anchor2_bn1(C2_foreground)
        C2_foreground = self.feature_anchor2_act1(C2_foreground)
        C2_box = self.feature_anchor2_conv2(C2)
        C2_box = self.feature_anchor2_bn2(C2_box)
        C2_box = self.feature_anchor2_act2(C2_box)

        C3 = self.feature_pool2(x)
        C3_foreground = self.feature_anchor2_conv1(C3)
        C3_foreground = self.feature_anchor2_bn1(C3_foreground)
        C3_foreground = self.feature_anchor2_act1(C3_foreground)
        C3_box = self.feature_anchor2_conv2(C3)
        C3_box = self.feature_anchor2_bn2(C3_box)
        C3_box = self.feature_anchor2_act2(C3_box)

        return (C2_foreground,C3_foreground),(C2_box,C3_box)