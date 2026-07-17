import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Multiply, Reshape
from tensorflow.keras.models import Model

def spatial_attention(input_feature):
    """
    输入: feature map (batch, C, H, W)  channels_first
    输出: 与输入相同尺寸，经过空间注意力加权后的特征图
    """
    # 沿通道维度计算均值和最大值
    avg_pool = tf.reduce_mean(input_feature, axis=1, keepdims=True)   # (batch, 1, H, W)
    max_pool = tf.reduce_max(input_feature, axis=1, keepdims=True)     # (batch, 1, H, W)
    concat = tf.concat([avg_pool, max_pool], axis=1)                   # (batch, 2, H, W)
    # 7x7卷积，输出单通道注意力图
    attention = Conv2D(1, 7, padding='same', data_format='channels_first', activation='sigmoid')(concat)
    # 加权
    return Multiply()([input_feature, attention])

def build_regression_cnn_spatial_attention(input_shape=(4,45,45)):
    """
    构建带空间注意力的回归CNN
    输入: (4,45,45)  输出: (225,) 对应15x15热力图
    """
    inputs = Input(shape=input_shape, name='image_patch')
    # 卷积块1
    x = Conv2D(32, 3, padding='same', data_format='channels_first')(inputs)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(2, data_format='channels_first')(x)   # (32,22,22)

    # 卷积块2
    x = Conv2D(64, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(2, data_format='channels_first')(x)   # (64,11,11)

    # 插入空间注意力（特征图尺寸 11x11）
    x = spatial_attention(x)

    # 卷积块3
    x = Conv2D(128, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # 全连接输出
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Dense(225, activation='sigmoid')(x)   # 15x15 热力图

    model = Model(inputs=inputs, outputs=outputs, name='regression_spatial_attention')
    return model