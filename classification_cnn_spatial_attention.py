"""
classification_cnn_spatial_attention.py
带空间注意力的WAMI二分类CNN
输入: (4, 21, 21)  channels_first
输出: 2类 softmax
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPool2D,
    Flatten, Dense, Multiply, Concatenate
)
from tensorflow.keras.models import Model


def spatial_attention(input_feature, kernel_size=5):
    """
    空间注意力模块 (channels_first)
    """
    avg_pool = tf.reduce_mean(input_feature, axis=1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=1, keepdims=True)
    concat = Concatenate(axis=1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size, padding='same', data_format='channels_first',
                       activation='sigmoid')(concat)
    return Multiply()([input_feature, attention])


def build_classification_cnn_spatial_attention(input_shape=(4, 21, 21)):
    inputs = Input(shape=input_shape, name='input_patch')

    # Conv Block 1
    x = Conv2D(32, 3, padding='same', data_format='channels_first')(inputs)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=(2, 2), data_format='channels_first')(x)  # (32, 10, 10)

    # Conv Block 2
    x = Conv2D(64, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # Spatial Attention
    x = spatial_attention(x, kernel_size=5)

    # Classifier
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='classification_spatial_attention')
    return model