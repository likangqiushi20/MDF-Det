import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

def build_scene_prior_net(input_shape=(256,256,1), num_filters=32):
    inputs = Input(shape=input_shape)
    # Encoder
    conv1 = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(num_filters, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(num_filters*2, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(num_filters*2, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(num_filters*4, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(num_filters*4, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(num_filters*8, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(num_filters*8, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    # Bottleneck
    conv5 = Conv2D(num_filters*16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(num_filters*16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Decoder
    up6 = Conv2D(num_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(num_filters*8, 3, activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(num_filters*8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(num_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(num_filters*4, 3, activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(num_filters*4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(num_filters*2, 2, activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(num_filters*2, 3, activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(num_filters*2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(num_filters, 2, activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(num_filters, 3, activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(num_filters, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid', name='prior_map')(conv9)
    model = Model(inputs=inputs, outputs=outputs)
    return model