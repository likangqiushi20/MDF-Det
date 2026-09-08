"""Content-driven semantic vehicle-prior network.

The model is fully convolutional and contains no coordinate channels or
absolute positional embeddings.  It is trained on native-resolution crops and
can be applied to an entire AOI at inference time.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L


@tf.keras.utils.register_keras_serializable(package="SPGFv2")
class GroupNormalization(L.Layer):
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        groups = min(self.groups, channels)
        while channels % groups:
            groups -= 1
        self.actual_groups = groups
        self.gamma = self.add_weight("gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight("beta", shape=(channels,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        channels = inputs.shape[-1]
        grouped = tf.reshape(inputs, [shape[0], shape[1], shape[2], self.actual_groups,
                                      channels // self.actual_groups])
        mean, variance = tf.nn.moments(grouped, axes=[1, 2, 4], keepdims=True)
        grouped = (grouped - mean) * tf.math.rsqrt(variance + self.epsilon)
        normalized = tf.reshape(grouped, shape)
        return normalized * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups, "epsilon": self.epsilon})
        return config


def _conv(x, filters, kernel=3, dilation=1):
    x = L.Conv2D(filters, kernel, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = GroupNormalization(groups=8)(x)
    return L.Activation("swish")(x)


def _residual(x, filters, stride=1):
    shortcut = x
    x = L.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = GroupNormalization(groups=8)(x)
    x = L.Activation("swish")(x)
    x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = GroupNormalization(groups=8)(x)
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = L.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = GroupNormalization(groups=8)(shortcut)
    return L.Activation("swish")(L.Add()([x, shortcut]))


def build_spgf_v4(base_filters=24):
    inputs = L.Input(shape=(None, None, 1), name="native_context")
    stem = _conv(inputs, base_filters)
    e1 = _residual(stem, base_filters)
    e2 = _residual(e1, base_filters * 2, stride=2)
    e3 = _residual(e2, base_filters * 4, stride=2)
    e4 = _residual(e3, base_filters * 8, stride=2)

    # ASPP preserves local texture while seeing road-scale context.
    branches = [_conv(e4, base_filters * 4, kernel=1)]
    branches.extend(_conv(e4, base_filters * 4, dilation=d) for d in (2, 4, 8))
    x = _conv(L.Concatenate()(branches), base_filters * 8, kernel=1)

    x = L.UpSampling2D(2, interpolation="bilinear")(x)
    x = _conv(L.Concatenate()([x, e3]), base_filters * 4)
    x = L.UpSampling2D(2, interpolation="bilinear")(x)
    x = _conv(L.Concatenate()([x, e2]), base_filters * 2)
    x = L.UpSampling2D(2, interpolation="bilinear")(x)
    x = _conv(L.Concatenate()([x, e1]), base_filters)
    outputs = L.Conv2D(2, 1, activation="sigmoid", name="semantic_prior")(x)
    return tf.keras.Model(inputs, outputs, name="spgf_v4")


# Backward-compatible alias for checkpoints and local scripts produced while
# the revised module was still called "semantic prior V2".
build_semantic_prior_v2 = build_spgf_v4


def _weighted_focal(target, probability, weight, alpha=0.75, gamma=2.0):
    probability = tf.clip_by_value(probability, 1e-6, 1.0 - 1e-6)
    pt = target * probability + (1.0 - target) * (1.0 - probability)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt) * weight
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(weight), 1.0)


def _weighted_dice(target, probability, weight):
    intersection = tf.reduce_sum(weight * target * probability, axis=(1, 2))
    denominator = tf.reduce_sum(weight * (target + probability), axis=(1, 2))
    return 1.0 - tf.reduce_mean((2.0 * intersection + 1.0) / (denominator + 1.0))


@tf.keras.utils.register_keras_serializable(package="SPGFv2")
def semantic_prior_v2_loss(y_true, y_pred):
    # y_true = [access target, motion target, access weight, motion weight]
    access_t, motion_t = y_true[..., 0], y_true[..., 1]
    access_w, motion_w = y_true[..., 2], y_true[..., 3]
    access_p, motion_p = y_pred[..., 0], y_pred[..., 1]
    access = _weighted_focal(access_t, access_p, access_w) + 0.5 * _weighted_dice(access_t, access_p, access_w)
    motion = _weighted_focal(motion_t, motion_p, motion_w) + 0.5 * _weighted_dice(motion_t, motion_p, motion_w)
    return access + 0.7 * motion


@tf.keras.utils.register_keras_serializable(package="SPGFv2")
def access_mae(y_true, y_pred):
    weight = y_true[..., 2]
    error = tf.abs(y_true[..., 0] - y_pred[..., 0]) * weight
    return tf.reduce_sum(error) / tf.maximum(tf.reduce_sum(weight), 1.0)


@tf.keras.utils.register_keras_serializable(package="SPGFv2")
def motion_mae(y_true, y_pred):
    weight = y_true[..., 3]
    error = tf.abs(y_true[..., 1] - y_pred[..., 1]) * weight
    return tf.reduce_sum(error) / tf.maximum(tf.reduce_sum(weight), 1.0)
