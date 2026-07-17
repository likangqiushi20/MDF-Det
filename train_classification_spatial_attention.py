"""
train_classification_spatial_attention_stream.py
功能：使用流式生成器训练带空间注意力的二分类CNN
网络结构：在原论文基础上，在第二个卷积块后插入空间注意力模块
参数：正600k，负3M，验证正6k负30k，SGD(lr=0.01,momentum=0.9,decay=1e-4)，60 epochs，batch_size=256
输出：saved_model.model (Keras格式) 和 saved_image_norm.model (pickle)
"""

import numpy as np
import tensorflow as tf
import hdf5storage
import glob
import pickle
import os
from tqdm import tqdm
import time

# ---------------------------- 空间注意力模块 ----------------------------
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPool2D,
    Flatten, Dense, Multiply, Concatenate, Input
)
from tensorflow.keras.models import Model

def spatial_attention(input_feature, kernel_size=5):
    """
    空间注意力模块 (channels_first)
    输入: (batch, C, H, W)
    输出: 加权后的特征图
    """
    avg_pool = tf.reduce_mean(input_feature, axis=1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=1, keepdims=True)
    concat = Concatenate(axis=1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size, padding='same', data_format='channels_first',
                       activation='sigmoid')(concat)
    return Multiply()([input_feature, attention])

def build_spatial_attention_model(input_shape=(4, 21, 21)):
    """构建带空间注意力的二分类CNN"""
    inputs = Input(shape=input_shape)

    # 卷积块1
    x = Conv2D(32, 3, padding='same', data_format='channels_first')(inputs)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=(2, 2), data_format='channels_first')(x)

    # 卷积块2
    x = Conv2D(64, 3, padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # 插入空间注意力
    x = spatial_attention(x, kernel_size=5)

    # 分类头
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ---------------------------- 训练主程序 ----------------------------
def main():
    start_time = time.time()

    # GPU配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ 已启用GPU显存动态增长，共 {len(gpus)} 个GPU可用")
        except RuntimeError as e:
            print(f"GPU配置出错: {e}")

    # 训练参数
    numPositive = 600000
    numNegative = 3000000
    numPositive_test = 6000
    numNegative_test = 30000
    batch_size = 256
    epochs = 60

    print("="*60)
    print("二分类CNN训练 - 空间注意力 + 流式加载")
    print("="*60)
    print(f"目标训练集: 正 {numPositive:,}, 负 {numNegative:,}")
    print(f"目标验证集: 正 {numPositive_test:,}, 负 {numNegative_test:,}")
    print(f"批大小: {batch_size}, 训练轮次: {epochs}")

    # 1. 扫描样本文件
    sample_folder = r"D:\Lkqs\sys\code\training_samples"  # 请根据实际情况修改
    print("\n1. 扫描样本文件...")
    file_paths = glob.glob(os.path.join(sample_folder, "wpafb_samples_frame_*.mat"))
    if not file_paths:
        print(f"错误：在 {sample_folder} 中未找到.mat文件")
        return
    print(f"找到 {len(file_paths)} 个.mat文件")

    total_pos = 0
    total_neg = 0
    for file_path in tqdm(file_paths, desc="扫描文件"):
        try:
            mat = hdf5storage.loadmat(file_path)
            if 'positive_dataset' in mat:
                pos_data = mat['positive_dataset']
                if pos_data.size > 0 and len(pos_data.shape) == 4:
                    total_pos += pos_data.shape[0]
            if 'negative_dataset' in mat:
                neg_data = mat['negative_dataset']
                if neg_data.size > 0 and len(neg_data.shape) == 4:
                    total_neg += neg_data.shape[0]
        except Exception as e:
            print(f"警告：扫描文件 {os.path.basename(file_path)} 时出错: {e}")
            continue

    print(f"总正样本数: {total_pos:,}")
    print(f"总负样本数: {total_neg:,}")

    # 调整目标样本数
    if total_pos < numPositive:
        print(f"警告：需要 {numPositive:,} 正样本，但只有 {total_pos:,} 可用")
        numPositive = total_pos
        numNegative = min(numNegative, total_neg, numPositive * 5)
    if total_neg < numNegative:
        print(f"警告：需要 {numNegative:,} 负样本，但只有 {total_neg:,} 可用")
        numNegative = total_neg
        numPositive = min(numPositive, total_pos, numNegative // 5)

    if total_pos < numPositive_test or total_neg < numNegative_test:
        print("错误：验证集所需样本不足")
        return

    print(f"\n最终训练参数:")
    print(f"训练集: 正 {numPositive:,}, 负 {numNegative:,}")
    print(f"验证集: 正 {numPositive_test:,}, 负 {numNegative_test:,}")

    # 2. 计算整体均值
    print("\n2. 计算整体均值...")
    save_dir = "./Models/BinaryClassification_SpatialAttention"
    os.makedirs(save_dir, exist_ok=True)
    mean_path = os.path.join(save_dir, "saved_image_norm.model")

    if os.path.exists(mean_path):
        with open(mean_path, "rb") as f:
            mean_nchw = pickle.load(f)
        print(f"✓ 已加载预计算均值: {mean_nchw}")
    else:
        sum_pixels = np.zeros(4, dtype=np.float64)
        count_pixels = 0
        for file_path in tqdm(file_paths, desc="计算均值"):
            try:
                mat = hdf5storage.loadmat(file_path)
                if 'positive_dataset' in mat:
                    pos_data = mat['positive_dataset']
                    if pos_data.size > 0:
                        sum_pixels += np.sum(pos_data, axis=(0,2,3), dtype=np.float64)
                        count_pixels += pos_data.shape[0] * 21 * 21
                if 'negative_dataset' in mat:
                    neg_data = mat['negative_dataset']
                    if neg_data.size > 0:
                        sum_pixels += np.sum(neg_data, axis=(0,2,3), dtype=np.float64)
                        count_pixels += neg_data.shape[0] * 21 * 21
            except:
                continue
        mean_nchw = (sum_pixels / count_pixels).astype(np.float32)
        with open(mean_path, "wb") as f:
            pickle.dump(mean_nchw, f)
        print(f"✓ 均值已保存: {mean_nchw}")

    # 3. 数据生成器
    def data_generator(file_paths, target_pos, target_neg, batch_size, mean, is_training=True):
        num_files = len(file_paths)
        pos_per_file = target_pos // num_files + 1
        neg_per_file = target_neg // num_files + 1

        while True:
            epoch_files = file_paths.copy()
            if is_training:
                np.random.shuffle(epoch_files)
            buffer = []
            for file_path in epoch_files:
                try:
                    mat = hdf5storage.loadmat(file_path)

                    if 'positive_dataset' in mat:
                        pos_data = mat['positive_dataset']
                        if pos_data.size > 0 and len(pos_data.shape) == 4:
                            n_pos = min(pos_per_file, pos_data.shape[0])
                            indices = np.random.choice(pos_data.shape[0], n_pos, replace=False)
                            for idx in indices:
                                sample = pos_data[idx].astype(np.float32)
                                sample -= mean[:, np.newaxis, np.newaxis]
                                buffer.append((sample, 0))

                    if 'negative_dataset' in mat:
                        neg_data = mat['negative_dataset']
                        if neg_data.size > 0 and len(neg_data.shape) == 4:
                            n_neg = min(neg_per_file, neg_data.shape[0])
                            indices = np.random.choice(neg_data.shape[0], n_neg, replace=False)
                            for idx in indices:
                                sample = neg_data[idx].astype(np.float32)
                                sample -= mean[:, np.newaxis, np.newaxis]
                                buffer.append((sample, 1))

                    while len(buffer) >= batch_size:
                        batch = buffer[:batch_size]
                        buffer = buffer[batch_size:]
                        X_batch = np.stack([item[0] for item in batch])
                        Y_batch = np.array([item[1] for item in batch])
                        yield X_batch, Y_batch
                except:
                    continue

    output_signature = (
        tf.TensorSpec(shape=(batch_size, 4, 21, 21), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.uint8)
    )

    # 4. 创建 Dataset
    train_gen = lambda: data_generator(file_paths, numPositive, numNegative, batch_size,
                                       mean=mean_nchw, is_training=True)
    val_gen = lambda: data_generator(file_paths, numPositive_test, numNegative_test, batch_size,
                                     mean=mean_nchw, is_training=False)

    train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature)
    val_dataset = tf.data.Dataset.from_generator(val_gen, output_signature=output_signature)

    steps_per_epoch = (numPositive + numNegative) // batch_size
    validation_steps = (numPositive_test + numNegative_test) // batch_size

    print(f"\n训练集 steps_per_epoch: {steps_per_epoch}")
    print(f"验证集 validation_steps: {validation_steps}")

    # 5. 构建模型
    print("\n3. 构建带空间注意力的分类CNN...")
    model = build_spatial_attention_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 6. 回调
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks = [checkpoint]

    # 7. 训练
    print("\n4. 开始训练...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # 8. 保存模型和参数
    print("\n5. 保存模型...")
    model_save_path = os.path.join(save_dir, "saved_model.model")
    model.save(model_save_path)
    print(f"模型已保存至: {model_save_path}")
    print(f"归一化参数已保存至: {mean_path}")

    # 9. 简要评估
    val_loss, val_acc = model.evaluate(val_dataset, steps=validation_steps, verbose=0)
    best_acc = max(history.history['val_accuracy'])
    print("="*60)
    print("训练结果摘要:")
    print(f"验证集准确率: {val_acc:.6f}, 损失: {val_loss:.6f}")
    print(f"最佳验证准确率: {best_acc:.6f}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print("="*60)

if __name__ == "__main__":
    main()