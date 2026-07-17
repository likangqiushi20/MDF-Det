"""
train_regression_spatial_attention_stream.py
流式加载所有 .mat 文件，训练空间注意力回归CNN
计算均值时采样部分样本，避免加载全部数据
"""
import tensorflow as tf
import scipy.io as sio
import glob
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from regression_cnn_spatial_attention import build_regression_cnn_spatial_attention

# GPU 内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def compute_mean_sampling(mat_files, max_samples=10000):
    """从 .mat 文件中采样计算均值（避免加载全部）"""
    X_samples = []
    total_loaded = 0
    for mat_path in mat_files:
        mat = sio.loadmat(mat_path)
        data = mat.get('dataset', np.array([]))
        if len(data) == 0:
            continue
        # 每个文件随机取一部分
        n = min(data.shape[0], max_samples // len(mat_files) + 1)
        idx = np.random.choice(data.shape[0], n, replace=False)
        X_samples.append(data[idx])
        total_loaded += n
        if total_loaded >= max_samples:
            break
    X_samples = np.concatenate(X_samples, axis=0)
    mean_img = np.mean(X_samples, axis=0)  # 形状 (4,45,45)
    return mean_img

def mat_generator(mat_files, batch_size, mean_img=None, shuffle=True):
    """生成器，逐文件读取 .mat，每次 yield 一个 batch"""
    if shuffle:
        mat_files = np.random.permutation(mat_files)
    X_batch = []
    Y_batch = []
    for mat_path in mat_files:
        mat = sio.loadmat(mat_path)
        data = mat.get('dataset', np.array([]))
        labels = mat.get('labelset', np.array([]))
        if len(data) == 0:
            continue
        # 归一化（减去均值）
        if mean_img is not None:
            data = data - mean_img
        # 打乱文件内样本顺序
        n = data.shape[0]
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for i in idx:
            X_batch.append(data[i])
            Y_batch.append(labels[i])
            if len(X_batch) == batch_size:
                yield np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)
                X_batch = []
                Y_batch = []
    if X_batch:
        yield np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)

def create_dataset(mat_files, batch_size, mean_img, shuffle=True):
    """创建 tf.data.Dataset"""
    dataset = tf.data.Dataset.from_generator(
        lambda: mat_generator(mat_files, batch_size, mean_img, shuffle),
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 4, 45, 45), (None, 225))
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_folder', required=True, help='存放 .mat 样本的文件夹')
    parser.add_argument('--model_save', default='regression_spatial_attention.h5')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例（按文件划分）')
    parser.add_argument('--max_samples_for_mean', type=int, default=50000, help='计算均值时使用的最大样本数')
    args = parser.parse_args()

    # 获取所有 .mat 文件（按文件名排序）
    all_files = sorted(glob.glob(os.path.join(args.mat_folder, "*.mat")))
    if len(all_files) == 0:
        print("错误：未找到 .mat 文件")
        return
    print(f"找到 {len(all_files)} 个 .mat 文件")

    # 按文件划分训练/验证集（保持帧顺序独立）
    split_idx = int(len(all_files) * (1 - args.val_split))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    print(f"训练文件数: {len(train_files)}, 验证文件数: {len(val_files)}")

    # 计算均值（采样）
    print("计算归一化均值...")
    mean_img = compute_mean_sampling(train_files, args.max_samples_for_mean)
    print(f"归一化均值形状: {mean_img.shape}")

    # 创建流式数据集
    train_dataset = create_dataset(train_files, args.batch_size, mean_img, shuffle=True)
    val_dataset = create_dataset(val_files, args.batch_size, mean_img, shuffle=False)

    # 构建模型
    model = build_regression_cnn_spatial_attention(input_shape=(4,45,45))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model_save, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # 训练
    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)

    # 保存模型和归一化参数
    model.save(args.model_save)
    np.savez("regression_norm_params.npz", mean_img=mean_img)
    print(f"模型已保存到 {args.model_save}")
    print("归一化参数已保存到 regression_norm_params.npz")

if __name__ == "__main__":
    main()