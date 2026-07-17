"""
compute_mean_only.py
从回归训练样本（.mat 文件）中采样计算归一化均值，保存为 regression_norm_params.npz
"""
import numpy as np
import scipy.io as sio
import glob
import os
import argparse
from tqdm import tqdm

def compute_mean_sampling(mat_files, max_samples=50000):
    """从 .mat 文件中采样计算均值（形状 (4,45,45)）"""
    X_samples = []
    total_loaded = 0
    for mat_path in tqdm(mat_files, desc="读取样本"):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_folder', required=True, help='存放 .mat 样本的文件夹')
    parser.add_argument('--max_samples', type=int, default=50000, help='用于计算均值的最大样本数')
    parser.add_argument('--output', default='regression_norm_params.npz', help='输出文件名')
    args = parser.parse_args()

    # 获取所有 .mat 文件
    all_files = sorted(glob.glob(os.path.join(args.mat_folder, "*.mat")))
    if len(all_files) == 0:
        print("错误：未找到 .mat 文件")
        return
    print(f"找到 {len(all_files)} 个 .mat 文件")

    # 计算均值
    print("计算归一化均值...")
    mean_img = compute_mean_sampling(all_files, args.max_samples)
    print(f"均值形状: {mean_img.shape}")

    # 保存
    np.savez(args.output, mean_img=mean_img)
    print(f"归一化参数已保存到 {args.output}")

if __name__ == "__main__":
    main()