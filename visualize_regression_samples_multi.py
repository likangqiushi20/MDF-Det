"""
save_regression_heatmaps_separate.py
随机选取5个回归训练样本，单独保存15×15和45×45热力图（GT/预测）。
每个样本生成独立的PNG文件，无坐标轴与边框，配色 jet。
用法：
    python save_regression_heatmaps_separate.py
    python save_regression_heatmaps_separate.py --model regression_spatial_attention.h5 --norm regression_norm_params.npz --cmap jet
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
import glob
import tensorflow as tf
import cv2

def save_heatmap(array, filepath, cmap='jet'):
    """保存纯净热力图，无坐标轴、无边框"""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(array, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='单独保存回归样本15×15和45×45热力图')
    parser.add_argument('--sample_folder', type=str,
                        default=r"D:\Lkqs\sys\code\regression_training_samples",
                        help='回归训练样本 .mat 文件所在文件夹')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='要保存的样本数量，默认20')
    parser.add_argument('--model', type=str, default=None,
                        help='可选：训练好的回归模型路径（.h5）')
    parser.add_argument('--norm', type=str, default=None,
                        help='可选：回归模型归一化参数文件（.npz）')
    parser.add_argument('--output_dir', type=str, default='./regression_heatmaps',
                        help='输出目录，默认为 ./regression_heatmaps')
    parser.add_argument('--cmap', type=str, default='jet',
                        help='热力图颜色映射，例如 jet, hot, plasma, inferno, RdYlBu_r 等')
    parser.add_argument('--save_input_frames', action='store_true',
                        help='是否同时保存四帧输入图像（可选）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 扫描所有 .mat 文件
    mat_files = glob.glob(os.path.join(args.sample_folder, "*.mat"))
    if not mat_files:
        print(f"错误：在 {args.sample_folder} 中未找到 .mat 文件。")
        return

    random.shuffle(mat_files)
    collected_samples = []
    target_count = min(args.num_samples, 5)

    for fpath in mat_files:
        try:
            mat = hdf5storage.loadmat(fpath)
            dataset = mat.get('dataset')
            labelset = mat.get('labelset')
            if dataset is None or labelset is None:
                continue
            N = dataset.shape[0]
            if N == 0:
                continue
            need = target_count - len(collected_samples)
            pick_indices = random.sample(range(N), min(need, N))
            basename = os.path.splitext(os.path.basename(fpath))[0]
            for idx in pick_indices:
                sample = dataset[idx].astype(np.float32)      # (4,45,45)
                label = labelset[idx]                          # (225,)
                collected_samples.append((basename, idx, sample, label))
            if len(collected_samples) >= target_count:
                break
        except Exception as e:
            print(f"处理文件 {fpath} 时出错: {e}")
            continue

    if len(collected_samples) == 0:
        print("未找到有效的回归样本。")
        return

    # 可选：加载模型
    model = None
    mean_img = None
    if args.model and args.norm:
        print(f"加载回归模型: {args.model}")
        model = tf.keras.models.load_model(args.model, compile=False)
        norm_data = np.load(args.norm)
        for k in ['mean_img', 'mean', 'mu']:
            if k in norm_data:
                mean_img = norm_data[k]
                break
        if mean_img is None:
            print("警告：归一化参数文件中未找到均值，将使用零均值。")
            mean_img = np.zeros((4, 45, 45), dtype=np.float32)

    # 处理每个样本
    for i, (basename, idx, sample, label_flat) in enumerate(collected_samples):
        print(f"生成样本 {i+1}/{len(collected_samples)}: {basename}_idx{idx}")
        prefix = f"{basename}_idx{idx}"

        # 真实标签 15×15 和 45×45
        gt_15 = np.reshape(label_flat, (15, 15))
        gt_45 = cv2.resize(gt_15, (45, 45), interpolation=cv2.INTER_LINEAR)

        # 保存真实标签热力图
        save_heatmap(gt_15, os.path.join(args.output_dir, f"{prefix}_gt_15x15.png"), cmap=args.cmap)
        save_heatmap(gt_45, os.path.join(args.output_dir, f"{prefix}_gt_45x45.png"), cmap=args.cmap)

        # 如果有模型，保存预测热力图
        if model is not None:
            input_tensor = (sample - mean_img).reshape(1, 4, 45, 45)
            pred = model.predict(input_tensor, verbose=0)      # (1,225)
            pred_15 = np.reshape(pred, (15, 15))
            pred_45 = cv2.resize(pred_15, (45, 45), interpolation=cv2.INTER_LINEAR)
            save_heatmap(pred_15, os.path.join(args.output_dir, f"{prefix}_pred_15x15.png"), cmap=args.cmap)
            save_heatmap(pred_45, os.path.join(args.output_dir, f"{prefix}_pred_45x45.png"), cmap=args.cmap)

        # 可选保存输入帧图像
        if args.save_input_frames:
            for j in range(4):
                fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                ax.imshow(sample[j], cmap='gray')
                ax.axis('off')
                plt.tight_layout(pad=0)
                frame_path = os.path.join(args.output_dir, f"{prefix}_frame{j}.png")
                plt.savefig(frame_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

    print(f"\n所有热力图已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()