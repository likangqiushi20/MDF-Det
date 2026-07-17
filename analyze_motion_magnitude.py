"""
analyze_motion_magnitude.py
计算并显示当前帧的运动显著性图 M 的值域统计。
用法：
    python analyze_motion_magnitude.py --png_folder "D:/AOI_sequences/AOI01" --frame 100
"""

import argparse
import cv2
import numpy as np
import os

from MovingObjectDetector.MotionEstimator import MotionEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png_folder', required=True, help='PNG 序列所在文件夹')
    parser.add_argument('--frame', type=int, required=True, help='要分析的帧号')
    args = parser.parse_args()

    # 找到当前帧和前一帧
    curr_path = os.path.join(args.png_folder, f"frame{args.frame:06d}.png")
    prev_path = os.path.join(args.png_folder, f"frame{args.frame - 1:06d}.png")

    if not os.path.exists(curr_path):
        print(f"错误：当前帧 {curr_path} 不存在")
        return
    if not os.path.exists(prev_path):
        print(f"前一帧 {prev_path} 不存在，无法计算光流")
        return

    curr_img = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    prev_img = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    if curr_img is None or prev_img is None:
        print("图像读取失败")
        return

    # 初始化光流估计器
    me = MotionEstimator(method='farneback', use_gpu=True, downsample_ratio=0.5)
    # 先喂入前一帧
    me.compute_flow(prev_img)
    # 计算当前帧的光流
    flow = me.compute_flow(curr_img)
    # 计算运动幅值图
    M = me.flow_to_magnitude(flow, smooth_sigma=3)

    # 统计量
    print(f"\n===== 运动显著性图 M 统计 (帧 {args.frame}) =====")
    print(f"形状: {M.shape}")
    print(f"最小值: {M.min():.4f}")
    print(f"最大值: {M.max():.4f}")
    print(f"均值:   {M.mean():.4f}")
    print(f"中位数: {np.median(M):.4f}")
    print(f"标准差: {M.std():.4f}")

    # 分位数
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"{q}% 分位数: {np.percentile(M, q):.4f}")

    # 像素数占比（区间）
    print("\n区间占比：")
    intervals = [(0,0.1), (0.1,0.2), (0.2,0.3), (0.3,0.4), (0.4,0.5),
                 (0.5,0.6), (0.6,0.7), (0.7,0.8), (0.8,0.9), (0.9,1.0)]
    total = M.size
    for low, high in intervals:
        count = np.sum((M >= low) & (M < high))
        print(f"  {low:.1f} ~ {high:.1f}: {count/total*100:.2f}%")

    # 可选：保存为文本文件
    out_txt = os.path.join(args.png_folder, f"motion_stats_frame{args.frame:06d}.txt")
    with open(out_txt, 'w') as f:
        f.write(f"frame {args.frame}\n")
        f.write(f"min={M.min()}, max={M.max()}, mean={M.mean()}, median={np.median(M)}\n")
        for q in [10,25,50,75,90,95,99]:
            f.write(f"q{q}: {np.percentile(M,q):.4f}\n")
        f.write("interval ratios:\n")
        for low,high in intervals:
            f.write(f"{low}-{high}: {np.sum((M>=low)&(M<high))/total*100:.2f}%\n")

    print(f"统计信息已保存到 {out_txt}")

if __name__ == "__main__":
    main()