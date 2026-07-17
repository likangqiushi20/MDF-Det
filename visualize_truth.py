"""
visualize_truth.py
功能：将测试集图像与真值点对应，绘制绿色圆圈并保存。
"""

import cv2
import numpy as np
import pandas as pd
import os

def main():
    # 路径配置
    test_image_folder = "D:/Lkqs/sys/code/WPAFB2009/training"
    truth_csv = "D:/Lkqs/sys/code/WPAFB2009/TrackTruth/TRAIN/20091021_truth_rset1_frames0100-0611.csv"
    output_folder = "./KSH"
    os.makedirs(output_folder, exist_ok=True)

    # 真值帧号
    truth_frame = 612
    # 测试图像文件名（根据您的实际命名，假设为 frame000612.png）
    image_filename = "frame000612.png"
    image_path = os.path.join(test_image_folder, image_filename)

    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 加载真值文件
    if not os.path.exists(truth_csv):
        print(f"错误：真值文件不存在 {truth_csv}")
        return
    df = pd.read_csv(truth_csv)

    # 筛选出指定帧的所有真值点（包括 M, I, O 类型）
    df_frame = df[df['FRAME_NUMBER'] == truth_frame]
    points = df_frame[['X', 'Y']].values.astype(np.int32)

    print(f"帧 {truth_frame} 共有 {len(points)} 个真值点")

    # 绘制绿色圆圈
    for (x, y) in points:
        # 确保点在图像范围内
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_color, (x, y), 6, (0, 255, 0), 2)

    # 保存结果
    output_path = os.path.join(output_folder, f"truth_frame_{truth_frame:06d}.png")
    cv2.imwrite(output_path, img_color)
    print(f"已保存至 {output_path}")

if __name__ == "__main__":
    main()