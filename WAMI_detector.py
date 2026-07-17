"""
WAMI_detector.py - 原始方案检测器（无空间注意力）
输出：绿色圆圈=分类直接输出，红色圆圈=回归输出
输出文件使用输入图像的原始帧号，并保存检测点CSV
"""

import argparse
import numpy as np
import cv2
import os
import timeit
import re
from copy import copy

from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
import TrainNetwork.BaseFunctions as basefunctions

def main():
    parser = argparse.ArgumentParser(description='原始WAMI检测器（无空间注意力）')
    parser.add_argument('-I', '--InputFolder', type=str,
                        default='D:/Lkqs/sys/code/WPAFB2009/train-404-437',
                        help='输入图像文件夹')
    parser.add_argument('-O', '--OutputFolder', type=str,
                        default='./WAMI-output',
                        help='输出图像文件夹')
    parser.add_argument('-N', '--NNModelFolder', type=str, default="Models/",
                        help='模型文件夹（包含 BinaryClassification 和 Regression）')
    parser.add_argument('-T', '--BSThreshold', type=int, default=8,
                        help='背景差分阈值（论文中全图使用8）')
    parser.add_argument('-NT', '--NumOfTemplate', type=int, default=3,
                        help='背景模板数量（论文中全图使用3）')
    args = parser.parse_args()

    # 输出文件夹
    out_folder = args.OutputFolder
    os.makedirs(out_folder, exist_ok=True)
    csv_folder = os.path.join(out_folder, "CSV")
    os.makedirs(csv_folder, exist_ok=True)
    print(f"输出图像文件夹: {out_folder}")
    print(f"CSV文件夹: {csv_folder}")

    # 加载原始模型（分类+回归）
    print("加载原始检测模型...")
    model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(args.NNModelFolder)
    model = (model_binary, aveImg_binary, model_regression, aveImg_regression)

    # 图像列表
    imagefolder = args.InputFolder
    filenames = [f for f in os.listdir(imagefolder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    filenames.sort()
    print(f"找到 {len(filenames)} 帧图像")

    if len(filenames) < args.NumOfTemplate:
        print(f"错误：需要至少 {args.NumOfTemplate} 帧作为模板")
        return

    # 初始化背景模型
    templates = []
    for i in range(args.NumOfTemplate):
        img_path = os.path.join(imagefolder, filenames[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        templates.append(img)
    bgt = BackgroundModel(num_of_template=args.NumOfTemplate, templates=templates)

    # 主循环
    for idx in range(args.NumOfTemplate, len(filenames)):
        start = timeit.default_timer()
        fname = filenames[idx]
        print(f"\n处理 {idx+1}/{len(filenames)}: {fname}")

        img = cv2.imread(os.path.join(imagefolder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 提取原始帧号（假设文件名包含数字）
        match = re.search(r'\d+', fname)
        if match:
            frame_num = match.group()
            out_img_name = f"frame{int(frame_num):06d}.png"
            out_csv_name = f"frame{int(frame_num):06d}.csv"
        else:
            out_img_name = f"frame{idx+1:06d}.png"
            out_csv_name = f"frame{idx+1:06d}.csv"
        print(f"  输出帧号: {out_img_name}")

        # 单应性计算
        try:
            Hs = bgt.doCalculateHomography(img)
        except Exception as e:
            print(f"  单应性计算失败，跳过该帧: {e}")
            bgt.updateTemplate(img)
            continue

        bgt.doMotionCompensationAndValidArea(img, Hs, img.shape)
        cand_centres, bg_props, bg_labels = bgt.doBackgroundSubtraction(img, thres=args.BSThreshold)
        print(f"  背景减除完成，候选点: {len(cand_centres)}")

        dr = DetectionRefinement(img, bgt.getCompensatedImages(), cand_centres,
                                 bg_props, bg_labels, model)
        det1, det2, _ = dr.do_refine_bs()

        # 提取中心点坐标 (x, y)
        det1_centers = []
        for d in det1:
            row, col = d["centre"][0], d["centre"][1]
            det1_centers.append([int(col), int(row)])
        det2_centers = []
        for d in det2:
            row, col = d["centre"][0], d["centre"][1]
            det2_centers.append([int(col), int(row)])

        # 可视化
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for (x, y) in det1_centers:
            if 0 < x < img.shape[1] and 0 < y < img.shape[0]:
                cv2.circle(out_img, (x, y), 6, (0, 255, 0), 2)   # 绿色
        for (x, y) in det2_centers:
            if 0 < x < img.shape[1] and 0 < y < img.shape[0]:
                cv2.circle(out_img, (x, y), 6, (0, 0, 255), 2)   # 红色

        out_img_path = os.path.join(out_folder, out_img_name)
        cv2.imwrite(out_img_path, out_img)
        print(f"  保存图像至 {out_img_path}")

        # 保存检测点到 CSV
        all_detections = np.array(det1_centers + det2_centers, dtype=np.float32)
        csv_path = os.path.join(csv_folder, out_csv_name)
        np.savetxt(csv_path, all_detections, delimiter=',', fmt='%.1f')
        print(f"  保存检测CSV: {csv_path}，检测数: {len(all_detections)}")

        bgt.updateTemplate(img)
        print(f"  耗时 {timeit.default_timer()-start:.2f}s")

if __name__ == "__main__":
    main()