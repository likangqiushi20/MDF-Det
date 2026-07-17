"""
visualize_candidate_extraction.py
展示候选中心提取全过程：
  配准 → 中值背景 → 差分 → 光流运动特征图 → 增强 → 二值化 → 候选中心提取与可视化
处理第300帧，使用前5帧（295-299）作为模板。
候选中心用绿色小点标记。
"""

import os
import cv2
import numpy as np
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.MotionEstimator import MotionEstimator

# ========== 配置参数 ==========
AOI = "02"
FRAME_NUM = 300
TEMPLATE_COUNT = 5
START_FRAME = FRAME_NUM - TEMPLATE_COUNT  # 295
PNG_FOLDER = f"D:/AOI_sequences/AOI{AOI}"
OUTPUT_DIR = f"./visualization_AOI{AOI}_{FRAME_NUM}/"

MOTION_ALPHA = 10
MOTION_THR = 0.5
THRESHOLD_T = 8

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取图像序列（295-300）
    images = {}
    for fn in range(START_FRAME, FRAME_NUM + 1):
        path = os.path.join(PNG_FOLDER, f"frame{fn:06d}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"找不到 {path}")
        images[fn] = img
        print(f"读取 frame {fn}")

    templates = [images[fn] for fn in range(START_FRAME, FRAME_NUM)]
    current_img = images[FRAME_NUM]
    prev_img = images.get(FRAME_NUM - 1)  # 299，用于光流

    # 2. 初始化背景模型并计算单应性
    bgt = BackgroundModel(num_of_template=TEMPLATE_COUNT, templates=templates)
    Hs = bgt.doCalculateHomography(current_img)

    # 3. 运动补偿（配准模板帧到当前帧坐标系）
    bgt.doMotionCompensationAndValidArea(current_img, Hs, current_img.shape)
    compensated = list(bgt.getCompensatedImages())  # 此时未亮度补偿

    # 4. 亮度补偿
    for i, thisTemplate in enumerate(compensated):
        diff = np.float64(current_img) - np.float64(thisTemplate)
        diff = cv2.GaussianBlur(diff, (21, 21), sigmaX=8)
        compensated[i] = np.uint8(np.clip(thisTemplate + diff, 0, 255))

    # 5. 保存配准后的模板帧
    for i, fn in enumerate(range(START_FRAME, FRAME_NUM)):
        out_path = os.path.join(OUTPUT_DIR, f"compensated_frame{fn:06d}.png")
        cv2.imwrite(out_path, compensated[i])

    # 6. 中值背景
    median_bg = np.median(compensated, axis=0).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "median_background.png"), median_bg)

    # 7. 差分热力图
    diff_map = cv2.absdiff(current_img, median_bg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "difference_map.png"), diff_map)

    # 8. 运动特征图（光流）
    me = MotionEstimator(method='farneback', use_gpu=True, downsample_ratio=0.5)
    if prev_img is not None:
        me.compute_flow(prev_img)
        flow = me.compute_flow(current_img)
        M = me.flow_to_magnitude(flow, smooth_sigma=3)
    else:
        M = np.ones_like(current_img, dtype=np.float32)
    M_vis = (M * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "motion_map.png"), M_vis)

    # 9. 融合增强：enhanced = D + α*(M - τ)
    diff_float = diff_map.astype(np.float32)
    motion_offset = MOTION_ALPHA * (M - MOTION_THR)
    enhanced = diff_float + motion_offset
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "enhanced_map.png"), enhanced)

    # 10. 阈值二值图
    binary = (enhanced >= THRESHOLD_T).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, "binary_thresh1.png"), binary)

    # --- 候选中心提取与可视化（使用绿色点） ---
    binary_bool = binary > 0
    h, w = binary_bool.shape
    ds_h, ds_w = h // 3, w // 3
    binary_ds = cv2.resize(binary_bool.astype(np.uint8), (ds_w, ds_h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    r, c = np.where(binary_ds)
    r_orig = np.int32(3 * r + 1)
    c_orig = np.int32(3 * c + 1)
    valid = (r_orig >= 0) & (r_orig < h) & (c_orig >= 0) & (c_orig < w)
    r_orig = r_orig[valid]
    c_orig = c_orig[valid]
    candidate_centers = np.column_stack((c_orig, r_orig))  # (x, y)

    # 在原图上绘制绿色小点（实心圆，半径1）
    vis_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
    for (x, y) in candidate_centers:
        cv2.circle(vis_img, (x, y), 1, (0, 255, 0), -1)   # 绿色实心点
    cv2.imwrite(os.path.join(OUTPUT_DIR, "candidate_centers.png"), vis_img)

    print(f"\n所有可视化结果已保存至 {OUTPUT_DIR}")
    print("生成文件列表：")
    print(f"  compensated_frame295~299.png  (5帧)")
    print(f"  median_background.png")
    print(f"  difference_map.png")
    print(f"  motion_map.png")
    print(f"  enhanced_map.png")
    print(f"  binary_thresh1.png")
    print(f"  candidate_centers.png  (带绿色点标记的候选中心)")

if __name__ == "__main__":
    main()