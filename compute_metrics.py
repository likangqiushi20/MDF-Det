import numpy as np
import pandas as pd
import os
import glob
from sklearn.neighbors import NearestNeighbors
from math import radians, sin, cos, sqrt, asin

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def filter_stationary_targets(gt_csv, dist_thresh_m=0.8):
    df = pd.read_csv(gt_csv)
    moving_mask = np.zeros(len(df), dtype=bool)
    for obj_id, group in df.groupby('id'):
        group = group.sort_values('FRAME_NUMBER')
        lats = group['LATITUDE'].values
        lons = group['LONGITUDE'].values
        for i in range(len(lats)-1):
            dist = haversine(lons[i], lats[i], lons[i+1], lats[i+1])
            if dist >= dist_thresh_m:
                moving_mask[group.index[i]] = True
                moving_mask[group.index[i+1]] = True
    df_moving = df[moving_mask]
    gt_by_frame = {}
    for frame, group in df_moving.groupby('FRAME_NUMBER'):
        points = group[['X', 'Y']].values.astype(np.float32)
        gt_by_frame[int(frame)] = points
    return gt_by_frame

def load_detections(detection_folder, frame_range):
    det_by_frame = {}
    total_dets = 0
    for frame in frame_range:
        fname = f"frame{frame:06d}.csv"
        path = os.path.join(detection_folder, fname)
        if not os.path.exists(path):
            det_by_frame[frame] = np.empty((0,2))
        else:
            points = np.loadtxt(path, delimiter=',', dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            det_by_frame[frame] = points
            total_dets += points.shape[0]
            if frame % 50 == 0:
                print(f"  加载帧 {frame}: {points.shape[0]} 个检测点")
    print(f"总计检测点数: {total_dets}")
    return det_by_frame

def compute_metrics_for_frame(gt_points, det_points, dist_thresh=10.0):
    if len(gt_points) == 0 and len(det_points) == 0:
        return 0, 0, 0
    if len(gt_points) == 0:
        return 0, len(det_points), 0
    if len(det_points) == 0:
        return 0, 0, len(gt_points)
    nbrs = NearestNeighbors(n_neighbors=1).fit(gt_points)
    distances, indices = nbrs.kneighbors(det_points)
    distances = distances.flatten()
    matched_gt = set()
    tp = 0
    for i, dist in enumerate(distances):
        if dist <= dist_thresh:
            gt_idx = indices[i][0]
            if gt_idx not in matched_gt:
                matched_gt.add(gt_idx)
                tp += 1
    fp = len(det_points) - tp
    fn = len(gt_points) - len(matched_gt)
    return tp, fp, fn

def main():
    # 配置
    gt_csv = "D:/Lkqs/sys/code/WPAFB2009/TrackTruth/TRAIN/20091021_truth_rset1_frames0100-0611.csv"
    detection_folder = "./WAMI-output-context/CSV"  # 请修改为实际路径
    start_frame = 407
    end_frame = 430  # 根据实际检测帧数调整，这里假设24帧：407-430
    dist_thresh_pixel = 10.0
    move_thresh_m = 0.8

    print("加载真值并过滤静止目标...")
    gt_by_frame = filter_stationary_targets(gt_csv, move_thresh_m)
    frame_range = range(start_frame, end_frame + 1)
    print(f"真值帧范围: {start_frame} - {end_frame}")

    print("加载检测结果...")
    det_by_frame = load_detections(detection_folder, frame_range)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    frames_processed = 0

    # 先打印前几帧的详细信息
    print("\n前5帧详细信息:")
    for frame in list(frame_range)[:5]:
        gt = gt_by_frame.get(frame, np.empty((0,2)))
        det = det_by_frame.get(frame, np.empty((0,2)))
        print(f"  帧 {frame}: GT点数 {len(gt)}, Det点数 {len(det)}")
        if len(det) > 0:
            print(f"    检测点示例: {det[:5]}")

    for frame in frame_range:
        gt = gt_by_frame.get(frame, np.empty((0,2)))
        det = det_by_frame.get(frame, np.empty((0,2)))
        tp, fp, fn = compute_metrics_for_frame(gt, det, dist_thresh_pixel)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        frames_processed += 1
        if frames_processed % 10 == 0:
            print(f"已处理 {frames_processed} 帧，累计 TP={total_tp}, FP={total_fp}, FN={total_fn}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n========== 评估结果 ==========")
    print(f"总帧数: {frames_processed}")
    print(f"总真值点数（移动）: {total_tp + total_fn}")
    print(f"总检测点数: {total_tp + total_fp}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数:            {f1:.4f}")

if __name__ == "__main__":
    main()