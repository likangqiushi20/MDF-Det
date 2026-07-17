"""
生成二值掩码标签：使用 OpenCV 画圆，速度快。
"""
import numpy as np
import cv2
import os
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def generate_mask(shape, points, radius=30):
    """使用 OpenCV 画圆生成掩码，速度极快"""
    mask = np.zeros(shape, dtype=np.float32)
    for (r, c) in points:
        r, c = int(round(r)), int(round(c))
        cv2.circle(mask, (c, r), radius, 1, -1)
    return mask

def load_ground_truth(gt_folder, frame_number, csv_name="20091021_truth_rset1_frames0100-0611.csv"):
    gt_path = os.path.join(gt_folder, csv_name)
    if not os.path.exists(gt_path):
        return np.empty((0,2))
    df = pd.read_csv(gt_path)
    df_frame = df[df['FRAME_NUMBER'] == frame_number]
    points = df_frame[['Y', 'X']].values
    return points

def process_frame(args):
    frame, image_folder, gt_folder, target_size, radius = args
    img_path = os.path.join(image_folder, f"frame{frame:06d}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    points = load_ground_truth(gt_folder, frame)
    if len(points) == 0:
        return None
    h, w = img.shape
    mask_full = generate_mask((h, w), points, radius=radius)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_full, target_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0.5).astype(np.float32)
    return (img_resized, mask_resized, frame)

def generate_labels_parallel(image_folder, gt_folder, frame_range, output_dir,
                             target_size=(256,256), radius=30, workers=8):
    os.makedirs(output_dir, exist_ok=True)
    args_list = [(frame, image_folder, gt_folder, target_size, radius) for frame in frame_range]
    with Pool(workers) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(process_frame, args_list), total=len(args_list)):
            if res is not None:
                results.append(res)
    for img_resized, mask_resized, frame in results:
        save_path = os.path.join(output_dir, f"frame{frame:06d}.npz")
        np.savez(save_path, image=img_resized, mask=mask_resized)
    print(f"生成完成，保存至 {output_dir}，共 {len(results)} 帧")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--gt_folder', required=True)
    parser.add_argument('--start_frame', type=int, default=100)
    parser.add_argument('--num_frames', type=int, default=412)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--radius', type=int, default=30, help='正样本半径（像素）')
    parser.add_argument('--target_size', type=str, default='256,256')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    target_size = tuple(map(int, args.target_size.split(',')))
    frame_range = range(args.start_frame, args.start_frame + args.num_frames)
    generate_labels_parallel(args.image_folder, args.gt_folder, frame_range, args.output_dir,
                             target_size=target_size, radius=args.radius, workers=args.workers)