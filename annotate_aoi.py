"""
annotate_aoi.py (修正版)
自动查找第100帧 .r1 文件，在 PNG 上标注矩形，输出经纬度边界。
"""

import os
import glob
import cv2
import numpy as np
from osgeo import gdal, osr

# ========== 配置 ==========
DATA_ROOT = r"D:\Lkqs\Data\wami_data"
PNG_PATH = r"D:\frame_100_full.png"          # 之前生成的完整 PNG
OUTPUT_ANNOTATED = r"D:\frame_100_annotated.png"  # 带标注的 PNG 副本
# =========================

# 三个训练子文件夹
TRAIN_SUBDIRS = [
    "WPAFB-21Oct2009-TRAIN_NITF_001",
    "WPAFB-21Oct2009-TRAIN_NITF_002",
    "WPAFB-21Oct2009-TRAIN_NITF_003"
]

def find_frame_100_nitf():
    """自动查找第 100 帧的 .r1 文件"""
    for subdir in TRAIN_SUBDIRS:
        search_path = os.path.join(
            DATA_ROOT, subdir,
            "WPAFB-21Oct2009", "Data", "TRAIN", "NITF", "*.r1"
        )
        files = sorted(glob.glob(search_path))
        # 假设第一个文件就是帧100（根据 MATLAB 脚本的逻辑）
        if files:
            return files[0]
    return None

# 自动查找 NITF 路径
NITF_PATH = find_frame_100_nitf()
if NITF_PATH is None:
    print("错误：未找到第 100 帧 .r1 文件，请检查 DATA_ROOT 路径。")
    exit(1)
print(f"使用 NITF 文件: {NITF_PATH}")

# ========== 全局变量 ==========
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
img_raw = None
img_disp = None
scale = 1.0
final_rect = None  # 存储最终矩形 (x1, y1, x2, y2)

def resize_to_fit(img, max_width=1600, max_height=900):
    """将图像缩放到适合屏幕的尺寸"""
    global scale
    h, w = img.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, img_disp, img_raw, scale, final_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (int(x / scale), int(y / scale))
        end_point = start_point
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (int(x / scale), int(y / scale))
            temp = img_raw.copy()
            cv2.rectangle(temp, start_point, end_point, (0, 255, 0), 2)
            img_disp = resize_to_fit(temp)
            cv2.imshow("AOI Annotation", img_disp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (int(x / scale), int(y / scale))
        cv2.rectangle(img_raw, start_point, end_point, (0, 255, 0), 2)
        img_disp = resize_to_fit(img_raw.copy())
        cv2.imshow("AOI Annotation", img_disp)
        # 保存最终矩形（确保左上右下顺序）
        x1, x2 = min(start_point[0], end_point[0]), max(start_point[0], end_point[0])
        y1, y2 = min(start_point[1], end_point[1]), max(start_point[1], end_point[1])
        final_rect = (x1, y1, x2, y2)
        print(f"已绘制矩形: 左上({x1},{y1}) 右下({x2},{y2})")

def pixel_to_geo(geotrans, col, row, proj_wkt=None):
    """像素坐标转经纬度"""
    x = geotrans[0] + col * geotrans[1] + row * geotrans[2]
    y = geotrans[3] + col * geotrans[4] + row * geotrans[5]
    if proj_wkt:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_wkt)
        dst_srs = osr.SpatialReference()
        dst_srs.SetWellKnownGeogCS("WGS84")
        transform = osr.CoordinateTransformation(src_srs, dst_srs)
        lon, lat, _ = transform.TransformPoint(x, y)
        return lon, lat
    else:
        return x, y

def main():
    global img_raw, img_disp, scale, final_rect

    # 读取 PNG
    img = cv2.imread(PNG_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图像 {PNG_PATH}")
        return

    # 处理为 8-bit 灰度显示
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转为三通道以便绘制彩色矩形
    img_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_disp = resize_to_fit(img_raw.copy())

    # 读取 NITF 地理信息
    ds = gdal.Open(NITF_PATH)
    if ds is None:
        print(f"错误：无法打开 NITF {NITF_PATH}")
        return
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    cv2.namedWindow("AOI Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("AOI Annotation", mouse_callback)

    print("\n操作说明：")
    print("  1. 用鼠标左键拖拽绘制矩形 AOI")
    print("  2. 按 'q' 键退出并输出边界经纬度")
    print("  3. 按 'r' 键重置矩形")

    while True:
        cv2.imshow("AOI Annotation", img_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 重置
            img_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_disp = resize_to_fit(img_raw.copy())
            final_rect = None
            print("矩形已重置")

    cv2.destroyAllWindows()

    if final_rect is not None:
        x1, y1, x2, y2 = final_rect
        # 保存带标注的 PNG
        cv2.imwrite(OUTPUT_ANNOTATED, img_raw)
        print(f"带标注的 PNG 已保存至: {OUTPUT_ANNOTATED}")

        # 计算经纬度边界
        ul_lon, ul_lat = pixel_to_geo(geotrans, x1, y1, proj)
        lr_lon, lr_lat = pixel_to_geo(geotrans, x2, y2, proj)
        print("\n" + "="*50)
        print("AOI 地理边界参数 (复制到裁剪脚本):")
        print(f"--ulx {ul_lon:.6f} --uly {ul_lat:.6f} --lrx {lr_lon:.6f} --lry {lr_lat:.6f}")
        print("="*50)
    else:
        print("未绘制任何矩形。")

if __name__ == "__main__":
    main()