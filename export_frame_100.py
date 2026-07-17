"""
export_frame_100.py
从 WPAFB 数据集中提取第 100 帧（.r1 分辨率）并保存为完整 PNG。
"""

import os
import glob
from osgeo import gdal

# 数据根目录
INPUT_ROOT = r"D:\Lkqs\Data\wami_data"
OUTPUT_PNG = r"D:\frame_100_full.png"

# 三个训练子文件夹
TRAIN_SUBDIRS = [
    "WPAFB-21Oct2009-TRAIN_NITF_001",
    "WPAFB-21Oct2009-TRAIN_NITF_002",
    "WPAFB-21Oct2009-TRAIN_NITF_003"
]

def find_frame_100():
    """查找第 100 帧对应的 .r1 文件（通常是第一个文件夹的第一帧）"""
    for subdir in TRAIN_SUBDIRS:
        search_path = os.path.join(
            INPUT_ROOT, subdir,
            "WPAFB-21Oct2009", "Data", "TRAIN", "NITF", "*.r1"
        )
        files = sorted(glob.glob(search_path))
        if len(files) >= 1:
            # 第一个文件通常对应帧 100（根据 MATLAB 脚本逻辑）
            return files[0]
    return None

def export_full_png(input_nitf, output_png):
    """将 NITF 完整转换为 PNG"""
    ds = gdal.Open(input_nitf)
    if ds is None:
        raise FileNotFoundError(f"无法打开 {input_nitf}")

    # 直接转换，保留原始分辨率
    gdal.Translate(output_png, ds, format='PNG', outputType=gdal.GDT_Byte)
    ds = None
    print(f"完整 PNG 已保存至: {output_png}")

if __name__ == "__main__":
    r1_path = find_frame_100()
    if r1_path is None:
        print("未找到第 100 帧 .r1 文件")
    else:
        print(f"找到帧 100: {r1_path}")
        export_full_png(r1_path, OUTPUT_PNG)