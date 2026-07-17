"""
extract_aois_from_wpafb_nitf.py
从 WPAFB 数据集的 .r1 NITF 文件中，批量提取多个 AOI 区域并保存为 PNG 序列。

参数说明：
    --input_root    : WAMI 数据根目录
    --output_root   : 输出根目录
    --start_frame   : 第一帧的输出帧号（默认 100）
    --num_frames    : 要处理的帧数（默认处理所有文件）
    --aois          : 要处理的 AOI 列表

示例：
    python extract_all_aois_from_wpafb_nitf.py \
        --input_root "D:/Lkqs/Data/wami_data" \
        --output_root "D:/AOI_sequences" \
        --start_frame 100 \
        --num_frames 512 \
        --aois 01 02 03 04 34 40 41
"""

import os
import glob
import argparse
from osgeo import gdal

# 三个训练子文件夹
TRAIN_SUBDIRS = [
    "WPAFB-21Oct2009-TRAIN_NITF_001",
    "WPAFB-21Oct2009-TRAIN_NITF_002",
    "WPAFB-21Oct2009-TRAIN_NITF_003"
]

# AOI 经纬度边界配置 (ulx, uly, lrx, lry)
AOI_CONFIGS = {
    '01': {'ulx': -84.125664, 'uly': 39.771185, 'lrx': -84.119895, 'lry': 39.765354},
    '02': {'ulx': -84.122165, 'uly': 39.786161, 'lrx': -84.116273, 'lry': 39.780145},
    '03': {'ulx': -84.101297, 'uly': 39.781737, 'lrx': -84.095836, 'lry': 39.775665},
    '04': {'ulx': -84.122289, 'uly': 39.766089, 'lrx': -84.116211, 'lry': 39.760628},
    '34': {'ulx': -84.126152, 'uly': 39.783150, 'lrx': -84.114983, 'lry': 39.776215},
    '40': {'ulx': -84.127380, 'uly': 39.772167, 'lrx': -84.118790, 'lry': 39.765719},
    '41': {'ulx': -84.121796, 'uly': 39.772351, 'lrx': -84.113391, 'lry': 39.764676},
}

def collect_r1_files(input_root):
    """收集所有 .r1 文件，并按文件名排序"""
    r1_files = []
    for subdir in TRAIN_SUBDIRS:
        search_path = os.path.join(
            input_root, subdir,
            "WPAFB-21Oct2009", "Data", "TRAIN", "NITF", "*.r1"
        )
        files = glob.glob(search_path)
        r1_files.extend(files)
        print(f"在 {subdir} 中找到 {len(files)} 个 .r1 文件")

    r1_files.sort()
    print(f"总共收集到 {len(r1_files)} 个 .r1 文件")
    return r1_files

def crop_nitf_by_geo_bounds(input_nitf, output_png, ulx, uly, lrx, lry):
    """使用 GDAL Warp 按地理坐标裁剪并保存为 PNG"""
    bounds = [ulx, lry, lrx, uly]  # GDAL 格式: [minX, minY, maxX, maxY]
    warp_options = gdal.WarpOptions(
        format='PNG',
        outputBounds=bounds,
        dstNodata=0,
        resampleAlg=gdal.GRA_Bilinear,
        warpMemoryLimit=1024,
        multithread=True,
        # 注意：不设置 GDAL_PAM_ENABLED=NO，以便生成 .aux.xml 文件
    )
    try:
        gdal.Warp(output_png, input_nitf, options=warp_options)
        return True
    except Exception as e:
        print(f"  裁剪失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="从 WPAFB 数据集批量提取多个 AOI 的 PNG 序列")
    parser.add_argument("--input_root", required=True, help="WAMI 数据根目录，包含三个 TRAIN_NITF 子文件夹")
    parser.add_argument("--output_root", required=True, help="输出根目录，各 AOI 子文件夹将创建于此")
    parser.add_argument("--start_frame", type=int, default=100, help="第一帧的输出帧号，默认为 100")
    parser.add_argument("--num_frames", type=int, default=None, help="要处理的帧数，默认处理所有文件")
    parser.add_argument("--aois", nargs='+', default=['01','02','03','04','34','40','41'],
                        help="要处理的 AOI 列表，默认全部")
    args = parser.parse_args()

    print("正在扫描 .r1 文件...")
    all_r1_files = collect_r1_files(args.input_root)
    total_available = len(all_r1_files)

    if total_available == 0:
        print("未找到任何 .r1 文件，请检查输入路径。")
        return

    # 确定要处理的文件范围
    if args.num_frames is not None:
        num_to_process = min(args.num_frames, total_available)
        r1_files = all_r1_files[:num_to_process]
    else:
        num_to_process = total_available
        r1_files = all_r1_files

    print(f"将处理前 {num_to_process} 个文件（共 {total_available} 个）")
    print(f"输出帧号范围: {args.start_frame:06d} ~ {args.start_frame + num_to_process - 1:06d}")

    # 为每个 AOI 创建输出目录
    for aoi_name in args.aois:
        if aoi_name not in AOI_CONFIGS:
            print(f"警告：未知的 AOI '{aoi_name}'，跳过。")
            continue
        out_dir = os.path.join(args.output_root, f"AOI{aoi_name}")
        os.makedirs(out_dir, exist_ok=True)

    # 遍历每一帧
    for i, r1_path in enumerate(r1_files):
        frame_num = args.start_frame + i
        print(f"\n处理文件 {i+1}/{num_to_process} -> 输出帧号 {frame_num:06d}")

        for aoi_name in args.aois:
            if aoi_name not in AOI_CONFIGS:
                continue
            cfg = AOI_CONFIGS[aoi_name]
            out_dir = os.path.join(args.output_root, f"AOI{aoi_name}")
            out_png = os.path.join(out_dir, f"frame{frame_num:06d}.png")

            success = crop_nitf_by_geo_bounds(
                r1_path, out_png,
                cfg['ulx'], cfg['uly'], cfg['lrx'], cfg['lry']
            )
            if not success:
                print(f"  AOI{aoi_name} 裁剪失败")

    print("\n所有任务完成！")

if __name__ == "__main__":
    main()