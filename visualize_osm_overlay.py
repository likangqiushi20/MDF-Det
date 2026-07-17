import os
import osmnx as ox
import geopandas as gpd
import rasterio
import numpy as np
from rasterio import features
from shapely.ops import unary_union
from shapely.geometry import mapping, box
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ================== 配置路径 ==================
NTF_FILE = r"D:\Lkqs\Data\wami_data\WPAFB-21Oct2009-TRAIN_NITF_002\WPAFB-21Oct2009\Data\TRAIN\NITF\20091021202919-01000403-VIS.ntf.r1"
PNG_FILE = r"D:\Lkqs\sys\code\WPAFB2009\training\frame000403.png"

ROAD_BUFFERS = {
    'motorway': 15, 'trunk': 12, 'primary': 10,
    'secondary': 8, 'tertiary': 6, 'residential': 4,
    'service': 3, 'unclassified': 3
}

print("=" * 60)
print("开始处理...")

# ================== 检查文件 ==================
if not os.path.exists(NTF_FILE):
    raise FileNotFoundError(f"NTF文件不存在: {NTF_FILE}")
if not os.path.exists(PNG_FILE):
    raise FileNotFoundError(f"PNG文件不存在: {PNG_FILE}")
print("文件检查通过 ✓")

# ================== 读取 .ntf ==================
print("正在读取 .ntf 地理信息...")
with rasterio.open(NTF_FILE) as src:
    bounds = src.bounds
    crs = src.crs
    transform = src.transform
    ntf_height = src.height
    ntf_width = src.width
    print(f"  NTF尺寸: {ntf_width} x {ntf_height}")
    print(f"  地理范围: {bounds}")
    print(f"  CRS: {crs}")

# ================== 加载 PNG ==================
print("正在加载 PNG 图像...")
png_img = Image.open(PNG_FILE).convert('L')
png_width, png_height = png_img.size
print(f"  PNG尺寸: {png_width} x {png_height}")

SCALE_TO_PNG = (png_width != ntf_width) or (png_height != ntf_height)
if SCALE_TO_PNG:
    print("  ⚠️ PNG与NTF尺寸不同")

# ================== 从OSM下载道路 ==================
print("正在从 OSM 下载道路数据...")
west, south, east, north = bounds
buffer_deg = 0.001
bbox = (west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg)
print(f"  查询范围: {bbox}")

ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"
gdf_roads = ox.features_from_bbox(bbox, tags={'highway': True})
print(f"  下载到 {len(gdf_roads)} 条道路特征。")

# ================== 道路分类与缓冲 ==================
print("正在处理道路分类与缓冲...")
gdf_roads_filtered = gdf_roads[gdf_roads['highway'].isin(ROAD_BUFFERS.keys())].copy()
if len(gdf_roads_filtered) == 0:
    raise ValueError("该区域没有找到道路。")
print(f"  筛选后道路数量: {len(gdf_roads_filtered)}")

gdf_roads_filtered['buffer_dist'] = gdf_roads_filtered['highway'].map(ROAD_BUFFERS)
gdf_roads_filtered = gdf_roads_filtered[~gdf_roads_filtered.geometry.is_empty]
if len(gdf_roads_filtered) == 0:
    raise ValueError("所有道路几何体为空")

# 投影到UTM
utm_crs = gdf_roads_filtered.estimate_utm_crs()
print(f"  目标UTM投影: {utm_crs}")
gdf_proj = gdf_roads_filtered.to_crs(utm_crs)

# 缓冲
gdf_proj['geometry_buffered'] = gdf_proj.geometry.buffer(gdf_proj['buffer_dist'])
gdf_proj = gdf_proj[~gdf_proj['geometry_buffered'].is_empty]

if len(gdf_proj) == 0:
    raise ValueError("缓冲后没有有效的多边形。")

# 合并所有道路面（此时是UTM坐标）
road_polygons_utm = unary_union(gdf_proj.geometry_buffered)
print(f"  合并后的几何类型 (UTM): {road_polygons_utm.geom_type}")

if road_polygons_utm.is_empty:
    raise ValueError("合并后的道路多边形为空。")

# ================== 关键修复：将UTM几何体转换为WGS84 ==================
print("正在将道路多边形从 UTM 转换到 WGS84...")
# 使用 GeoSeries 实现几何体坐标转换
road_utm_series = gpd.GeoSeries([road_polygons_utm], crs=utm_crs)
road_wgs84_series = road_utm_series.to_crs("EPSG:4326")
road_polygons_wgs84 = road_wgs84_series.iloc[0]  # 提取单个几何体
print(f"  转换后的几何类型: {road_polygons_wgs84.geom_type}")

# ================== 提取多边形 ==================
def extract_polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    elif geom.geom_type == 'MultiPolygon':
        return list(geom.geoms)
    elif geom.geom_type == 'GeometryCollection':
        polys = []
        for g in geom.geoms:
            polys.extend(extract_polygons(g))
        return polys
    else:
        # 对于LineString等，尝试缓冲
        try:
            buffered = geom.buffer(2)
            if buffered.geom_type == 'Polygon':
                return [buffered]
            elif buffered.geom_type == 'MultiPolygon':
                return list(buffered.geoms)
            else:
                return []
        except:
            return []

polygons_to_raster = extract_polygons(road_polygons_wgs84)
print(f"  提取到 {len(polygons_to_raster)} 个多边形。")

if len(polygons_to_raster) == 0:
    # 备选：直接对原始线缓冲（但使用WGS84，缓冲单位是度，会很小）
    print("  警告：未提取到多边形，尝试直接缓冲原始线...")
    gdf_lines_wgs84 = gdf_roads_filtered.to_crs("EPSG:4326")
    # 在WGS84中缓冲，使用0.00005度（约5.5米）
    gdf_lines_wgs84['geom_buf'] = gdf_lines_wgs84.geometry.buffer(0.00005)
    gdf_lines_wgs84 = gdf_lines_wgs84[~gdf_lines_wgs84['geom_buf'].is_empty]
    if len(gdf_lines_wgs84) > 0:
        merged_buf = unary_union(gdf_lines_wgs84.geom_buf)
        polygons_to_raster = extract_polygons(merged_buf)
        print(f"  备选提取到 {len(polygons_to_raster)} 个多边形。")

if len(polygons_to_raster) == 0:
    raise ValueError("没有有效的多边形可用于栅格化。")

# ================== 诊断：检查转换后多边形范围 ==================
print("\n" + "=" * 60)
print("🔍 坐标对齐诊断")
print("=" * 60)
first_poly = polygons_to_raster[0]
poly_bounds = first_poly.bounds
print(f"  转换后多边形地理范围: {poly_bounds}")
print(f"  图像地理边界: {bounds}")

# 检查多边形是否与图像边界有重叠
img_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
overlap = first_poly.intersects(img_box)
print(f"  多边形与图像有重叠吗? {overlap}")
if not overlap:
    print("  ⚠️ 警告：多边形完全在图像范围之外！")
    print("  可能原因：CRS转换失败或图像边界有误。")
else:
    print("  ✅ 多边形与图像有重叠，坐标对齐正确。")

# ================== 栅格化 ==================
print("\n" + "=" * 60)
print("正在生成道路掩码...")
print("=" * 60)

shapes = [(mapping(poly), 1) for poly in polygons_to_raster if not poly.is_empty]

if len(shapes) == 0:
    raise ValueError("没有有效的形状用于栅格化。")

mask_ntf = features.rasterize(
    shapes=shapes,
    out_shape=(ntf_height, ntf_width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

print(f"  掩码最大值: {mask_ntf.max()}")
print(f"  掩码唯一值: {np.unique(mask_ntf)}")
print(f"  道路像素占比: {mask_ntf.mean():.4f}")

if mask_ntf.max() == 0:
    print("\n⚠️ 警告：掩码全为0，说明多边形可能太小（缓冲不足）或坐标仍有偏差。")
    print("建议：适当增加缓冲区宽度再试。")

# ================== 缩放与可视化 ==================
if SCALE_TO_PNG:
    from PIL import Image as PILImage
    print(f"  缩放掩码至 PNG 尺寸...")
    mask_pil = PILImage.fromarray(mask_ntf * 255)
    mask_pil_resized = mask_pil.resize((png_width, png_height), PILImage.NEAREST)
    mask_png = np.array(mask_pil_resized) / 255.0
else:
    mask_png = mask_ntf.astype(np.float32)

print("\n正在生成可视化图像...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(png_img, cmap='gray')
axes[0].set_title('Original PNG')
axes[0].axis('off')

axes[1].imshow(mask_png, cmap='gray')
axes[1].set_title('Road Mask')
axes[1].axis('off')

axes[2].imshow(png_img, cmap='gray')
axes[2].imshow(mask_png, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('road_overlay_result.png', dpi=150, bbox_inches='tight')
print("\n✅ 可视化结果已保存为 'road_overlay_result.png'")
print("=" * 60)
plt.show()