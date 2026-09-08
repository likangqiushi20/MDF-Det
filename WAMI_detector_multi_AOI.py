"""
WAMI_detector_multi_AOI.py
同时对多个 AOI 进行推理，支持模块开关与参数调节（加性独立融合公式）。
输出增加：
    - 每个 AOI 的平均每帧耗时
    - 总耗时及平均每帧耗时
    - 内存使用情况（需安装 psutil，否则仅显示提示）
"""

import argparse
import numpy as np
import cv2
import os
import timeit
import tensorflow as tf
import pandas as pd
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
from osgeo import gdal, osr

# 尝试导入 psutil 用于内存监测
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from MovingObjectDetector.MotionAwareBackgroundModel import MotionAwareBackgroundModel
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
from MovingObjectDetector.ImageProcFunc import CalcHomography

# --------------------- 地理工具函数 ---------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

AOI_GEO_BOUNDS = {
    '01': {'ulx': -84.125664, 'uly': 39.771185, 'lrx': -84.119895, 'lry': 39.765354},
    '02': {'ulx': -84.122165, 'uly': 39.786161, 'lrx': -84.116273, 'lry': 39.780145},
    '03': {'ulx': -84.101297, 'uly': 39.781737, 'lrx': -84.095836, 'lry': 39.775665},
    '04': {'ulx': -84.122289, 'uly': 39.766089, 'lrx': -84.116211, 'lry': 39.760628},
    '34': {'ulx': -84.126152, 'uly': 39.783150, 'lrx': -84.114983, 'lry': 39.776215},
    '40': {'ulx': -84.127380, 'uly': 39.772167, 'lrx': -84.118790, 'lry': 39.765719},
    '41': {'ulx': -84.121796, 'uly': 39.772351, 'lrx': -84.113391, 'lry': 39.764676},
}

def load_gt(truth_csv, aoi_bounds, frame_range, min_move_m=0.8):
    df = pd.read_csv(truth_csv)
    frame_col = 'FRAME_NUMBER'
    track_col = 'id'
    lat_col, lon_col = 'LATITUDE', 'LONGITUDE'
    df = df[(df[frame_col] >= frame_range[0]) & (df[frame_col] <= frame_range[1])]
    grouped = df.groupby(track_col)
    static_set = set()
    for tid, group in grouped:
        group = group.sort_values(frame_col)
        frames = group[frame_col].values
        lats = group[lat_col].values
        lons = group[lon_col].values
        if len(frames) < 2: continue
        for i in range(len(frames)-1):
            dist = haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1])
            if dist < min_move_m:
                static_set.add((frames[i], tid))
                static_set.add((frames[i+1], tid))
    def is_moving(row): return (row[frame_col], row[track_col]) not in static_set
    df_moving = df[df.apply(is_moving, axis=1)]
    min_lon, max_lon = min(aoi_bounds['ulx'], aoi_bounds['lrx']), max(aoi_bounds['ulx'], aoi_bounds['lrx'])
    min_lat, max_lat = min(aoi_bounds['uly'], aoi_bounds['lry']), max(aoi_bounds['uly'], aoi_bounds['lry'])
    df_moving = df_moving[(df_moving[lon_col] >= min_lon) & (df_moving[lon_col] <= max_lon) &
                          (df_moving[lat_col] >= min_lat) & (df_moving[lat_col] <= max_lat)]
    truth = defaultdict(list)
    for _, row in df_moving.iterrows():
        truth[int(row[frame_col])].append((row[lon_col], row[lat_col]))
    return dict(truth)

def pixel_to_geo(png_path, x, y):
    ds = gdal.Open(png_path)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    xp = gt[0] + x*gt[1] + y*gt[2]
    yp = gt[3] + x*gt[4] + y*gt[5]
    if proj:
        src = osr.SpatialReference(); src.ImportFromWkt(proj)
        dst = osr.SpatialReference(); dst.SetWellKnownGeogCS("WGS84")
        t = osr.CoordinateTransformation(src, dst)
        lon, lat, _ = t.TransformPoint(xp, yp)
        return lon, lat
    return xp, yp

def geo_to_pixel(png_path, lon, lat):
    ds = gdal.Open(png_path)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    if proj:
        src = osr.SpatialReference(); src.SetWellKnownGeogCS("WGS84")
        dst = osr.SpatialReference(); dst.ImportFromWkt(proj)
        t = osr.CoordinateTransformation(src, dst)
        xp, yp, _ = t.TransformPoint(lon, lat)
    else: xp, yp = lon, lat
    inv = gdal.InvGeoTransform(gt)
    col = int(inv[0] + inv[1]*xp + inv[2]*yp)
    row = int(inv[3] + inv[4]*xp + inv[5]*yp)
    return col, row

def match_geo(det_lonlats, truth_lonlats, thresh_m=5.0):
    if not det_lonlats and not truth_lonlats: return 0,0,0
    if not det_lonlats: return 0,0,len(truth_lonlats)
    if not truth_lonlats: return 0,len(det_lonlats),0
    matched = [False]*len(truth_lonlats)
    tp = 0
    for dl, dlat in det_lonlats:
        best = float('inf'); best_idx = -1
        for i, (gl, glt) in enumerate(truth_lonlats):
            if matched[i]: continue
            d = haversine_distance(dlat, dl, glt, gl)
            if d < best: best = d; best_idx = i
        if best_idx != -1 and best <= thresh_m:
            matched[best_idx] = True; tp += 1
    fp = len(det_lonlats) - tp
    fn = sum(1 for m in matched if not m)
    return tp, fp, fn

def merge_close_points(points, dist):
    if len(points) == 0: return []
    clusters = []
    for x, y in points:
        best = float('inf'); best_idx = -1
        for i, (sx, sy, cnt) in enumerate(clusters):
            cx, cy = sx/cnt, sy/cnt
            d = np.hypot(x-cx, y-cy)
            if d < best: best = d; best_idx = i
        if best_idx != -1 and best < dist:
            sx, sy, cnt = clusters[best_idx]
            clusters[best_idx] = (sx+x, sy+y, cnt+1)
        else:
            clusters.append((x, y, 1))
    return [(sx/cnt, sy/cnt) for sx, sy, cnt in clusters]

def merge_close_scored_points(points, dist):
    """Merge points as before while retaining the maximum regression score."""
    if len(points) == 0: return []
    clusters = []
    for x, y, score in points:
        best = float('inf'); best_idx = -1
        for i, (sx, sy, cnt, max_score) in enumerate(clusters):
            cx, cy = sx/cnt, sy/cnt
            d = np.hypot(x-cx, y-cy)
            if d < best: best = d; best_idx = i
        if best_idx != -1 and best < dist:
            sx, sy, cnt, max_score = clusters[best_idx]
            clusters[best_idx] = (sx+x, sy+y, cnt+1, max(max_score, score))
        else:
            clusters.append((x, y, 1, score))
    return [(sx/cnt, sy/cnt, max_score) for sx, sy, cnt, max_score in clusters]

def filter_by_border(points, shape, border):
    h, w = shape[:2]
    return [(x, y) for x, y in points if border <= x < w-border and border <= y < h-border]

def filter_by_scene_prior(det_list, prior_map, img_shape, thresh=0.1,
                          confidence_bypass=None, soft_alpha=None,
                          soft_threshold=None):
    if len(det_list) == 0: return []
    if prior_map.shape[:2] != img_shape[:2]:
        prior_map = cv2.resize(prior_map, (img_shape[1], img_shape[0]))
    filtered = []
    for d in det_list:
        r, c = d['centre']
        prior_score = float(prior_map[int(r), int(c)])
        confidence = float(d.get('confidence', 0.0))
        if soft_alpha is not None:
            final_score = confidence * (soft_alpha + (1.0 - soft_alpha) * prior_score)
            if final_score >= soft_threshold:
                filtered.append(d)
            continue
        prior_ok = prior_score >= thresh
        confidence_ok = (confidence_bypass is not None and
                         confidence >= confidence_bypass)
        if prior_ok or confidence_ok:
            filtered.append(d)
    return filtered

def process_one_aoi(aoi, png_folder, out_dir, params, models):
    model_bin, aveImg_bin, model_reg, aveImg_reg = models
    static_prior = params.get('static_prior', None)

    frame_nums = list(range(params['start_frame'], params['start_frame'] + params['num_frames']))
    aoi_bounds = AOI_GEO_BOUNDS[aoi]
    truth_by_frame = load_gt(params['truth_csv'], aoi_bounds,
                             (frame_nums[0], frame_nums[-1]),
                             min_move_m=params['min_move_m'])

    csv_dir = os.path.join(out_dir, "CSV")
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    templates = []
    for i in range(params['num_templates']):
        png_path = os.path.join(png_folder, f"frame{frame_nums[i]:06d}.png")
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(png_path)
        templates.append(img)

    if params['use_motion']:
        bgt = MotionAwareBackgroundModel(
            num_of_template=params['num_templates'], templates=templates,
            motion_method='farneback',
            downsample_ratio=0.5, use_gpu=True,
            motion_alpha=params['motion_alpha'],
            motion_thr=params['motion_thr'])
        bgt.blur_kernel = 21
        bgt.blur_sigma = 8.0
    else:
        bgt = BackgroundModel(num_of_template=params['num_templates'],
                              templates=templates)

    total_tp = total_fp = total_fn = 0
    total_gt = total_det = 0
    border = params['border_exclude']

    # 计时相关
    total_time = 0.0
    proc_frames = len(frame_nums) - params['num_templates']
    
    for idx in range(params['num_templates'], len(frame_nums)):
        frame_num = frame_nums[idx]
        png_path = os.path.join(png_folder, f"frame{frame_num:06d}.png")
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        t_start = timeit.default_timer()

        prior_map_full = None
        if params['use_scene_prior'] and static_prior is not None:
            prior_map_full = cv2.resize(static_prior, (img.shape[1], img.shape[0]))

        Hs = []
        for i in range(params['num_templates']):
            H, _ = CalcHomography(templates[i], img, num_of_features=params['feature_count'])
            Hs.append(H if H is not None else np.eye(3, dtype=np.float32))
        bgt.Hs = Hs
        bgt.doMotionCompensationAndValidArea(img, Hs, img.shape)

        cand_centres, bg_props, bg_labels = bgt.doBackgroundSubtraction(img, thres=params['BSThreshold'])

        dr = DetectionRefinement(img, bgt.getCompensatedImages(), cand_centres,
                                 bg_props, bg_labels,
                                 (model_bin, aveImg_bin, model_reg, aveImg_reg),
                                 classifier_threshold=params['classifier_threshold'],
                                 regression_threshold=params['regression_threshold'])
        det1, det2, _ = dr.do_refine_bs()
        dets = det1 + det2

        if params['use_scene_prior'] and prior_map_full is not None:
            dets = filter_by_scene_prior(
                dets, prior_map_full, img.shape, params['prior_threshold'],
                params.get('prior_confidence_bypass'),
                params.get('prior_soft_alpha'),
                params.get('prior_soft_threshold'))

        det_pixels_all = [(int(d['centre'][1]), int(d['centre'][0]),
                           float(d.get('confidence', 1.0))) for d in dets]
        h, w = img.shape[:2]
        det_pixels_border = [(x, y, score) for x, y, score in det_pixels_all
                             if border <= x < w-border and border <= y < h-border]
        det_pixels_merged_scored = merge_close_scored_points(
            det_pixels_border, params['merge_distance_pixel'])
        det_pixels_merged = [(x, y) for x, y, _ in det_pixels_merged_scored]
        det_scores_merged = [score for _, _, score in det_pixels_merged_scored]

        truth_lonlats_all = truth_by_frame.get(frame_num, [])
        truth_pixels_all = []
        for lon, lat in truth_lonlats_all:
            res = geo_to_pixel(png_path, lon, lat)
            if res is not None: truth_pixels_all.append(res)
        truth_pixels_border = filter_by_border(truth_pixels_all, img.shape, border)

        det_lonlats = []
        for x, y in det_pixels_merged:
            try: det_lonlats.append(pixel_to_geo(png_path, x, y))
            except: pass

        valid_truth = []
        for lon, lat in truth_lonlats_all:
            col, row = geo_to_pixel(png_path, lon, lat)
            if (col, row) in truth_pixels_border: valid_truth.append((lon, lat))

        tp, fp, fn = match_geo(det_lonlats, valid_truth, params['match_distance_m'])
        total_tp += tp; total_fp += fp; total_fn += fn
        total_gt += len(valid_truth); total_det += len(det_lonlats)

        csv_path = os.path.join(csv_dir, f"frame{frame_num:06d}.csv")
        if det_lonlats:
            scored_lonlats = [(lon, lat, score) for (lon, lat), score
                              in zip(det_lonlats, det_scores_merged)]
            np.savetxt(csv_path, np.array(scored_lonlats), delimiter=',', fmt='%.6f',
                       header='lon,lat,confidence', comments='')
        else:
            np.savetxt(csv_path, np.empty((0,3)), delimiter=',', fmt='%.6f',
                       header='lon,lat,confidence', comments='')

        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if border > 0:
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (128, 128, 128), border * 2)
            vis_img = cv2.addWeighted(overlay, 0.2, vis_img, 0.8, 0)
            cv2.rectangle(vis_img, (border, border),
                          (img.shape[1] - border, img.shape[0] - border),
                          (0, 255, 0), 1)
        for (x, y) in det_pixels_merged:
            cv2.circle(vis_img, (int(x), int(y)), 6, (0, 0, 255), 2)
        for (x, y) in truth_pixels_border:
            cv2.circle(vis_img, (int(x), int(y)), 4, (0, 255, 255), 1)
        vis_path = os.path.join(vis_dir, f"frame{frame_num:06d}.png")
        cv2.imwrite(vis_path, vis_img)

        bgt.updateTemplate(img)
        templates.pop(0); templates.append(img)

        t_end = timeit.default_timer()
        total_time += (t_end - t_start)

    precision = total_tp/(total_tp+total_fp) if total_tp+total_fp else 0.0
    recall = total_tp/(total_tp+total_fn) if total_tp+total_fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    avg_time = total_time / proc_frames if proc_frames > 0 else 0.0
    return precision, recall, f1, total_gt, total_det, total_tp, total_fp, total_fn, total_time, avg_time, proc_frames


def main():
    parser = argparse.ArgumentParser(description='Multi-AOI WAMI Detector with Additive Motion Fusion')
    parser.add_argument('--aoi_list', nargs='+', default=['01','02','03','34','40','41'])
    parser.add_argument('--png_root', required=True)
    parser.add_argument('--png-folder', default=None,
                        help='Direct AOI image directory override; use only with one AOI in --aoi_list')
    parser.add_argument('--truth_csv', required=True)
    parser.add_argument('--start_frame', type=int, default=295)
    parser.add_argument('--num_frames', type=int, default=25)
    parser.add_argument('--output_base', required=True)
    parser.add_argument('--binary_model_dir', default='Models/')
    parser.add_argument('--regression_model', default='regression_spatial_attention.h5')
    parser.add_argument('--regression_norm', default='regression_norm_params.npz')
    parser.add_argument('--classifier-threshold', type=float, default=0.7,
                        help='Binary-classifier acceptance threshold')
    parser.add_argument('--regression-threshold', type=float, default=0.17,
                        help='Regression heatmap peak threshold')
    parser.add_argument('--static-prior-map', default=None,
                        help='SPGF-V4 HxW .npy prior generated once for the fixed AOI')
    parser.add_argument('--prior-dilation-radius', type=int, default=0,
                        help='Dilate a static prior map by this many pixels before filtering')
    parser.add_argument('--full-aoi-prior-model', default=None,
                        help='Native-resolution SPGF-V4 Keras model; predicts one fixed AOI prior from --full-aoi-prior-context')
    parser.add_argument('--full-aoi-prior-context', default=None,
                        help='Fixed context_background.png used by --full-aoi-prior-model')

    # Explicit negative switches are required for paired revision ablations.
    # Python 3.7-compatible form (the published wami_detector environment).
    parser.add_argument('--use-motion', '--use_motion', dest='use_motion', action='store_true', default=True)
    parser.add_argument('--no-motion', '--no_motion', dest='use_motion', action='store_false')
    parser.add_argument('--use-scene-prior', '--use_scene_prior', dest='use_scene_prior', action='store_true', default=False)
    parser.add_argument('--no-scene-prior', '--no_scene_prior', dest='use_scene_prior', action='store_false')

    parser.add_argument('-T', '--BSThreshold', type=float, default=2)
    parser.add_argument('--motion_alpha', type=float, default=10.0, help='加性运动权重 α')
    parser.add_argument('--motion_thr', type=float, default=0.1, help='运动阈值 τ')
    parser.add_argument('--prior_threshold', type=float, default=0)
    parser.add_argument('--prior-confidence-bypass', type=float, default=None,
                        help='Keep detections at or above this regression confidence even when the prior is low')
    parser.add_argument('--prior-soft-alpha', type=float, default=None,
                        help='Use Creg*(alpha+(1-alpha)*Pscene) soft fusion instead of hard filtering')
    parser.add_argument('--prior-soft-threshold', type=float, default=None,
                        help='Final score threshold for --prior-soft-alpha')
    parser.add_argument('--num_templates', type=int, default=5)
    parser.add_argument('--feature_count', type=int, default=2000)
    parser.add_argument('--border_exclude', type=int, default=50)
    parser.add_argument('--match_distance_m', type=float, default=5.0)
    parser.add_argument('--min_move_m', type=float, default=1.2)
    parser.add_argument('--merge_distance_pixel', type=float, default=6.0)
    args = parser.parse_args()
    if (args.static_prior_map or args.full_aoi_prior_model or
            args.full_aoi_prior_context) and len(args.aoi_list) != 1:
        parser.error('A fixed SPGF prior/model context is AOI-specific; provide '
                     'exactly one AOI in --aoi_list')
    if (args.prior_soft_alpha is None) != (args.prior_soft_threshold is None):
        parser.error('--prior-soft-alpha and --prior-soft-threshold must be supplied together')
    if args.prior_soft_alpha is not None and not 0.0 <= args.prior_soft_alpha <= 1.0:
        parser.error('--prior-soft-alpha must be within [0,1]')

    # 内存初始值
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"程序启动时内存占用: {mem_start:.2f} MB")
    else:
        print("psutil 未安装，无法监测内存。")

    print("加载模型...")
    import TrainNetwork.BaseFunctions as bf
    model_bin, aveImg_bin, _, _ = bf.ReadModels(args.binary_model_dir)
    model_reg = tf.keras.models.load_model(args.regression_model, compile=False)
    reg_norm = np.load(args.regression_norm)
    aveImg_reg = None
    for k in ['mean_img','mean','mu']:
        if k in reg_norm: aveImg_reg = reg_norm[k]; break
    if aveImg_reg is None: raise KeyError("回归均值未找到")

    static_prior = None
    if args.full_aoi_prior_model or args.full_aoi_prior_context:
        if not (args.full_aoi_prior_model and args.full_aoi_prior_context):
            raise ValueError('--full-aoi-prior-model and --full-aoi-prior-context must be supplied together')
        if args.static_prior_map:
            raise ValueError('--static-prior-map cannot be combined with --full-aoi-prior-model')
        context = cv2.imread(args.full_aoi_prior_context, cv2.IMREAD_GRAYSCALE)
        if context is None:
            raise FileNotFoundError(args.full_aoi_prior_context)
        print('Loading native-resolution full-AOI scene-prior network...')
        from SPGF.model import GroupNormalization
        full_prior_net = tf.keras.models.load_model(
            args.full_aoi_prior_model,
            custom_objects={'GroupNormalization': GroupNormalization}, compile=False)
        expected = tuple(full_prior_net.input_shape[1:3])
        if all(value is not None for value in expected) and context.shape != expected:
            raise ValueError(f'Full-AOI prior context {context.shape} does not match model input {expected}')
        prediction = full_prior_net.predict(
            context[None, ..., None].astype(np.float32) / 255.0, verbose=0)[0]
        if prediction.ndim != 3 or prediction.shape[-1] != 2:
            raise ValueError('SPGF-V4 must output access and motion probability channels')
        static_prior = np.sqrt(np.clip(prediction[..., 0] * prediction[..., 1], 0, 1))
        print(f'Native-resolution prior ready: shape={static_prior.shape}, mean={static_prior.mean():.4f}')
    elif args.static_prior_map:
        static_prior = np.asarray(np.load(args.static_prior_map), np.float32)
        if static_prior.ndim != 2:
            raise ValueError('--static-prior-map must contain one HxW array')
        if args.prior_dilation_radius < 0:
            raise ValueError('--prior-dilation-radius must be non-negative')
        if args.prior_dilation_radius > 0:
            radius = args.prior_dilation_radius
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            static_prior = cv2.dilate(static_prior, kernel)
        print('Loading static scene-occurrence prior...')
    elif args.use_scene_prior:
        parser.error('--use-scene-prior requires --static-prior-map or both '
                     '--full-aoi-prior-model and --full-aoi-prior-context')

    models = (model_bin, aveImg_bin, model_reg, aveImg_reg)

    common_params = dict(
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        BSThreshold=args.BSThreshold,
        motion_alpha=args.motion_alpha,
        motion_thr=args.motion_thr,
        prior_threshold=args.prior_threshold,
        prior_confidence_bypass=args.prior_confidence_bypass,
        prior_soft_alpha=args.prior_soft_alpha,
        prior_soft_threshold=args.prior_soft_threshold,
        num_templates=args.num_templates,
        feature_count=args.feature_count,
        border_exclude=args.border_exclude,
        match_distance_m=args.match_distance_m,
        min_move_m=args.min_move_m,
        merge_distance_pixel=args.merge_distance_pixel,
        classifier_threshold=args.classifier_threshold,
        regression_threshold=args.regression_threshold,
        truth_csv=args.truth_csv,
        use_motion=args.use_motion,
        use_scene_prior=(args.use_scene_prior or static_prior is not None),
        static_prior=static_prior,
    )

    aoi_metrics = {}
    all_tp = all_fp = all_fn = 0
    all_gt = all_det = 0
    total_processing_time = 0.0
    total_processing_frames = 0

    for aoi in args.aoi_list:
        if args.png_folder:
            if len(args.aoi_list) != 1:
                raise ValueError('--png-folder is valid only when exactly one AOI is requested')
            png_folder = args.png_folder
        else:
            png_folder = os.path.join(args.png_root, f"AOI{aoi}")
        out_dir = os.path.join(args.output_base, f"AOI{aoi}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n#### 处理 AOI {aoi} ####")
        p, r, f1, gt, det, tp, fp, fn, aoi_time, aoi_avg_time, aoi_frames = process_one_aoi(
            aoi, png_folder, out_dir, common_params, models
        )
        aoi_metrics[aoi] = (p, r, f1, gt, det, tp, fp, fn)
        all_tp += tp; all_fp += fp; all_fn += fn
        all_gt += gt; all_det += det
        total_processing_time += aoi_time
        total_processing_frames += aoi_frames

        with open(os.path.join(out_dir, "metrics.txt"), 'w') as f:
            f.write(f"AOI {aoi}\n")
            f.write(f"Precision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\n")
            f.write(f"GT={gt}, Det={det}, TP={tp}, FP={fp}, FN={fn}\n")
            f.write(f"Proc. frames: {aoi_frames}\n")
            f.write(f"Total time: {aoi_time:.2f} seconds\n")
            f.write(f"Avg. time per frame: {aoi_avg_time:.2f} seconds\n")

    overall_p = all_tp/(all_tp+all_fp) if all_tp+all_fp > 0 else 0.0
    overall_r = all_tp/(all_tp+all_fn) if all_tp+all_fn > 0 else 0.0
    overall_f1 = 2*overall_p*overall_r/(overall_p+overall_r) if overall_p+overall_r > 0 else 0.0
    overall_avg_time = total_processing_time / total_processing_frames if total_processing_frames > 0 else 0.0

    summary_path = os.path.join(args.output_base, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("========== Multi-AOI Evaluation Summary ==========\n")
        for aoi in args.aoi_list:
            p, r, f1, gt, det, tp, fp, fn = aoi_metrics[aoi]
            f.write(f"AOI {aoi}: P={p:.4f}, R={r:.4f}, F1={f1:.4f} | GT={gt}, Det={det}, TP={tp}, FP={fp}, FN={fn}\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Overall: P={overall_p:.4f}, R={overall_r:.4f}, F1={overall_f1:.4f}\n")
        f.write(f"Total GT={all_gt}, Total Det={all_det}, TP={all_tp}, FP={all_fp}, FN={all_fn}\n")
        f.write(f"Total processed frames: {total_processing_frames}\n")
        f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
        f.write(f"Overall average time per frame: {overall_avg_time:.2f} seconds\n")
        if HAS_PSUTIL:
            mem_end = process.memory_info().rss / (1024 * 1024)
            f.write(f"Memory usage - start: {mem_start:.2f} MB, end: {mem_end:.2f} MB, delta: {mem_end - mem_start:.2f} MB\n")

    print("\n" + "="*50)
    print(f"Overall Precision: {overall_p:.4f}")
    print(f"Overall Recall:    {overall_r:.4f}")
    print(f"Overall F1-score:  {overall_f1:.4f}")
    print(f"Total processed frames: {total_processing_frames}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Overall average time per frame: {overall_avg_time:.2f} seconds")
    if HAS_PSUTIL:
        mem_end = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage - start: {mem_start:.2f} MB, end: {mem_end:.2f} MB, delta: {mem_end - mem_start:.2f} MB")
    print(f"汇总文件已保存至 {summary_path}")

if __name__ == "__main__":
    main()
