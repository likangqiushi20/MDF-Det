import numpy as np
import cv2
import matplotlib.pyplot as plt
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
import TrainNetwork.BaseFunctions as basefunctions
import timeit
from copy import copy
from MovingObjectDetector.BaseFunctions import TimePropagate, TimePropagate_, draw_error_ellipse2d
import hdf5storage
import os
import tensorflow as tf

# 导入全局上下文网络
from global_context_net import build_global_context_net

# 配置参数
input_image_idx = 10          # 起始帧索引
image_idx_offset = 0
num_of_template = 4           # 背景模板数（论文中为3-4）
imagefolder = "E:/WPAFB-images/training/"
writeimagefolder = "E:/WPAFB-detections/savefig/"
model_folder = "C:/Users/yifan/Google Drive/PythonSync/wasabi-detection-python/Models/"

# 加载原始模型（二分类CNN、回归CNN等）
model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)
model = (model_binary, aveImg_binary, model_regression, aveImg_regression)

# 加载全局上下文网络（轻量级GCNet）
# 注意：这里使用与训练时相同的网络结构，如果训练时保存了权重，应加载权重
global_net = build_global_context_net(target_size=512)
# 如果训练时保存了全局网络的权重，可以取消注释加载
# global_net.load_weights("global_context_weights.h5")

# 加载分类CNN的归一化参数（由训练脚本保存）
norm_params = np.load("classifier_norm_params.npz")   # 包含 mean_img, mean_global

# 加载预计算的变换矩阵（可选，用于配准）
matlabfile = hdf5storage.loadmat('C:/Users/yifan/Google Drive/PythonSync/wasabi-detection-python/Models/Data/TransformationMatrices_train.mat')
TransformationMatrices = matlabfile.get("TransMatrix")

# 初始化背景模型（使用前 num_of_template 帧）
images = []
for i in range(num_of_template):
    frame_idx = input_image_idx + image_idx_offset + i - num_of_template
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    images.append(ReadImage)
bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

# 主循环：处理20帧
for i in range(20):
    starttime = timeit.default_timer()
    
    # 读取当前帧
    frame_idx = input_image_idx + image_idx_offset + i
    input_image = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    
    # 更新单应性矩阵（使用预计算的变换矩阵）
    Hs = bgt.doUpdateHomography(TransformationMatrices, frame_idx-1)
    # 可选：实时计算单应性矩阵（较慢）
    # Hs = bgt.doCalculateHomography(input_image)
    
    # 运动补偿，获取对齐的历史帧
    bgt.doMotionCompensationAndValidArea(input_image, Hs, input_image.shape)
    
    # 背景减除，获取候选中心、连通区域属性等
    CandiateCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels = bgt.doBackgroundSubtraction(
        input_image, thres=8, CompensateBrightness=False)
    print("background subtraction finished...")
    
    # ========== 新增：提取当前帧的全局特征 ==========
    # 输入图像需为 (H,W,1) 并添加 batch 维度
    img_for_global = np.expand_dims(input_image, axis=-1)   # (H,W,1)
    img_for_global = np.expand_dims(img_for_global, axis=0) # (1,H,W,1)
    global_feature = global_net.predict(img_for_global)[0]  # (64,)
    # ================================================
    
    # 创建 DetectionRefinement 实例，传入全局特征和归一化参数
    dr = DetectionRefinement(
        input_image, bgt.getCompensatedImages(), CandiateCentres,
        BackgroundSubtractionProperties, BackgroundSubtractionLabels, model,
        global_net=global_net,
        global_feature=global_feature,
        norm_params=norm_params
    )
    Detections1, Detections2, RefinedCentres = dr.do_refine_bs()
    
    # 转换检测结果为可视化格式
    Detections1_for_img = [[ele["centre"][1], ele["centre"][0]] for ele in Detections1]
    Detections1_for_img = np.int64(np.asarray(Detections1_for_img))
    Detections2_for_img = [[ele["centre"][1], ele["centre"][0]] for ele in Detections2]
    Detections2_for_img = np.int64(np.asarray(Detections2_for_img))
    
    # 绘制检测结果到输出图像
    output_image = copy(input_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    for thisDetection in Detections1_for_img:
        if thisDetection[0] > 0 and thisDetection[0] < input_image.shape[1] and thisDetection[1] > 0 and thisDetection[1] < input_image.shape[0]:
            cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 5, (0, 200, 0), 1)
    for thisDetection in Detections2_for_img:
        if thisDetection[0] > 0 and thisDetection[0] < input_image.shape[1] and thisDetection[1] > 0 and thisDetection[1] < input_image.shape[0]:
            cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 5, (0, 0, 200), 1)
    
    print("Draw image finished...")
    print("Write image to " + writeimagefolder + "%05d.jpg" % frame_idx)
    cv2.imwrite(writeimagefolder + "%05d.jpg" % (frame_idx), output_image)
    
    # 更新背景模型（将当前帧加入模板队列）
    bgt.updateTemplate(input_image)
    
    endtime = timeit.default_timer()
    print("Processing Time (Total): " + str(endtime - starttime) + " s... ")
