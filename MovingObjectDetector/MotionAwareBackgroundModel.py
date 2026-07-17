import cv2
import numpy as np
import skimage.measure as measure
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.MotionEstimator import MotionEstimator
import MovingObjectDetector.ImageProcFunc as ImageProcessing


class MotionAwareBackgroundModel(BackgroundModel):

    def __init__(self, num_of_template, templates, motion_method='farneback',
                 downsample_ratio=0.5, use_gpu=True,
                 motion_alpha=10.0, motion_thr=0.1):
        """
        motion_alpha : 运动信息强度 α，控制 (M - thr) 缩放
        motion_thr   : 运动阈值 τ
        融合公式: enhanced = D + α * (M - τ)
        """
        super().__init__(num_of_template, templates)
        self.motion_estimator = MotionEstimator(
            method=motion_method,
            use_gpu=use_gpu,
            downsample_ratio=downsample_ratio
        )
        self.motion_alpha = motion_alpha
        self.motion_thr = motion_thr
        self.prev_frame_for_flow = None
        self.flow_magnitude = None

    def doBackgroundSubtractionWithMotion(self, input_image, thres=6, compensate_brightness=True):
        if compensate_brightness:
            for i, thisTemplate in enumerate(self.CompensatedImages):
                diff = np.float64(input_image) - np.float64(thisTemplate)
                diff = cv2.GaussianBlur(diff, (21, 21), sigmaX=8)
                self.CompensatedImages[i] = np.uint8(np.clip(thisTemplate + diff, 0, 255))

        thisBackground = np.median(self.CompensatedImages, axis=0).astype(np.uint8)
        self.Background = thisBackground
        subtractionResult = cv2.absdiff(input_image, thisBackground)

        if self.prev_frame_for_flow is not None:
            flow = self.motion_estimator.compute_flow(input_image)
            self.flow_magnitude = self.motion_estimator.flow_to_magnitude(flow, smooth_sigma=3)
        else:
            self.flow_magnitude = np.ones_like(input_image, dtype=np.float32)

        # ---------- 加性独立融合 ----------
        diff_float = subtractionResult.astype(np.float32)
        # 运动偏移项：α * (M - τ)
        motion_offset = self.motion_alpha * (self.flow_magnitude - self.motion_thr)
        enhanced = diff_float + motion_offset

        if hasattr(self, 'invalidArea') and self.invalidArea is not None:
            enhanced[self.invalidArea] = diff_float[self.invalidArea]

        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        # ------------------------------------

        subtractionResultBW = np.uint8(enhanced >= thres)
        if hasattr(self, 'invalidArea') and self.invalidArea is not None:
            subtractionResultBW[self.invalidArea] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        subtractionResultBW = cv2.morphologyEx(subtractionResultBW, cv2.MORPH_OPEN, kernel)

        subtractionResultBW_ds = cv2.resize(
            subtractionResultBW,
            (int(subtractionResultBW.shape[1]/3), int(subtractionResultBW.shape[0]/3)),
            interpolation=cv2.INTER_NEAREST
        )
        r, c = np.where(subtractionResultBW_ds == 1)
        r = np.int32(3 * r + 1)
        c = np.int32(3 * c + 1)
        valid = (r >= 0) & (r < input_image.shape[0]) & (c >= 0) & (c < input_image.shape[1])
        CandiateRegionCentres = np.array(list(zip(r[valid], c[valid])))

        BackgroundSubtractionLabels = measure.label(subtractionResultBW, connectivity=1)
        BackgroundSubtractionProperties = measure.regionprops(BackgroundSubtractionLabels)

        self.prev_frame_for_flow = input_image.copy()
        return CandiateRegionCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels

    def doBackgroundSubtraction(self, input_image, thres=6, CompensateBrightness=True):
        return self.doBackgroundSubtractionWithMotion(input_image, thres, CompensateBrightness)

    def get_flow_magnitude(self):
        return self.flow_magnitude