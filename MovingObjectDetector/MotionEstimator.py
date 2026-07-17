"""
MotionEstimator.py
光流计算与运动显著性图生成
支持 Farneback (CPU/GPU) 和快速帧差法
"""

import cv2
import numpy as np

class MotionEstimator:
    def __init__(self, method='farneback', use_gpu=True, downsample_ratio=0.5):
        self.method = method
        self.use_gpu = use_gpu
        self.downsample_ratio = downsample_ratio
        self.prev_gray = None
        self.proc_shape = None
        self.gpu_available = False

        if self.use_gpu:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.gpu_available = True
            except:
                self.gpu_available = False

        if self.use_gpu and not self.gpu_available:
            print("警告：GPU 光流不可用，回退到 CPU 模式。")

    def _get_proc_shape(self, original_shape):
        if self.downsample_ratio < 1.0:
            h, w = original_shape[:2]
            return int(round(h * self.downsample_ratio)), int(round(w * self.downsample_ratio))
        return original_shape[:2]

    def _preprocess(self, img):
        if self.downsample_ratio < 1.0:
            if self.proc_shape is None:
                self.proc_shape = self._get_proc_shape(img.shape)
            return cv2.resize(img, (self.proc_shape[1], self.proc_shape[0]), interpolation=cv2.INTER_LINEAR)
        return img

    def _postprocess(self, data, original_shape):
        if self.downsample_ratio < 1.0:
            h, w = original_shape[:2]
            return cv2.resize(data, (w, h), interpolation=cv2.INTER_LINEAR)
        return data

    def compute_flow(self, current_gray):
        original_shape = current_gray.shape
        curr_proc = self._preprocess(current_gray)

        if self.prev_gray is None:
            self.prev_gray = curr_proc
            return np.zeros((*original_shape, 2), dtype=np.float32)

        if self.prev_gray.shape != curr_proc.shape:
            curr_proc = cv2.resize(curr_proc, (self.prev_gray.shape[1], self.prev_gray.shape[0]))

        if self.method == 'farneback':
            if self.use_gpu and self.gpu_available:
                prev_gpu = cv2.cuda_GpuMat()
                curr_gpu = cv2.cuda_GpuMat()
                prev_gpu.upload(self.prev_gray)
                curr_gpu.upload(curr_proc)
                flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(
                    numLevels=5, pyrScale=0.5, fastPyramids=True,
                    winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0
                ).calc(prev_gpu, curr_gpu, None)
                flow = flow_gpu.download()
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_proc, None,
                    pyr_scale=0.5, levels=5, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
        elif self.method == 'frame_diff':
            diff = cv2.absdiff(self.prev_gray, curr_proc)
            flow = np.stack([diff, diff], axis=-1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.prev_gray = curr_proc
        flow = self._postprocess(flow, original_shape)
        return flow

    def flow_to_magnitude(self, flow, smooth_sigma=5):
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        if smooth_sigma > 0:
            mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=smooth_sigma)
        mag_max = mag.max()
        if mag_max > 1e-6:
            mag = mag / mag_max
        return mag.astype(np.float32)