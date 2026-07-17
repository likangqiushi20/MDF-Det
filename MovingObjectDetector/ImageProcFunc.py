import cv2
import numpy as np
import timeit

def CalcHomography(frame1, frame2, num_of_features=3000):
    starttime = timeit.default_timer()

    orb = cv2.ORB_create(nfeatures=num_of_features, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        print("ORB 特征点不足，无法计算单应性")
        return np.eye(3, dtype=np.float32), 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance <= 50]

    pts_src = []
    pts_dst = []
    for m in good_matches:
        pts_src.append(kp1[m.queryIdx].pt)
        pts_dst.append(kp2[m.trainIdx].pt)

    if len(pts_src) < 4:
        print(f"有效匹配点过少 ({len(pts_src)})，无法估计单应性")
        return np.eye(3, dtype=np.float32), 0.0

    if len(pts_src) <= 1000:
        print(f"警告：匹配特征点较少 ({len(pts_src)})，估计可能不可靠...")

    H, status = cv2.findHomography(
        np.array(pts_src), np.array(pts_dst),
        method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2000, confidence=0.995
    )

    if H is None:
        print("单应性计算失败，返回单位矩阵")
        return np.eye(3, dtype=np.float32), 0.0

    match_ratio = np.sum(status) / len(status) if len(status) > 0 else 0.0
    endtime = timeit.default_timer()
    return H, match_ratio


def ImageRegistration(srcImg, dstShape, H):
    starttime = timeit.default_timer()
    if dstShape is None:
        h, w = srcImg.shape[:2]
    else:
        h, w = dstShape[:2]

    im_out = cv2.warpPerspective(
        srcImg, H, (w, h),
        borderValue=255,
        flags=cv2.INTER_LINEAR
    )
    endtime = timeit.default_timer()
    return im_out