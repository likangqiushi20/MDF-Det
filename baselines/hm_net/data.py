"""Real HM-Net frame pairs and augmented full-resolution crop targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from torch import Tensor

from hm_net.sgr import augment_feedback_centers, render_weighted_centers
from hm_net.targets import ClassifiedPoint, build_targets
from hmrn.data import select_crop_origin
from common.data.frame_cache import FixedGridFrameCache, IndexedNpzTruth


@dataclass(frozen=True)
class HMNetSample:
    current: Tensor
    previous: Tensor
    feedback: Tensor
    center: Tensor
    motion: Tensor
    precision: Tensor
    motion_mask: Tensor
    precision_mask: Tensor
    origin_xy: tuple[int, int]


def _normalize(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    return (
        values / float(np.iinfo(image.dtype).max)
        if np.issubdtype(image.dtype, np.integer)
        else values
    )


class HMNetData:
    def __init__(
        self, manifest_path: Path, grids_path: Path, truth_cache_path: Path
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixed = json.loads(grids_path.read_text(encoding="utf-8"))
        profile = fixed["papers"]["hm_net"]
        self.grids = profile["grids"]
        self.level = profile["level"]
        self.root = Path(self.manifest["dataset_root"])
        self.lookup = {
            (frame["set"], frame["frame_number"]): frame
            for frame in self.manifest["frames"]
        }
        self.truth = np.load(truth_cache_path)

    def read_frame(self, split: str, frame: int, aoi: str) -> np.ndarray:
        grid = self.grids[aoi]
        relative = self.lookup[(split, frame)]["levels"][self.level]["path"]
        with rasterio.open(self.root / relative) as source:
            with WarpedVRT(
                source,
                crs=grid["crs"],
                transform=Affine(*grid["transform"]),
                width=grid["width"],
                height=grid["height"],
                resampling=Resampling.bilinear,
                nodata=0,
            ) as vrt:
                return vrt.read(1)

    def points(
        self, split: str, frame: int, aoi: str
    ) -> list[ClassifiedPoint]:
        code = 0 if split == "train" else 1
        mask = (
            (self.truth["split"] == code)
            & (self.truth["frame"] == frame)
            & (self.truth["aoi"] == int(aoi))
        )
        return [
            ClassifiedPoint(int(track_id), float(x), float(y), int(class_id))
            for track_id, x, y, class_id in zip(
                self.truth["track_id"][mask],
                self.truth["x"][mask],
                self.truth["y"][mask],
                self.truth["class_id"][mask],
            )
        ]

    def read_pair(
        self, split: str, frame: int, aoi: str
    ) -> tuple[np.ndarray, np.ndarray, list[ClassifiedPoint], list[ClassifiedPoint]]:
        return (
            self.read_frame(split, frame - 1, aoi),
            self.read_frame(split, frame, aoi),
            self.points(split, frame - 1, aoi),
            self.points(split, frame, aoi),
        )

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "HMNetData":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class CachedHMNetData:
    """TRAIN-only HM-Net reader backed by the shared r1 frame cache."""

    def __init__(self, frame_cache_path: Path, truth_cache_path: Path) -> None:
        self.frames = FixedGridFrameCache(frame_cache_path)
        self.truth = IndexedNpzTruth(
            truth_cache_path, ("track_id", "x", "y", "class_id"), split_code=0
        )

    def points(self, split: str, frame: int, aoi: str) -> list[ClassifiedPoint]:
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN truth only")
        rows = self.truth.rows(frame, aoi)
        return [
            ClassifiedPoint(int(track_id), float(x), float(y), int(class_id))
            for track_id, x, y, class_id in zip(
                rows["track_id"], rows["x"], rows["y"], rows["class_id"]
            )
        ]

    def read_pair(self, split: str, frame: int, aoi: str):
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN frames only")
        return (
            self.frames.read_frame(frame - 1, aoi),
            self.frames.read_frame(frame, aoi),
            self.points(split, frame - 1, aoi),
            self.points(split, frame, aoi),
        )

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "CachedHMNetData":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def build_training_sample(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_points: list[ClassifiedPoint],
    current_points: list[ClassifiedPoint],
    rng: np.random.Generator,
    *,
    width: int = 960,
    height: int = 544,
    feedback_frame_dropout_probability: float = 0.0,
    feedback_center_dropout_probability: float = 0.0,
) -> HMNetSample:
    proxy = [
        # select_crop_origin only consumes x/y.
        type("Anchor", (), {"x": point.x, "y": point.y})()
        for point in current_points
    ]
    left, top = select_crop_origin(
        current_image.shape, proxy, rng, width=width, height=height
    )
    rows, columns = slice(top, top + height), slice(left, left + width)
    local_current = [
        ClassifiedPoint(
            point.track_id, point.x - left, point.y - top, point.class_id
        )
        for point in current_points
        if left <= point.x < left + width and top <= point.y < top + height
    ]
    shifted_previous = [
        ClassifiedPoint(
            point.track_id, point.x - left, point.y - top, point.class_id
        )
        for point in previous_points
    ]
    feedback_truth = [
        (point.x, point.y, point.class_id)
        for point in shifted_previous
        if 0 <= point.x < width and 0 <= point.y < height
        and rng.random() >= feedback_center_dropout_probability
    ]
    if rng.random() < feedback_frame_dropout_probability:
        feedback = torch.zeros(2, height, width)
    else:
        feedback = render_weighted_centers(
            height,
            width,
            2,
            augment_feedback_centers(feedback_truth, rng),
        )
    center, motion, precision, motion_mask, precision_mask = build_targets(
        local_current, shifted_previous, height, width
    )
    return HMNetSample(
        torch.from_numpy(_normalize(current_image[rows, columns]))[None],
        torch.from_numpy(_normalize(previous_image[rows, columns]))[None],
        feedback,
        center,
        motion,
        precision,
        motion_mask,
        precision_mask,
        (left, top),
    )
