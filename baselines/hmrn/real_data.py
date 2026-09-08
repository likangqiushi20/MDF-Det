"""Read consecutive HMRN r1 frames and tracked truth on fixed AOI grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from hmrn.targets import TrackedPoint
from common.data.frame_cache import FixedGridFrameCache, IndexedNpzTruth


class HMRNData:
    def __init__(
        self, manifest_path: Path, fixed_grids_path: Path, truth_cache_path: Path
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixed = json.loads(fixed_grids_path.read_text(encoding="utf-8"))
        self.grids = fixed["papers"]["hmrn"]["grids"]
        self.dataset_root = Path(self.manifest["dataset_root"])
        self.lookup = {
            (frame["set"], frame["frame_number"]): frame
            for frame in self.manifest["frames"]
        }
        self.truth = np.load(truth_cache_path)

    def read_frame(self, split: str, frame_number: int, aoi: str) -> np.ndarray:
        grid = self.grids[aoi]
        relative = self.lookup[(split, frame_number)]["levels"]["r1"]["path"]
        with rasterio.open(self.dataset_root / relative) as source:
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

    def truth_points(
        self, split: str, frame_number: int, aoi: str
    ) -> list[TrackedPoint]:
        split_code = 0 if split == "train" else 1
        mask = (
            (self.truth["split"] == split_code)
            & (self.truth["frame"] == frame_number)
            & (self.truth["aoi"] == int(aoi))
        )
        return [
            TrackedPoint(int(track_id), float(x), float(y))
            for track_id, x, y in zip(
                self.truth["track_id"][mask],
                self.truth["x"][mask],
                self.truth["y"][mask],
            )
        ]

    def read_pair(
        self, split: str, current_frame: int, aoi: str
    ) -> tuple[np.ndarray, np.ndarray, list[TrackedPoint], list[TrackedPoint]]:
        first = self.manifest["frame_sets"][split]["expected_start"]
        if current_frame <= first:
            raise ValueError("A previous frame is required")
        return (
            self.read_frame(split, current_frame - 1, aoi),
            self.read_frame(split, current_frame, aoi),
            self.truth_points(split, current_frame - 1, aoi),
            self.truth_points(split, current_frame, aoi),
        )

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "HMRNData":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class CachedHMRNData:
    """TRAIN-only HMRN reader backed by the shared r1 frame cache."""

    def __init__(self, frame_cache_path: Path, truth_cache_path: Path) -> None:
        self.frames = FixedGridFrameCache(frame_cache_path)
        self.truth = IndexedNpzTruth(
            truth_cache_path, ("track_id", "x", "y"), split_code=0
        )

    def truth_points(
        self, split: str, frame_number: int, aoi: str
    ) -> list[TrackedPoint]:
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN frames only")
        rows = self.truth.rows(frame_number, aoi)
        return [
            TrackedPoint(int(track_id), float(x), float(y))
            for track_id, x, y in zip(rows["track_id"], rows["x"], rows["y"])
        ]

    def read_pair(
        self, split: str, current_frame: int, aoi: str
    ) -> tuple[np.ndarray, np.ndarray, list[TrackedPoint], list[TrackedPoint]]:
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN frames only")
        return (
            self.frames.read_frame(current_frame - 1, aoi),
            self.frames.read_frame(current_frame, aoi),
            self.truth_points(split, current_frame - 1, aoi),
            self.truth_points(split, current_frame, aoi),
        )

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "CachedHMRNData":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
