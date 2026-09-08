"""Read r0 ClusterNet frame stacks and fixed-grid cached truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from common.data.frame_cache import FixedGridFrameCache, IndexedNpzTruth


class ClusterNetData:
    def __init__(self, manifest_path: Path, grids_path: Path, truth_cache: Path):
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixed = json.loads(grids_path.read_text(encoding="utf-8"))
        profile = fixed["papers"]["clusternet"]
        self.grids = profile["grids"]
        self.level = profile["level"]
        self.root = Path(self.manifest["dataset_root"])
        self.lookup = {
            (item["set"], item["frame_number"]): item
            for item in self.manifest["frames"]
        }
        self.truth = np.load(truth_cache)

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

    def centered_stack(self, split: str, frame: int, aoi: str, channels: int = 5):
        if channels % 2 != 1:
            raise ValueError("ClusterNet centered temporal stacks must be odd")
        radius = channels // 2
        return [
            self.read_frame(split, value, aoi)
            for value in range(frame - radius, frame + radius + 1)
        ]

    def truth_xy(self, split: str, frame: int, aoi: str, moving_only: bool = True):
        split_code = 0 if split == "train" else 1
        mask = (
            (self.truth["split"] == split_code)
            & (self.truth["aoi"] == int(aoi))
            & (self.truth["frame"] == frame)
        )
        if moving_only:
            mask &= self.truth["moving"]
        return list(zip(self.truth["x"][mask].tolist(), self.truth["y"][mask].tolist()))

    def close(self):
        self.truth.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class CachedClusterNetData:
    """TRAIN-only ClusterNet reader backed by the shared comparison cache."""

    def __init__(self, frame_cache_path: Path, truth_cache_path: Path):
        self.frames = FixedGridFrameCache(frame_cache_path)
        self.truth = IndexedNpzTruth(
            truth_cache_path, ("x", "y", "moving"), split_code=0
        )

    def centered_stack(self, split: str, frame: int, aoi: str, channels: int = 5):
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN frames only")
        if channels % 2 != 1:
            raise ValueError("ClusterNet centered temporal stacks must be odd")
        radius = channels // 2
        return [
            self.frames.read_frame(value, aoi)
            for value in range(frame - radius, frame + radius + 1)
        ]

    def truth_xy(self, split: str, frame: int, aoi: str, moving_only: bool = True):
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN truth only")
        rows = self.truth.rows(frame, aoi)
        selected = rows["moving"] if moving_only else np.ones(len(rows["x"]), dtype=bool)
        return list(zip(rows["x"][selected].tolist(), rows["y"][selected].tolist()))

    def close(self):
        self.truth.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
