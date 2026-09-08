"""Read causal CATLoss frame stacks and cached truth on a fixed AOI grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from common.data.frame_cache import FixedGridFrameCache, IndexedNpzTruth


class CATLossData:
    def __init__(
        self,
        manifest_path: Path,
        fixed_grids_path: Path,
        truth_cache_path: Path,
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixed = json.loads(fixed_grids_path.read_text(encoding="utf-8"))
        profile = fixed["papers"]["catloss"]
        self.grids = profile["grids"]
        self.level = profile["level"]
        self.dataset_root = Path(self.manifest["dataset_root"])
        self.lookup = {
            (frame["set"], frame["frame_number"]): frame
            for frame in self.manifest["frames"]
        }
        self.truth = np.load(truth_cache_path)

    def read_frame(self, split: str, frame_number: int, aoi: str) -> np.ndarray:
        grid = self.grids[aoi]
        relative = self.lookup[(split, frame_number)]["levels"][self.level]["path"]
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

    def read_causal_stack(
        self,
        split: str,
        frame_number: int,
        aoi: str,
        channels: int = 4,
    ) -> list[np.ndarray]:
        first = self.manifest["frame_sets"][split]["expected_start"]
        if frame_number - channels + 1 < first:
            raise ValueError("Causal stack reaches before the frame-set boundary")
        return [
            self.read_frame(split, frame, aoi)
            for frame in range(frame_number - channels + 1, frame_number + 1)
        ]

    def truth_xy(
        self,
        split: str,
        frame_number: int,
        aoi: str,
    ) -> list[tuple[float, float]]:
        split_code = 0 if split == "train" else 1
        mask = (
            (self.truth["split"] == split_code)
            & (self.truth["frame"] == frame_number)
            & (self.truth["aoi"] == int(aoi))
        )
        return list(zip(self.truth["x"][mask].tolist(), self.truth["y"][mask].tolist()))

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "CATLossData":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class CachedCATLossData:
    """TRAIN-only CATLoss reader using the shared comparison frame cache."""

    def __init__(self, frame_cache_path: Path, truth_cache_path: Path) -> None:
        self.frames = FixedGridFrameCache(frame_cache_path)
        self.truth = IndexedNpzTruth(truth_cache_path, ("x", "y"), split_code=0)

    def read_causal_stack(
        self, split: str, frame_number: int, aoi: str, channels: int = 4
    ) -> list[np.ndarray]:
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN frames only")
        return [
            self.frames.read_frame(frame, aoi)
            for frame in range(frame_number - channels + 1, frame_number + 1)
        ]

    def truth_xy(
        self, split: str, frame_number: int, aoi: str
    ) -> list[tuple[float, float]]:
        if split != "train":
            raise ValueError("The optimized cache contains TRAIN truth only")
        rows = self.truth.rows(frame_number, aoi)
        return list(zip(rows["x"].tolist(), rows["y"].tolist()))

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "CachedCATLossData":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
