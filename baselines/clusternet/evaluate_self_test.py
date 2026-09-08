from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .model import ClusterNet, FoveaNet
from .utils import add_external_repo, decode_local_maxima, origins_cover_bounds


def infer_probability(images, cluster, fovea, device, cluster_quantile: float, batch_size: int):
    from clusternet.postprocess import extract_roobis
    values = np.stack(images).astype(np.float32) / 255.0
    with torch.inference_mode():
        scoremap = cluster(torch.from_numpy(values[None]).to(device))[0, 0].cpu().numpy()
    roobis = extract_roobis(scoremap, float(np.quantile(scoremap, cluster_quantile)))
    height, width = images[-1].shape
    origins = sorted(set(
        origin for roobi in roobis
        for origin in origins_cover_bounds(roobi.input_bounds, width, height)
    ))
    sum_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    window_1d = np.maximum(np.hanning(128).astype(np.float32), 0.05)
    window = window_1d[:, None] * window_1d[None, :]
    for start in range(0, len(origins), batch_size):
        current = origins[start:start + batch_size]
        patches = np.stack([
            np.stack([image[y:y + 128, x:x + 128] for image in images])
            for x, y in current
        ]).astype(np.float32) / 255.0
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probability = torch.sigmoid(fovea(torch.from_numpy(patches).to(device))).cpu().numpy()[:, 0]
        for small, (x, y) in zip(probability, current):
            expanded = cv2.resize(small.astype(np.float32), (128, 128), interpolation=cv2.INTER_LINEAR)
            sum_map[y:y + 128, x:x + 128] += expanded * window
            weight_map[y:y + 128, x:x + 128] += window
    output = np.zeros_like(sum_map)
    np.divide(sum_map, weight_map, out=output, where=weight_map > 0)
    return output, {"roobis": len(roobis), "patches": len(origins), "covered_fraction": float(np.count_nonzero(weight_map) / weight_map.size)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ClusterNet plus FoveaNet-v2 with absolute threshold sweep.")
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "self_test"], default="self_test")
    parser.add_argument("--frame-cache", type=Path)
    parser.add_argument("--aoi", default="01")
    parser.add_argument("--first-frame", type=int, default=612)
    parser.add_argument("--last-frame", type=int, default=631)
    parser.add_argument("--cluster-quantile", type=float, default=0.90)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--match-radius", type=float, default=9.815537)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    add_external_repo(args.external_repo)
    from clusternet.real_data import CachedClusterNetData, ClusterNetData
    from common.metrics.point_matching import match_points
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cluster = ClusterNet(5).to(device).eval()
    cluster.load_state_dict(checkpoint["cluster"])
    model_config = checkpoint.get("model_config", {"temporal_channels": 5, "base_channels": 16})
    fovea = FoveaNet(**model_config).to(device).eval()
    fovea.load_state_dict(checkpoint.get("fovea", checkpoint.get("model")))
    totals = {threshold: {"tp": 0, "fp": 0, "fn": 0, "detections": 0} for threshold in args.thresholds}
    diagnostics = {"roobis": 0, "patches": 0, "covered_fraction": []}
    started = time.perf_counter()
    if args.split == "train":
        if args.frame_cache is None:
            raise ValueError("--frame-cache is required for TRAIN validation")
        data_context = CachedClusterNetData(args.frame_cache, args.truth_cache)
        lower, upper = 100, 611
    else:
        data_context = ClusterNetData(args.manifest, args.fixed_grids, args.truth_cache)
        lower, upper = 612, 1124
    with data_context as data:
        def read(frame):
            value = int(np.clip(frame, lower, upper))
            if args.split == "train":
                return data.frames.read_frame(value, args.aoi)
            return data.read_frame("self_test", value, args.aoi)
        images = deque([read(value) for value in range(args.first_frame - 2, args.first_frame + 3)], maxlen=5)
        for frame in range(args.first_frame, args.last_frame + 1):
            if frame != args.first_frame:
                images.append(read(frame + 2))
            probability, diag = infer_probability(list(images), cluster, fovea, device, args.cluster_quantile, args.batch_size)
            truth = data.truth_xy(args.split, frame, args.aoi, moving_only=True)
            for threshold in args.thresholds:
                detections = [(x, y) for x, y, _ in decode_local_maxima(probability, threshold, args.nms_radius)]
                matched = match_points(detections, truth, args.match_radius)
                row = totals[threshold]
                row["tp"] += matched.true_positives
                row["fp"] += matched.false_positives
                row["fn"] += matched.false_negatives
                row["detections"] += len(detections)
            diagnostics["roobis"] += diag["roobis"]
            diagnostics["patches"] += diag["patches"]
            diagnostics["covered_fraction"].append(diag["covered_fraction"])
            print(json.dumps({"frame": frame, **diag}), flush=True)
    sweep = []
    for threshold, counts in totals.items():
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.append({"threshold": threshold, **counts, "precision": precision, "recall": recall, "f1": f1})
    result = {
        "schema_version": 1, "status": "ok", "split": args.split, "aoi": args.aoi,
        "frames": [args.first_frame, args.last_frame], "cluster_quantile": args.cluster_quantile,
        "nms_radius": args.nms_radius, "match_radius": args.match_radius,
        "mean_covered_fraction": float(np.mean(diagnostics["covered_fraction"])),
        "roobis": diagnostics["roobis"], "patches": diagnostics["patches"],
        "sweep": sweep, "selected": max(sweep, key=lambda row: row["f1"]),
        "elapsed_seconds": time.perf_counter() - started,
        "threshold_note": (
            "single externally frozen threshold evaluated"
            if len(args.thresholds) == 1
            else "best threshold selected on the evaluated frames"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
