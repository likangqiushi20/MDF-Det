"""Topology-preserving post-processing for the six leakage-free V4 priors.

The operation is deliberately image-only: no target-AOI ground truth is read.
It connects short directional gaps, removes compact isolated responses and keeps
the result soft so that it can still be used as a probabilistic prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AOIS = ("01", "02", "03", "34", "40", "41")


def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.uint8)
    center = (length - 1) / 2.0
    radius = center
    angle = np.deg2rad(angle_deg)
    dx, dy = radius * np.cos(angle), radius * np.sin(angle)
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, 1, 1, cv2.LINE_8)
    return kernel


def directional_close(mask: np.ndarray, length: int) -> np.ndarray:
    closed = mask.copy()
    for angle in (0, 30, 60, 90, 120, 150):
        candidate = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, line_kernel(length, angle),
            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        closed = cv2.bitwise_or(closed, candidate)
    return closed


def clean_prior(
    prior: np.ndarray,
    sigma: float,
    weak_threshold: float,
    strong_threshold: float,
    close_length: int,
    compact_min_area: int,
    elongated_min_area: int,
    minimum_elongation: float,
    suppression_factor: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    probability = np.clip(prior.astype(np.float32), 0.0, 1.0)
    smooth = cv2.GaussianBlur(probability, (0, 0), sigma)
    weak = np.uint8(smooth >= weak_threshold)
    strong = np.uint8(smooth >= strong_threshold)

    # Bridge only short gaps that are supported by a directional strong ridge.
    closed_strong = directional_close(strong, close_length)
    bridge = cv2.bitwise_and(closed_strong, cv2.dilate(weak, np.ones((5, 5), np.uint8)))
    support = cv2.bitwise_or(weak, bridge)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    keep = np.zeros_like(support)
    kept_components = removed_components = 0
    kept_pixels = removed_pixels = 0
    component_rows = []
    for label in range(1, count):
        component = labels == label
        if not np.any(strong[component]):
            removed_components += 1
            removed_pixels += int(stats[label, cv2.CC_STAT_AREA])
            continue
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        elongation = max(width, height) / max(1, min(width, height))
        is_large_region = area >= compact_min_area
        is_road_like = area >= elongated_min_area and elongation >= minimum_elongation
        accepted = is_large_region or is_road_like
        component_rows.append({
            "area": area, "width": width, "height": height,
            "elongation": float(elongation), "kept": bool(accepted)})
        if accepted:
            keep[component] = 1
            kept_components += 1
            kept_pixels += area
        else:
            removed_components += 1
            removed_pixels += area

    # Preserve probability ordering in accepted regions. New bridge pixels get
    # at least weak-threshold confidence; rejected regions are attenuated, not
    # hard-zeroed, which protects recall when used with a confidence bypass.
    accepted_probability = np.maximum(probability, smooth)
    accepted_probability[bridge.astype(bool)] = np.maximum(
        accepted_probability[bridge.astype(bool)], weak_threshold)
    cleaned = probability * suppression_factor
    cleaned[keep.astype(bool)] = accepted_probability[keep.astype(bool)]
    cleaned = cv2.GaussianBlur(cleaned, (0, 0), 0.7)
    cleaned = np.clip(cleaned, 0.0, 1.0).astype(np.float32)

    def topology_stats(array: np.ndarray, threshold: float = .15) -> dict:
        binary = np.uint8(array >= threshold)
        n, _, component_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        areas = component_stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.empty(0, np.int32)
        active = int(areas.sum())
        return {
            "components_ge_0p15": int(len(areas)),
            "small_components_lt_500px_ge_0p15": int(np.sum(areas < 500)),
            "largest_component_fraction_ge_0p15":
                float(areas.max() / active) if active else 0.0,
        }

    report = {
        "shape": list(probability.shape),
        "original_mean": float(probability.mean()),
        "processed_mean": float(cleaned.mean()),
        "original_coverage_ge_0p15": float((probability >= .15).mean()),
        "processed_coverage_ge_0p15": float((cleaned >= .15).mean()),
        "original_coverage_ge_0p30": float((probability >= .30).mean()),
        "processed_coverage_ge_0p30": float((cleaned >= .30).mean()),
        "weak_support_coverage": float(weak.mean()),
        "topology_mask_coverage": float(keep.mean()),
        "bridge_pixels": int(np.count_nonzero(bridge & ~weak)),
        "kept_components": kept_components,
        "removed_components": removed_components,
        "kept_component_pixels": kept_pixels,
        "removed_component_pixels": removed_pixels,
        "original_topology": topology_stats(probability),
        "processed_topology": topology_stats(cleaned),
        "components": component_rows,
    }
    return cleaned, keep, report


def heat_overlay(context: np.ndarray, prior: np.ndarray, title: str) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(np.clip(prior * 255, 0, 255)), cv2.COLORMAP_JET)
    canvas = cv2.addWeighted(cv2.cvtColor(context, cv2.COLOR_GRAY2BGR), .55, heat, .45, 0)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(canvas, title, (14, 38), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data" / "semantic_prior_v4_six_aoi")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "semantic_prior_v4_topology")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--weak-threshold", type=float, default=.08)
    parser.add_argument("--strong-threshold", type=float, default=.35)
    parser.add_argument("--close-length", type=int, default=15)
    parser.add_argument("--compact-min-area", type=int, default=2500)
    parser.add_argument("--elongated-min-area", type=int, default=500)
    parser.add_argument("--minimum-elongation", type=float, default=2.5)
    parser.add_argument("--suppression-factor", type=float, default=.20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    processed_panels, comparison_panels, reports = [], [], {}
    for aoi in AOIS:
        source_folder = args.source / f"aoi{aoi}_prior"
        output_folder = args.output / f"aoi{aoi}_prior"
        output_folder.mkdir(parents=True, exist_ok=True)
        context = cv2.imread(str(source_folder / "context.png"), cv2.IMREAD_GRAYSCALE)
        prior = np.load(source_folder / "combined_prior.npy").astype(np.float32)
        if context is None:
            raise FileNotFoundError(source_folder / "context.png")

        cleaned, topology, report = clean_prior(
            prior, args.sigma, args.weak_threshold, args.strong_threshold,
            args.close_length, args.compact_min_area, args.elongated_min_area,
            args.minimum_elongation, args.suppression_factor)
        reports[aoi] = report
        np.save(output_folder / "combined_prior_topology.npy", cleaned)
        np.save(output_folder / "topology_mask.npy", topology)
        cv2.imwrite(str(output_folder / "combined_prior_topology.png"),
                    np.uint8(cleaned * 255))
        cv2.imwrite(str(output_folder / "topology_mask.png"), topology * 255)
        cv2.imwrite(str(output_folder / "context.png"), context)

        before = heat_overlay(context, prior, f"AOI{aoi} original")
        after = heat_overlay(
            context, cleaned,
            f"AOI{aoi} topology  mean={cleaned.mean():.3f}  >=.15={(cleaned >= .15).mean():.1%}")
        cv2.imwrite(str(output_folder / "combined_prior_topology_overlay.png"), after)
        cv2.imwrite(str(output_folder / "before_after.png"), np.hstack((before, after)))
        processed_panels.append(cv2.resize(after, (564, 564), interpolation=cv2.INTER_AREA))
        comparison_panels.append(cv2.resize(np.hstack((before, after)),
                                            (1128, 564), interpolation=cv2.INTER_AREA))

    montage = np.vstack((np.hstack(processed_panels[:3]), np.hstack(processed_panels[3:])))
    comparisons = np.vstack(comparison_panels)
    cv2.imwrite(str(args.output / "v4_topology_six_aoi_montage.png"), montage)
    cv2.imwrite(str(args.output / "v4_topology_before_after_montage.png"), comparisons)
    payload = {
        "method": "V4-T topology-preserving post-processing (GT-free)",
        "parameters": {
            "sigma": args.sigma, "weak_threshold": args.weak_threshold,
            "strong_threshold": args.strong_threshold, "close_length": args.close_length,
            "compact_min_area": args.compact_min_area,
            "elongated_min_area": args.elongated_min_area,
            "minimum_elongation": args.minimum_elongation,
            "suppression_factor": args.suppression_factor,
        },
        "aoi_reports": reports,
    }
    (args.output / "postprocess_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"parameters": payload["parameters"],
                      "aoi_summary": {k: {x: v[x] for x in (
                          "original_mean", "processed_mean",
                          "original_coverage_ge_0p15", "processed_coverage_ge_0p15",
                          "kept_components", "removed_components")}
                                      for k, v in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
