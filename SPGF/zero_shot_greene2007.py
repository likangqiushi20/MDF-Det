"""Zero-shot SPGF-V4 generalization test on Greene2007 r1 imagery.

The selected NITF r1 frames are warped to the first frame's geographic grid,
saved as PNG, and reduced to a registered temporal median.  Six representative
native-resolution crops are then evaluated by the WPAFB-trained SPGF-V4 model.
No Greene labels or fine-tuning are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from osgeo import gdal

try:
    from .infer import tiled_predict
    from .model import GroupNormalization
except ImportError:  # Allow: python SPGF/zero_shot_greene2007.py ...
    from infer import tiled_predict
    from model import GroupNormalization


DEFAULT_REGIONS = (
    ("A", "Residential arterial", 850, 1100, 1024, 1024),
    ("B", "Commercial / parking", 1800, 1350, 1024, 1024),
    ("C", "Highway interchange", 2800, 1400, 1024, 1024),
    ("D", "Park / residential", 750, 2700, 1024, 1024),
    ("E", "Highway / residential", 2150, 2700, 1024, 1024),
    ("F", "Wooded residential", 3550, 2300, 1024, 1024),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--resolution", default="r1")
    parser.add_argument("--tile", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def reference_grid(path: Path):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"GDAL cannot open {path}")
    transform = dataset.GetGeoTransform()
    width, height = dataset.RasterXSize, dataset.RasterYSize
    left = transform[0]
    top = transform[3]
    right = left + width * transform[1]
    bottom = top + height * transform[5]
    return dataset.GetProjection(), (left, bottom, right, top), width, height


def warp_to_reference(path: Path, projection: str, bounds, width: int, height: int):
    source = gdal.Open(str(path), gdal.GA_ReadOnly)
    warped = gdal.Warp(
        "", source, format="MEM", dstSRS=projection, outputBounds=bounds,
        width=width, height=height, resampleAlg=gdal.GRA_Bilinear,
        srcNodata=0, dstNodata=0,
    )
    if warped is None:
        raise RuntimeError(f"GDAL warp failed for {path}")
    return warped.ReadAsArray().astype(np.uint8)


def contrast(image: np.ndarray):
    values = image[image > 0]
    low, high = np.percentile(values, (0.5, 99.5)) if len(values) else (0, 255)
    return np.uint8(np.clip((image.astype(np.float32) - low) * 255.0 /
                            max(float(high - low), 1.0), 0, 255))


def save_overview(image: np.ndarray, regions, destination: Path):
    scale = min(1.0, 1800.0 / max(image.shape))
    canvas = cv2.cvtColor(cv2.resize(contrast(image), None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    colours = [(44, 160, 44), (31, 119, 180), (255, 127, 14),
               (148, 103, 189), (214, 39, 40), (23, 190, 207)]
    for (key, _name, x, y, width, height), colour in zip(regions, colours):
        p1 = (round(x * scale), round(y * scale))
        p2 = (round((x + width) * scale), round((y + height) * scale))
        cv2.rectangle(canvas, p1, p2, colour, 5, cv2.LINE_AA)
        cv2.putText(canvas, key, (p1[0] + 12, p1[1] + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(canvas, key, (p1[0] + 12, p1[1] + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    png_root = args.output / f"png_{args.resolution}_registered"
    crop_root = args.output / "regions"
    png_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    files = sorted(args.source.glob(f"*.ntf.{args.resolution}"))
    if len(files) < args.frame_count:
        raise RuntimeError(f"Only {len(files)} {args.resolution} files were found")
    files = files[:args.frame_count]
    projection, bounds, width, height = reference_grid(files[0])

    region_stacks = {region[0]: [] for region in DEFAULT_REGIONS}
    first_registered = None
    frame_rows = []
    for index, source_path in enumerate(files, 1):
        image = warp_to_reference(source_path, projection, bounds, width, height)
        if first_registered is None:
            first_registered = image.copy()
        output_path = png_root / f"frame{index:06d}_{source_path.stem}.png"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        for key, _name, x, y, crop_width, crop_height in DEFAULT_REGIONS:
            crop = image[y:y + crop_height, x:x + crop_width]
            if crop.shape != (crop_height, crop_width):
                raise ValueError(f"Region {key} is outside the r1 image")
            region_stacks[key].append(crop)
        frame_rows.append({"index": index, "source": str(source_path),
                           "png": str(output_path.resolve())})
        print(f"[{index}/{len(files)}] {source_path.name} -> {output_path.name}", flush=True)

    save_overview(first_registered, DEFAULT_REGIONS, args.output / "selected_regions_overview.png")
    model = tf.keras.models.load_model(
        args.model, custom_objects={"GroupNormalization": GroupNormalization},
        compile=False,
    )
    model.predict(np.zeros((1, args.tile, args.tile, 1), np.float32), verbose=0)

    results = []
    contexts, priors, overlays = [], [], []
    for key, name, x, y, crop_width, crop_height in DEFAULT_REGIONS:
        context = np.median(np.stack(region_stacks[key]), axis=0).astype(np.uint8)
        prediction = tiled_predict(model, context, args.tile, args.overlap)
        access = prediction[..., 0].astype(np.float32)
        motion = prediction[..., 1].astype(np.float32)
        combined = np.sqrt(np.clip(access * motion, 0.0, 1.0)).astype(np.float32)

        folder = crop_root / f"region_{key.lower()}"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "context.png"), context)
        cv2.imwrite(str(folder / "combined_prior.png"), np.uint8(combined * 255))
        np.savez_compressed(folder / "spgf_v4_prior.npz", access=access,
                            motion=motion, combined=combined)
        heat_rgb = np.uint8(plt.get_cmap("inferno")(combined)[..., :3] * 255)
        heat = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(cv2.cvtColor(contrast(context), cv2.COLOR_GRAY2BGR),
                                  0.58, heat, 0.42, 0)
        cv2.imwrite(str(folder / "prior_overlay.png"), overlay)
        contexts.append(contrast(context))
        priors.append(combined)
        overlays.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        results.append({
            "region": key, "description": name, "box_xywh": [x, y, crop_width, crop_height],
            "mean_probability": float(combined.mean()),
            "coverage_ge_0p10": float((combined >= 0.10).mean()),
            "coverage_ge_0p20": float((combined >= 0.20).mean()),
            "coverage_ge_0p30": float((combined >= 0.30).mean()),
        })

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    figure, axes = plt.subplots(3, len(DEFAULT_REGIONS), figsize=(13.2, 6.6), squeeze=False)
    heat_artist = None
    for column, ((key, name, *_), context, prior, overlay) in enumerate(
            zip(DEFAULT_REGIONS, contexts, priors, overlays)):
        axes[0, column].imshow(context, cmap="gray", vmin=0, vmax=255)
        heat_artist = axes[1, column].imshow(prior, cmap="inferno", vmin=0, vmax=1)
        axes[2, column].imshow(overlay)
        axes[0, column].set_title(f"({key}) {name}", fontsize=10)
    for axis in axes.flat:
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_visible(False)
    axes[0, 0].set_ylabel("Greene2007 r1")
    axes[1, 0].set_ylabel("SPGF-V4 prior")
    axes[2, 0].set_ylabel("Overlay")
    figure.subplots_adjust(left=.075, right=.995, top=.89, bottom=.025,
                           wspace=.018, hspace=.055)
    color_axis = figure.add_axes((.27, .935, .50, .018))
    colorbar = figure.colorbar(heat_artist, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Vehicle-presence probability", labelpad=2)
    color_axis.xaxis.set_label_position("top"); color_axis.xaxis.set_ticks_position("top")
    figure.savefig(args.output / "greene2007_spgf_v4_zero_shot_3x6.png",
                   dpi=args.dpi, facecolor="white", bbox_inches="tight")
    figure.savefig(args.output / "greene2007_spgf_v4_zero_shot_3x6.pdf",
                   facecolor="white", bbox_inches="tight")
    plt.close(figure)

    report = {
        "dataset": "Greene2007", "resolution": args.resolution,
        "protocol": "zero-shot; WPAFB-trained SPGF-V4; no Greene labels or fine-tuning",
        "reference_file": str(files[0]), "reference_shape": [height, width],
        "registered_context_frames": len(files), "frames": frame_rows,
        "model": str(args.model.resolve()), "regions": results,
    }
    (args.output / "greene2007_spgf_v4_zero_shot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"regions": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    gdal.UseExceptions()
    main()
