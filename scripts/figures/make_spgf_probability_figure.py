"""Create a publication-ready 2x3 visualization of SPGF-V4 outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=612)
    parser.add_argument("--aois", nargs="+", default=("01", "02", "03", "34", "40", "41"))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
    })
    columns = len(args.aois)
    aspect_ratios = []
    for aoi in args.aois:
        image_path = args.frame_root / f"aoi{aoi}" / f"frame{args.frame:06d}.png"
        with Image.open(image_path) as source:
            width, height = source.size
        aspect_ratios.append(width / height)
    figure, axes = plt.subplots(
        2, columns, figsize=(8.7, 3.25), squeeze=False,
        gridspec_kw={"width_ratios": aspect_ratios},
    )
    heatmap_artist = None
    panel_labels = tuple(f"({chr(ord('a') + index)})" for index in range(columns))

    for column, aoi in enumerate(args.aois):
        image_path = args.frame_root / f"aoi{aoi}" / f"frame{args.frame:06d}.png"
        prior_path = args.prior_root / f"aoi{aoi}" / "prior" / "combined_prior_topology.npy"
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        prior = np.asarray(np.load(prior_path), dtype=np.float32)
        if image.shape != prior.shape:
            raise ValueError(f"AOI{aoi}: image {image.shape} != prior {prior.shape}")

        low, high = np.percentile(image, (1.0, 99.5))
        axes[0, column].imshow(image, cmap="gray", vmin=low, vmax=high,
                               interpolation="nearest")
        heatmap_artist = axes[1, column].imshow(
            prior, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="bilinear"
        )
        axes[0, column].set_title(f"AOI {aoi}", pad=4)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    for column, label in enumerate(panel_labels):
        axes[1, column].text(
            0.5, -0.105, label, transform=axes[1, column].transAxes,
            ha="center", va="top", color="black", fontsize=9,
        )

    axes[0, 0].set_ylabel("Input image", labelpad=7)
    axes[1, 0].set_ylabel("SPGF prior", labelpad=7)
    figure.subplots_adjust(left=0.068, right=0.995, top=0.805, bottom=0.105,
                           wspace=0.025, hspace=0.045)
    colorbar_axis = figure.add_axes((0.23, 0.925, 0.58, 0.022))
    colorbar = figure.colorbar(heatmap_artist, cax=colorbar_axis,
                               orientation="horizontal")
    colorbar.set_label("Vehicle-presence probability", labelpad=2)
    colorbar.set_ticks((0.0, 0.25, 0.5, 0.75, 1.0))
    colorbar_axis.xaxis.set_label_position("top")
    colorbar_axis.xaxis.set_ticks_position("top")

    stem = args.output / f"spgf_v4_probability_maps_2x{columns}"
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight",
                   facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight",
                   facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)
    print(stem)


if __name__ == "__main__":
    main()
