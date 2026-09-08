"""Create a paper figure comparing SPGF outputs on WPAFB and Greene2007."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


AOIS = ("01", "02", "03", "34", "40", "41")
GREENE = (
    ("A", "Residential arterial"),
    ("B", "Commercial / parking"),
    ("C", "Highway interchange"),
    ("D", "Park / residential"),
    ("E", "Highway / residential"),
    ("F", "Wooded residential"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wpafb-root", type=Path, required=True)
    parser.add_argument("--greene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def read_gray(path: Path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def display_gray(image: np.ndarray):
    values = image[image > 0]
    low, high = np.percentile(values, (1.0, 99.5)) if len(values) else (0, 255)
    return np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0)


def overlay(image: np.ndarray, prior: np.ndarray):
    gray = display_gray(image)
    heat = plt.get_cmap("inferno")(np.clip(prior, 0, 1))[..., :3]
    return np.clip(0.58 * gray[..., None] + 0.42 * heat, 0, 1)


def configure_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
    })

    wpafb = []
    for aoi in AOIS:
        folder = args.wpafb_root / f"aoi{aoi}_prior"
        context = read_gray(folder / "context.png")
        prior = np.asarray(np.load(folder / "combined_prior.npy"), np.float32)
        if context.shape != prior.shape:
            raise ValueError(f"AOI{aoi}: context {context.shape} != prior {prior.shape}")
        wpafb.append((context, prior))

    greene = []
    for key, _description in GREENE:
        folder = args.greene_root / "regions" / f"region_{key.lower()}"
        context = read_gray(folder / "context.png")
        with np.load(folder / "spgf_v4_prior.npz") as data:
            prior = np.asarray(data["combined"], np.float32)
        if context.shape != prior.shape:
            raise ValueError(f"Greene {key}: context {context.shape} != prior {prior.shape}")
        greene.append((context, prior))

    figure = plt.figure(figsize=(13.4, 13.0), facecolor="white")
    grid = figure.add_gridspec(
        8, 6,
        height_ratios=(1, 1, 1, .20, 1, 1, 1, .20),
        left=.075, right=.995, top=.895, bottom=.025,
        wspace=.025, hspace=.055,
    )

    wpafb_axes = [[figure.add_subplot(grid[row, column]) for column in range(6)]
                  for row in range(0, 3)]
    greene_axes = [[figure.add_subplot(grid[row, column]) for column in range(6)]
                   for row in range(4, 7)]

    heat_artist = None
    for column, (aoi, (context, prior)) in enumerate(zip(AOIS, wpafb)):
        wpafb_axes[0][column].imshow(display_gray(context), cmap="gray", vmin=0, vmax=1)
        heat_artist = wpafb_axes[1][column].imshow(
            prior, cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        wpafb_axes[2][column].imshow(overlay(context, prior))
        wpafb_axes[0][column].set_title(f"AOI {aoi}", fontsize=11, pad=4)

    for column, (((key, description), (context, prior))) in enumerate(zip(GREENE, greene)):
        greene_axes[0][column].imshow(display_gray(context), cmap="gray", vmin=0, vmax=1)
        greene_axes[1][column].imshow(
            prior, cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        greene_axes[2][column].imshow(overlay(context, prior))
        greene_axes[0][column].set_title(
            f"({key}) {description}", fontsize=9.5, pad=4)

    for axes in wpafb_axes + greene_axes:
        for axis in axes:
            configure_axis(axis)
    for axes in (wpafb_axes, greene_axes):
        axes[0][0].set_ylabel("Input image", fontsize=11, labelpad=8)
        axes[1][0].set_ylabel("SPGF prior", fontsize=11, labelpad=8)
        axes[2][0].set_ylabel("Overlay", fontsize=11, labelpad=8)

    wpafb_label = figure.add_subplot(grid[3, :])
    greene_label = figure.add_subplot(grid[7, :])
    for axis, text in ((wpafb_label, "(a) WPAFB 2009"),
                       (greene_label, "(b) Greene 2007 (zero-shot)")):
        axis.axis("off")
        axis.text(.5, .5, text, ha="center", va="center",
                  fontsize=13, fontweight="bold")

    color_axis = figure.add_axes((.29, .943, .43, .012))
    colorbar = figure.colorbar(heat_artist, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Vehicle-presence probability", labelpad=3, fontsize=11)
    colorbar.set_ticks((0.0, 0.25, 0.50, 0.75, 1.0))
    color_axis.xaxis.set_label_position("top")
    color_axis.xaxis.set_ticks_position("top")

    stem = args.output / "spgf_wpafb_greene_zero_shot_comparison"
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi,
                   bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.resolve())


if __name__ == "__main__":
    main()
