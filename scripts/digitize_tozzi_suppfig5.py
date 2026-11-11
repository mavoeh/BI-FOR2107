"""Read Tozzi et al. (2024) Supplementary Figure 5 into per-feature biotype means.

The figure shows, for each of the six biotypes, the mean regional circuit
score of all 41 features as a bar (in SD from the healthy reference). The
supplement gives no table of these values, so the bars are measured in the
embedded raster image (1342 x 1737 px, 220 ppi): each panel's y-axis runs
from +5.0 (top of the axis line) to -5.0 (bottom axis line), about 21.8 px
per SD, and each bar's length is the summed colour fraction of its rows
(sub-pixel), with the black SE whisker in the bar centre excluded. Bars that
show no colour are below about 0.02 SD and recorded as 0. The circuit means
of the result agree with Fig. 3 of the main text to 0.024 SD on average.

Usage:
    python scripts/digitize_tozzi_suppfig5.py literature/2024_tozzi_suppl.pdf \\
        configs/tozzi_biotype_feature_means.tsv [--overlay check.png]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BIOTYPES = ["DC+SC+AC+", "AC-", "NSA+PA+", "CA+", "NTCC-CA-", "DxSxAxNxPxCx"]
# x-axis order of the figure; codes as in Supplementary Table 18
FEATURES = [
    "D2D1", "D1D3", "D1D4", "D2D4", "D3D4", "S1S3", "S2S4", "S1S2",
    "A2A1", "A3A1", "A4A2", "A5A3", "A4A6", "A5A7",
    "NS1", "NS2", "NS3", "NS4", "NS5", "NS2NS1", "NS3NS1", "NS4NS1", "NS5NS1",
    "NT1", "NT2", "NT3", "NT2NT1", "NT3NT1",
    "NTN1", "NTN2", "NTN3", "NTN2NTN1", "NTN3NTN1",
    "P1", "P2", "P3", "C1", "C2", "C3", "C1C2", "C3C2",
]  # fmt: skip
# bar colour per feature: default, salience, attention, negative, positive, cognitive
COLOUR_GROUP = [0] * 5 + [1] * 3 + [2] * 6 + [3] * 19 + [4] * 3 + [5] * 5
PAGE = 11
# y-axis lines of the six panels (first and last pixel row) and bar centres
PANELS = [(217, 435), (468, 686), (708, 926), (948, 1166), (1191, 1409), (1427, 1645)]
FIRST_BAR, BAR_SPACING = 266.3, 19.60
SAMPLE_OFFSETS = (-7, -6, -5, -4, -3, 3, 4, 5, 6, 7)


def extract_image(pdf: Path, directory: Path) -> Path:
    subprocess.run(
        ["pdfimages", "-f", str(PAGE), "-l", str(PAGE), "-png", str(pdf), str(directory / "fig")],
        check=True,
    )
    image = directory / "fig-000.png"
    if Image.open(image).size != (1342, 1737):
        raise ValueError(f"Unexpected Supplementary Figure 5 image size in {pdf}")
    return image


def measure(image: Path) -> pd.DataFrame:
    pixels = np.asarray(Image.open(image).convert("RGB")).astype(float)
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    centres = FIRST_BAR + BAR_SPACING * np.arange(len(FEATURES))
    blocks = {
        (panel, bar): saturation[top : bottom + 1, [round(centre + d) for d in SAMPLE_OFFSETS]]
        for panel, (top, bottom) in enumerate(PANELS)
        for bar, centre in enumerate(centres)
    }
    # full-bar saturation per colour, from all bars of that colour
    reference = {}
    for group in set(COLOUR_GROUP):
        values = np.concatenate(
            [block.ravel() for (_, bar), block in blocks.items() if COLOUR_GROUP[bar] == group]
        )
        reference[group] = np.percentile(values[values > 40], 60)
    rows = []
    for panel, (top, bottom) in enumerate(PANELS):
        per_sd = (bottom - top) / 10
        rows_y = np.arange(top, bottom + 1)
        values = []
        for bar in range(len(FEATURES)):
            block = blocks[(panel, bar)]
            fraction = np.clip((block - 20) / (reference[COLOUR_GROUP[bar]] - 20), 0, 1)
            fraction[block < 30] = 0
            fraction = fraction.max(axis=1)
            mass = fraction.sum()
            if mass == 0:
                values.append(0.0)
                continue
            above = (fraction * rows_y).sum() / mass < (top + bottom) / 2
            values.append((1 if above else -1) * mass / per_sd)
        rows.append(values)
    return pd.DataFrame(rows, index=pd.Index(BIOTYPES, name="biotype"), columns=FEATURES)


def tail_counts(image: Path, thresholds: tuple[float, ...] = (3.0, 4.0)) -> pd.DataFrame:
    """Patient dots beyond +-t SD of the healthy reference, per biotype and feature.

    Dots are the grey points of individual patients, jittered across the bar
    width. They are sparse in the tails, so connected grey blobs there count
    patients; the black SE whisker in the bar centre is excluded. Values
    beyond the +-5 SD axis are not drawn, so the counts are lower bounds.
    """
    from scipy import ndimage

    pixels = np.asarray(Image.open(image).convert("RGB")).astype(float)
    # isolated tail dots are faint (grey 212-230); validated against an empty strip
    grey = (pixels.mean(axis=2) < 235) & (pixels.max(axis=2) - pixels.min(axis=2) < 40)
    centres = FIRST_BAR + BAR_SPACING * np.arange(len(FEATURES))
    rows = []
    for biotype, (top, bottom) in zip(BIOTYPES, PANELS, strict=True):
        zero, per_sd = (top + bottom) / 2, (bottom - top) / 10
        for feature, centre in zip(FEATURES, centres, strict=True):
            left, right = round(centre - 9), round(centre + 9)
            mask = grey[top + 2 : bottom - 2, left : right + 1].copy()
            mask[:, round(centre) - left - 1 : round(centre) - left + 2] = False
            labels, count = ndimage.label(mask)
            if not count:
                blobs = np.empty(0)
            else:
                blobs = np.array(ndimage.center_of_mass(mask, labels, range(1, count + 1)))[:, 0]
            sd = np.abs(top + 2 + blobs - zero) / per_sd
            rows.append(
                {
                    "biotype": biotype,
                    "feature": feature,
                    **{f"beyond_{t:g}sd": int(np.sum(sd > t)) for t in thresholds},
                }
            )
    return pd.DataFrame(rows)


def overlay(image: Path, means: pd.DataFrame, path: Path) -> None:
    """Draw the measured bar ends on the figure for visual checking."""
    picture = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    centres = FIRST_BAR + BAR_SPACING * np.arange(len(FEATURES))
    for (top, bottom), (_, row) in zip(PANELS, means.iterrows(), strict=True):
        zero, per_sd = (top + bottom) / 2, (bottom - top) / 10
        for centre, value in zip(centres, row.to_numpy(), strict=True):
            y = zero - value * per_sd
            draw.line([(centre - 9, y), (centre + 9, y)], fill=(255, 0, 255), width=1)
    picture.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--tails", type=Path, help="also write patient dots beyond 3 and 4 SD")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        image = extract_image(args.pdf, Path(directory))
        means = measure(image)
        if args.overlay:
            overlay(image, means, args.overlay)
        if args.tails:
            tail_counts(image).to_csv(args.tails, sep="\t", index=False)
    means.round(2).to_csv(args.output, sep="\t")
    print(f"Wrote {args.output}: {means.shape[0]} biotypes x {means.shape[1]} features")


if __name__ == "__main__":
    main()
