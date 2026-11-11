"""Read Tozzi et al. (2024) clustering-evaluation curves and silhouette plot.

From the embedded images (220 ppi) of the supplement:
- Supplementary Figure 2a/b: mean silhouette and its multinormal-null
  p-value for k = 2-15;
- Supplementary Figure 4: leave-one-out and leave-20%-out adjusted Rand
  index for k = 2-15;
- Supplementary Figure 2c: the silhouette width of each of the 801 patients
  in the six-cluster solution.

Curve points are the filled dots (thin lines removed by a morphological
opening), placed on each panel's y-axis from its tick marks; the k = 6 values
agree with the main text (silhouette .065, p .016, ARI .80 and .35). In the
silhouette plot every patient is one grey bar (about 1.1 pixel rows), sorted
within each cluster; clusters are split where the widths jump back up, each
is resampled to its known size, and the pixel scale is set by the mean
silhouette of panel a. The per-cluster means agree with the printed ones.

Usage:
    python scripts/digitize_tozzi_clustering_figures.py literature/2024_tozzi_suppl.pdf configs
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

KS = list(range(2, 16))
# silhouette plot, top to bottom (Supplementary Figure 2c)
SILHOUETTE_CLUSTERS = [
    ("NSA+PA+", 154),
    ("DC+SC+AC+", 169),
    ("DxSxAxNxPxCx", 44),
    ("AC-", 161),
    ("CA+", 258),
    ("NTCC-CA-", 15),
]


def extract(pdf: Path, page: int, directory: Path) -> np.ndarray:
    subprocess.run(
        [
            "pdfimages",
            "-f",
            str(page),
            "-l",
            str(page),
            "-png",
            str(pdf),
            str(directory / f"p{page}"),
        ],
        check=True,
    )
    return np.asarray(Image.open(directory / f"p{page}-000.png").convert("RGB")).astype(float)


def curve(
    pixels: np.ndarray, x_range: tuple[int, int], y_range: tuple[int, int], ticks: list[float]
) -> np.ndarray:
    """Values of the 14 dots (k = 2-15) of one panel.

    `ticks` are the y-axis tick values from top to bottom; the tick marks are
    the short dark runs just left of the panel's left axis.
    """
    dark = (pixels.mean(axis=2) < 120) & (np.ptp(pixels, axis=2) < 60)
    region = dark[y_range[0] : y_range[1], x_range[0] : x_range[1]]
    dots = ndimage.binary_opening(region, structure=np.ones((7, 7)))
    labels, count = ndimage.label(dots)
    centres = np.array(ndimage.center_of_mass(dots, labels, range(1, count + 1)))
    sizes = ndimage.sum(dots, labels, range(1, count + 1))
    centres = centres[(sizes > 40) & (sizes < 400)]
    centres = centres[np.argsort(centres[:, 1])] + [y_range[0], x_range[0]]
    if len(centres) != len(KS):
        raise ValueError(f"Expected {len(KS)} curve points, found {len(centres)}")
    # left axis: first column with a long dark run; ticks just left of it
    axis = next(
        x for x in range(x_range[0], x_range[1]) if dark[y_range[0] : y_range[1], x].sum() > 150
    )
    column = dark[y_range[0] : y_range[1], axis - 4 : axis - 1].any(axis=1)
    rows = np.flatnonzero(column) + y_range[0]
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 2) + 1)
    tick_rows = [g.mean() for g in groups if len(g)]
    if len(tick_rows) != len(ticks):
        raise ValueError(f"Expected {len(ticks)} ticks, found {len(tick_rows)}")
    # linear axis: fit through the ticks, so points beyond them are not clamped
    return np.polyval(np.polyfit(tick_rows, ticks, 1), centres[:, 0])


def silhouette_widths(pixels: np.ndarray, mean_silhouette: float) -> pd.DataFrame:
    bars = (pixels.mean(axis=2) > 150) & (pixels.mean(axis=2) < 230)
    bars[:, :860] = False
    zero = int(np.argmax(bars.sum(axis=0)))
    active = np.flatnonzero(bars.any(axis=1))
    signed = []
    for row in active:
        xs = np.flatnonzero(bars[row]) - zero
        signed.append(xs.max() if xs.max() > -xs.min() else xs.min())
    signed = np.array(signed, dtype=float)
    bounds = np.sort(np.argsort(-np.diff(signed))[: len(SILHOUETTE_CLUSTERS) - 1]) + 1
    rows = []
    segments = np.split(signed, bounds)
    resampled = [
        np.interp((np.arange(n) + 0.5) / n * len(segment), np.arange(len(segment)) + 0.5, segment)
        for segment, (_, n) in zip(segments, SILHOUETTE_CLUSTERS, strict=True)
    ]
    scale = np.concatenate(resampled).mean() / mean_silhouette
    for (name, n), values in zip(SILHOUETTE_CLUSTERS, resampled, strict=True):
        for rank, value in enumerate(values / scale):
            rows.append({"biotype": name, "size": n, "rank": rank, "silhouette": round(value, 4)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        fig2 = extract(args.pdf, 8, Path(directory))
        fig4 = extract(args.pdf, 10, Path(directory))
    curves = pd.DataFrame(
        {
            "k": KS,
            "silhouette": curve(fig2, (100, 860), (60, 490), [0.12, 0.10, 0.08, 0.06]),
            "p_multinormal": curve(fig2, (100, 860), (650, 1070), [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
            # every other tick is labelled (0.86, 0.82, 0.78; 0.38, 0.34, 0.30)
            "loo_ari": curve(fig4, (60, 700), (90, 430), [0.88, 0.86, 0.84, 0.82, 0.80, 0.78]),
            "subsample_ari": curve(
                fig4, (780, 1420), (90, 430), [0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
            ),
        }
    )
    curves.round(4).to_csv(
        args.output_directory / "tozzi_published_curves.tsv", sep="\t", index=False
    )
    widths = silhouette_widths(fig2, float(curves.set_index("k").loc[6, "silhouette"]))
    widths.to_csv(args.output_directory / "tozzi_silhouette_widths.tsv", sep="\t", index=False)
    print(curves.round(4).to_string(index=False))
    print(
        widths.groupby("biotype", sort=False).silhouette.agg(
            ["size", "mean", lambda s: (s < 0).mean()]
        )
    )


if __name__ == "__main__":
    main()
