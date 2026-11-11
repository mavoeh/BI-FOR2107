"""Read Tozzi et al. (2024) Supplementary Figure 14a into a correlation table.

Panel a is an unthresholded heatmap of the Spearman correlations between his
14 task-free and 27 task features across all 801 patients, coloured on a
blue-white-red scale from -0.5 to +0.5. The supplement gives no table, so each
cell's median colour is matched to the nearest colour of the colour bar
(130 px for 1.0, i.e. about 0.01 per pixel; precision is limited by the
colour steps to roughly +-0.02).

Usage:
    python scripts/digitize_tozzi_suppfig14.py literature/2024_tozzi_suppl.pdf \\
        configs/tozzi_taskfree_task_correlations.tsv [--check check.png]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PAGE = 22
TASK_FREE = [
    "D2D1", "D1D3", "D1D4", "D2D4", "D3D4", "S1S3", "S2S4", "S1S2",
    "A2A1", "A3A1", "A4A2", "A5A3", "A4A6", "A5A7",
]  # fmt: skip
TASK = [
    "C3C2", "C1C2", "C3", "C2", "C1", "P3", "P2", "P1", "NTN3NTN1", "NTN2NTN1",
    "NTN3", "NTN2", "NTN1", "NT3NT1", "NT2NT1", "NT3", "NT2", "NT1",
    "NS5NS1", "NS4NS1", "NS3NS1", "NS2NS1", "NS5", "NS4", "NS3", "NS2", "NS1",
]  # fmt: skip
# panel a grid (left/top edge and cell size) and the colour bar, in pixels
GRID_X, GRID_Y, CELL_W, CELL_H = 155.5, 35.5, 24.36, 24.4
BAR_X, BAR_TOP, BAR_BOTTOM = (1531, 1547), 313, 442


def extract_image(pdf: Path, directory: Path) -> Path:
    subprocess.run(
        ["pdfimages", "-f", str(PAGE), "-l", str(PAGE), "-png", str(pdf), str(directory / "fig")],
        check=True,
    )
    image = directory / "fig-000.png"
    if Image.open(image).size != (1648, 769):
        raise ValueError(f"Unexpected Supplementary Figure 14 image size in {pdf}")
    return image


def measure(image: Path) -> pd.DataFrame:
    pixels = np.asarray(Image.open(image).convert("RGB")).astype(float)
    bar_rows = np.arange(BAR_TOP, BAR_BOTTOM + 1)
    bar = np.median(pixels[BAR_TOP : BAR_BOTTOM + 1, BAR_X[0] : BAR_X[1]], axis=1)
    bar_values = 0.5 - (bar_rows - BAR_TOP) / (BAR_BOTTOM - BAR_TOP)
    values = np.zeros((len(TASK), len(TASK_FREE)))
    for row in range(len(TASK)):
        for column in range(len(TASK_FREE)):
            x = round(GRID_X + CELL_W * (column + 0.5))
            y = round(GRID_Y + CELL_H * (row + 0.5))
            colour = np.median(pixels[y - 6 : y + 7, x - 6 : x + 7].reshape(-1, 3), axis=0)
            distance = ((bar - colour) ** 2).sum(axis=1)
            values[row, column] = bar_values[np.argmin(distance)]
    return pd.DataFrame(values, index=pd.Index(TASK, name="task"), columns=TASK_FREE)


def check(table: pd.DataFrame, image: Path, path: Path) -> None:
    """Original panel next to the digitized values, for visual comparison."""
    import matplotlib.pyplot as plt

    original = Image.open(image).crop((40, 0, 512, 769))
    figure, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].imshow(original)
    axes[0].axis("off")
    axes[1].imshow(table.to_numpy(), cmap="bwr", vmin=-0.5, vmax=0.5)
    axes[1].set_yticks(range(len(TASK)), TASK, fontsize=6)
    axes[1].set_xticks(range(len(TASK_FREE)), TASK_FREE, rotation=90, fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        image = extract_image(args.pdf, Path(directory))
        table = measure(image)
        if args.check:
            check(table, image, args.check)
    table.round(3).to_csv(args.output, sep="\t")
    print(f"Wrote {args.output}: {table.shape[0]} task x {table.shape[1]} task-free features")


if __name__ == "__main__":
    main()
