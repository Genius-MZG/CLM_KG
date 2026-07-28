from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent / "recovered_data"
RTC = ROOT / "planetary_computer_rtc"
READY = ROOT / "figure_ready"
OUT = ROOT / "figure_outputs"
PHASES = [
    ("baseline", "基线"),
    ("rising", "上涨期"),
    ("peak", "峰值期"),
    ("early_recession", "早退水期"),
    ("late_recession", "晚退水期"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_formats(fig: plt.Figure) -> list[str]:
    names = []
    for ext, dpi in (("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)):
        path = OUT / f"figure04_rtc_data_availability.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        names.append(path.name)
    return names


def main() -> None:
    gate = json.loads((READY / "font_gate.json").read_text(encoding="utf-8"))
    western = gate.get("selected_western_family")
    chinese = gate.get("selected_chinese_family")
    if not gate.get("formal_export_allowed") or western != "Times New Roman" or chinese not in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}:
        raise RuntimeError("Formal export blocked by exact font gate")

    masks = []
    bounds = None
    crs = None
    for phase, _ in PHASES:
        path = RTC / phase / f"{phase}_valid_mask_30m.tif"
        with rasterio.open(path) as src:
            masks.append(src.read(1).astype(bool))
            bounds = tuple(src.bounds)
            crs = str(src.crs)
    if crs != "EPSG:32610" or any(mask.shape != masks[0].shape for mask in masks):
        raise RuntimeError("RTC availability grids are not aligned in EPSG:32610")

    stack = np.stack(masks)
    count = stack.sum(axis=0).astype("uint8")
    fractions = stack.mean(axis=(1, 2))
    common = stack.all(axis=0)
    left, bottom, right, top = bounds
    extent = [left / 1000, right / 1000, bottom / 1000, top / 1000]

    fig = plt.figure(figsize=(7.0866, 3.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.24)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    cmap = ListedColormap(["#F4F4F4", "#D8E3E7", "#B9CDD4", "#94B5BF", "#6F9BA8", "#497E8D"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    image = ax_map.imshow(count, cmap=cmap, norm=norm, interpolation="nearest", extent=extent, origin="upper")
    if common.any() and not common.all():
        xs = np.linspace(extent[0], extent[1], common.shape[1])
        ys = np.linspace(extent[3], extent[2], common.shape[0])
        ax_map.contour(xs, ys, common.astype("uint8"), levels=[0.5], colors="black", linewidths=0.7)
    ax_map.set_title("五期双极化数据可用次数", fontfamily=chinese, fontsize=9)
    ax_map.set_xlabel("Easting (km)", fontfamily=western, fontsize=8)
    ax_map.set_ylabel("Northing (km)", fontfamily=western, fontsize=8)
    for label in [*ax_map.get_xticklabels(), *ax_map.get_yticklabels()]:
        label.set_fontfamily(western)
        label.set_fontsize(7)
    cbar = fig.colorbar(image, ax=ax_map, orientation="horizontal", fraction=0.075, pad=0.12, ticks=np.arange(6))
    cbar.set_label("Available phases", fontfamily=western, fontsize=8)
    for label in cbar.ax.get_xticklabels():
        label.set_fontfamily(western)
        label.set_fontsize(7)

    x = np.arange(len(PHASES))
    bars = ax_bar.bar(x, fractions * 100, width=0.58, edgecolor="black", linewidth=0.5, color="#8C8C8C")
    ax_bar.axhline(common.mean() * 100, color="black", linestyle="--", linewidth=0.9, label="Five-phase common")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([label for _, label in PHASES], fontfamily=chinese, fontsize=7, rotation=28, ha="right")
    ax_bar.set_ylabel("Non-NoData coverage (%)", fontfamily=western, fontsize=8)
    ax_bar.set_title("阶段覆盖率", fontfamily=chinese, fontsize=9)
    ax_bar.set_ylim(0, 102)
    for label in ax_bar.get_yticklabels():
        label.set_fontfamily(western)
        label.set_fontsize(7)
    for bar, value in zip(bars, fractions):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, min(value * 100 + 0.7, 100.3), f"{value * 100:.1f}", ha="center", va="bottom", fontfamily=western, fontsize=6.5)
    ax_bar.legend(frameon=False, prop={"family": western, "size": 7}, loc="lower right")

    fig.suptitle("RTC 数据可用区空间分布与覆盖统计", fontfamily=chinese, fontsize=10, y=0.99)
    fig.text(0.012, 0.012, "仅表示 VV/VH 非 NoData；不等同于 OPERA 质量掩膜或原 HPC 严格有效区", ha="left", va="bottom", fontfamily=chinese, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.22)
    rendered = save_formats(fig)
    plt.close(fig)

    qa = {
        "crs": crs,
        "grid_shape": list(count.shape),
        "availability_count_range": [int(count.min()), int(count.max())],
        "phase_data_availability_fraction": {phase: float(value) for (phase, _), value in zip(PHASES, fractions)},
        "common_data_availability_fraction": float(common.mean()),
        "strict_quality_mask_available": False,
        "scope": "Spatial non-NoData availability only; not an OPERA quality mask or original HPC strict valid-area result",
        "formal_export_allowed": True,
        "rendered_files": rendered,
        "sha256": {name: digest(OUT / name) for name in rendered},
        "qa_passed": crs == "EPSG:32610" and len(rendered) == 4,
    }
    (OUT / "figure04_spatial_qa_status.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not qa["qa_passed"]:
        raise RuntimeError(json.dumps(qa, ensure_ascii=False))
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
