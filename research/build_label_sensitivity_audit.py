from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds, xy
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parent / "recovered_data"
MODEL = ROOT / "model_inputs"
READY = ROOT / "figure_ready"
OUT = ROOT / "label_audit"
FIG = ROOT / "figure_outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

WATER_THRESHOLDS = [90, 95, 99]
BLOCK_M = 10_000.0
THIN = 3
LABEL_COLORS = ["#f3f3f3", "#d8c6a3", "#4c78a8"]
DENSITY_COLORS = ["#eef1f3", "#c7d4dd", "#8fa9bb", "#55758c"]
DENSITY_EDGES = [1, 1_000, 5_000, 10_000, np.inf]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile, src.transform


def save_formats(fig, stem: str):
    paths = []
    for ext, dpi in (("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        paths.append(p)
    return paths


def apply_tick_font(axis, family: str) -> None:
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontfamily(family)
        label.set_fontsize(7)
    for offset_text in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
        offset_text.set_fontfamily(family)
        offset_text.set_fontsize(7)


def add_scale_north(axis, bounds: tuple[float, float, float, float], western: str) -> None:
    left, bottom, right, top = bounds
    x0 = left + 0.06 * (right - left)
    y0 = bottom + 0.07 * (top - bottom)
    length = 10_000.0
    axis.plot([x0, x0 + length], [y0, y0], color="#222222", linewidth=1.2, zorder=8)
    tick = 0.008 * (top - bottom)
    axis.plot([x0, x0], [y0 - tick, y0 + tick], color="#222222", linewidth=0.8, zorder=8)
    axis.plot([x0 + length, x0 + length], [y0 - tick, y0 + tick], color="#222222", linewidth=0.8, zorder=8)
    axis.text(x0 + length / 2, y0 + 0.018 * (top - bottom), "10 km", ha="center", va="bottom", fontfamily=western, fontsize=7)
    axis.annotate("N", xy=(0.94, 0.91), xytext=(0.94, 0.78), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="bottom", fontfamily=western, fontsize=8, arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": "#222222"})


def main() -> None:
    occurrence, profile, transform = read(MODEL / "jrc_gsw_v1_5_occurrence_30m.tif")
    seasonality, _, _ = read(MODEL / "jrc_gsw_v1_5_seasonality_30m.tif")
    extent, _, _ = read(MODEL / "jrc_gsw_v1_5_extent_30m.tif")
    common, _, _ = read(ROOT / "planetary_computer_rtc/common_valid_mask_30m.tif")
    gate = json.loads((READY / "font_gate.json").read_text(encoding="utf-8"))
    western = gate.get("selected_western_family")
    chinese = gate.get("selected_chinese_family")
    if not gate.get("formal_export_allowed") or western != "Times New Roman" or chinese not in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}:
        raise RuntimeError("Formal export blocked by exact font gate")
    if occurrence.shape != common.shape or seasonality.shape != common.shape or extent.shape != common.shape:
        raise RuntimeError("JRC/common grid mismatch")

    rows = np.arange(0, occurrence.shape[0], THIN)
    cols = np.arange(0, occurrence.shape[1], THIN)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr = rr.ravel(); cc = cc.ravel()
    valid = common[rr, cc] == 1
    rr = rr[valid]; cc = cc[valid]
    xs, ys = xy(transform, rr, cc, offset="center")
    xs = np.asarray(xs); ys = np.asarray(ys)
    block_x = np.floor(xs / BLOCK_M).astype(int)
    block_y = np.floor(ys / BLOCK_M).astype(int)
    block_id = np.char.add(np.char.add(block_x.astype(str), "_"), block_y.astype(str))

    stable_land = (occurrence[rr, cc] == 0) & (extent[rr, cc] == 0)
    summaries = []
    block_rows = []
    for threshold in WATER_THRESHOLDS:
        water = (seasonality[rr, cc] == 12) & (occurrence[rr, cc] >= threshold) & (extent[rr, cc] == 1)
        summaries.append({
            "water_occurrence_threshold": threshold,
            "water_samples_90m": int(water.sum()),
            "stable_land_samples_90m": int(stable_land.sum()),
            "ambiguous_or_other_samples_90m": int((~water & ~stable_land).sum()),
        })
        frame = pd.DataFrame({"block_id": block_id, "water": water.astype(int), "land": stable_land.astype(int)})
        grouped = frame.groupby("block_id", as_index=False).sum()
        grouped["threshold"] = threshold
        grouped["has_both_classes"] = (grouped["water"] > 0) & (grouped["land"] > 0)
        block_rows.append(grouped)

    summary = pd.DataFrame(summaries)
    blocks = pd.concat(block_rows, ignore_index=True)
    summary.to_csv(OUT / "jrc_label_threshold_sensitivity.csv", index=False)
    blocks.to_csv(OUT / "jrc_10km_block_class_counts.csv", index=False)

    t95 = blocks[blocks["threshold"] == 95].copy()
    t95 = t95.sort_values(["has_both_classes", "water", "land"], ascending=[False, False, False])
    selected = t95[t95["has_both_classes"]].head(20).copy()
    selected.to_csv(OUT / "candidate_20_blocks_threshold95.csv", index=False)
    selected_ids = set(selected["block_id"].astype(str))

    left, bottom, right, top = array_bounds(occurrence.shape[0], occurrence.shape[1], transform)
    bounds = (left, bottom, right, top)
    row_grid = rows[:, None]
    col_grid = cols[None, :]
    common_90 = common[row_grid, col_grid] == 1
    land_90 = (occurrence[row_grid, col_grid] == 0) & (extent[row_grid, col_grid] == 0) & common_90
    water_90 = (seasonality[row_grid, col_grid] == 12) & (occurrence[row_grid, col_grid] >= 95) & (extent[row_grid, col_grid] == 1) & common_90
    label_map = np.zeros(common_90.shape, dtype=np.uint8)
    label_map[land_90] = 1
    label_map[water_90] = 2

    fig, axes = plt.subplots(1, 2, figsize=(7.0866, 2.85), sharex=True, sharey=True)
    ax = axes[0]
    ax.imshow(label_map, extent=[left, right, bottom, top], origin="upper", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(LABEL_COLORS), vmin=0, vmax=2, zorder=0)
    for block in selected.itertuples(index=False):
        bx, by = (int(part) for part in str(block.block_id).split("_", 1))
        ax.add_patch(Rectangle((bx * BLOCK_M, by * BLOCK_M), BLOCK_M, BLOCK_M, fill=False, edgecolor="#7a3e65", linewidth=0.8, zorder=4))

    stations_path = ROOT / "grdc_pair_station_metadata.csv"
    if stations_path.exists():
        stations = pd.read_csv(stations_path)
        lon_col = "long_pp" if "long_pp" in stations.columns else "long_org"
        lat_col = "lat_pp" if "lat_pp" in stations.columns else "lat_org"
        sx, sy = warp_transform("EPSG:4326", profile["crs"], stations[lon_col].astype(float).tolist(), stations[lat_col].astype(float).tolist())
        ax.scatter(sx, sy, s=18, marker="o", facecolor="#ffffff", edgecolor="#222222", linewidth=0.6, zorder=5)

    ax.set_title("90 m 候选标签（JRC 阈值 95%）", fontfamily=chinese, fontsize=9)
    ax.set_xlabel("Easting (m)", fontfamily=western, fontsize=8)
    ax.set_ylabel("Northing (m)", fontfamily=western, fontsize=8)
    ax.set_aspect("equal")
    apply_tick_font(ax, western)
    ax.legend(handles=[
        Patch(facecolor=LABEL_COLORS[1], edgecolor="none", label="Candidate stable land"),
        Patch(facecolor=LABEL_COLORS[2], edgecolor="none", label="Candidate permanent water"),
    ], loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.86, prop={"family": western, "size": 5.8})
    add_scale_north(ax, bounds, western)

    ax = axes[1]
    for block in t95.itertuples(index=False):
        bx, by = (int(part) for part in str(block.block_id).split("_", 1))
        total = int(block.water) + int(block.land)
        density_class = int(np.digitize([total], DENSITY_EDGES[1:-1], right=False)[0])
        density_class = min(density_class, len(DENSITY_COLORS) - 1)
        ax.add_patch(Rectangle((bx * BLOCK_M, by * BLOCK_M), BLOCK_M, BLOCK_M, facecolor=DENSITY_COLORS[density_class], edgecolor="#ffffff", linewidth=0.25, zorder=1))
        if str(block.block_id) in selected_ids:
            ax.add_patch(Rectangle((bx * BLOCK_M, by * BLOCK_M), BLOCK_M, BLOCK_M, fill=False, edgecolor="#7a3e65", linewidth=0.85, zorder=3))
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)
    ax.set_title("10 km 标签样本密度与双类分块", fontfamily=chinese, fontsize=9)
    ax.set_xlabel("Easting (m)", fontfamily=western, fontsize=8)
    ax.set_aspect("equal")
    apply_tick_font(ax, western)
    density_labels = ["1–999", "1,000–4,999", "5,000–9,999", "≥10,000"]
    density_handles = [Patch(facecolor=color, edgecolor="#aaaaaa", linewidth=0.3, label=label) for color, label in zip(DENSITY_COLORS, density_labels)]
    density_handles.append(Patch(facecolor="none", edgecolor="#7a3e65", label="Selected dual-class block"))
    ax.legend(handles=density_handles, title="Candidate samples / block", loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.86, columnspacing=0.8, handlelength=1.4, prop={"family": western, "size": 5.2}, title_fontproperties={"family": western, "size": 5.4})

    fig.suptitle("候选标签与 10 km 空间分块（公开源重建）", y=0.985, fontfamily=chinese, fontsize=10)
    fig.text(0.5, 0.018, f"95% 阈值下真实可用双类分块为 {len(selected)} 个；未补造第 20 个分块。", ha="center", va="bottom", fontfamily=chinese, fontsize=6.8)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.78, wspace=0.14)
    rendered = save_formats(fig, "figure05_candidate_label_block_audit")
    plt.close(fig)

    report = {
        "source": "JRC Global Surface Water v1.5 and recovered common RTC grid",
        "grid_crs": str(profile.get("crs")),
        "grid_shape": list(occurrence.shape),
        "sampling_spacing_m": 90,
        "block_size_m": BLOCK_M,
        "thresholds_tested": WATER_THRESHOLDS,
        "main_map_threshold": 95,
        "stable_land_candidate_rule": "occurrence == 0 and extent == 0",
        "permanent_water_candidate_rule": "seasonality == 12 and occurrence >= threshold and extent == 1",
        "candidate_blocks_at_threshold95": int(len(selected)),
        "shortfall_vs_frozen_20_blocks": int(max(0, 20 - len(selected))),
        "no_synthetic_block_added": True,
        "formal_export_allowed": True,
        "selected_western_family": western,
        "selected_chinese_family": chinese,
        "rendered_files": [p.name for p in rendered],
        "scientific_constraint": "This spatial figure uses real JRC rasters and real 10 km block counts. It does not reproduce unavailable frozen HPC label exclusions or invent a twentieth block.",
        "qa_passed": bool(len(rendered) == 4 and len(selected) > 0 and summary["water_samples_90m"].min() > 0 and summary["stable_land_samples_90m"].min() > 0),
    }
    if not report["qa_passed"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    (OUT / "label_audit_status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = []
    for p in sorted(list(OUT.glob("*")) + rendered):
        if p.is_file() and p.name != "sha256_manifest.csv":
            manifest.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(manifest).to_csv(OUT / "sha256_manifest.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
