from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent / "recovered_data"
RTC = ROOT / "planetary_computer_rtc"
READY = ROOT / "figure_ready"
OUT = ROOT / "figure_outputs"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = [
    ("baseline", "基线", "2017-11-01"),
    ("rising", "上涨期", "2017-11-13"),
    ("peak", "峰值期", "2017-11-25"),
    ("early_recession", "早退水期", "2017-12-07"),
    ("late_recession", "晚退水期", "2017-12-19"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
        meta = {
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": tuple(src.bounds),
        }
    return array, meta


def save_four_formats(fig: plt.Figure, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for ext, dpi in (("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    return outputs


def set_numeric_tick_font(ax: plt.Axes, western: str) -> None:
    for label in ax.get_yticklabels():
        label.set_fontfamily(western)
        label.set_fontsize(7)


def main() -> None:
    gate_path = READY / "font_gate.json"
    status_path = RTC / "rtc_extraction_status.json"
    common_path = RTC / "common_valid_mask_30m.tif"
    required = [gate_path, status_path, common_path]
    for phase, _, _ in PHASES:
        required.extend([
            RTC / phase / f"{phase}_vv_rtc_30m.tif",
            RTC / phase / f"{phase}_vh_rtc_30m.tif",
            RTC / phase / f"{phase}_valid_mask_30m.tif",
        ])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing RTC figure inputs: " + ", ".join(missing))

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    western = gate.get("selected_western_family")
    chinese = gate.get("selected_chinese_family")
    if not gate.get("formal_export_allowed") or western != "Times New Roman" or chinese not in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}:
        raise RuntimeError("Formal RTC export blocked by exact font gate")

    common, common_meta = read_raster(common_path)
    common_valid = common == 1
    vv_db: dict[str, np.ndarray] = {}
    vh_db: dict[str, np.ndarray] = {}
    phase_masks: dict[str, np.ndarray] = {}

    for phase, _, _ in PHASES:
        vv, meta_vv = read_raster(RTC / phase / f"{phase}_vv_rtc_30m.tif")
        vh, meta_vh = read_raster(RTC / phase / f"{phase}_vh_rtc_30m.tif")
        mask, _ = read_raster(RTC / phase / f"{phase}_valid_mask_30m.tif")
        if meta_vv["crs"] != "EPSG:32610" or meta_vh["crs"] != "EPSG:32610":
            raise RuntimeError(f"Unexpected CRS for {phase}")
        if vv.shape != common.shape or vh.shape != common.shape or mask.shape != common.shape:
            raise RuntimeError(f"Grid mismatch for {phase}")
        valid = common_valid & (mask == 1) & np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
        if not valid.any():
            raise RuntimeError(f"No common valid positive RTC pixels for {phase}")
        vv_out = np.full(vv.shape, np.nan, dtype="float32")
        vh_out = np.full(vh.shape, np.nan, dtype="float32")
        vv_out[valid] = 10.0 * np.log10(vv[valid])
        vh_out[valid] = 10.0 * np.log10(vh[valid])
        vv_db[phase] = vv_out
        vh_db[phase] = vh_out
        phase_masks[phase] = mask == 1

    vv_values = np.concatenate([a[np.isfinite(a)] for a in vv_db.values()])
    vh_values = np.concatenate([a[np.isfinite(a)] for a in vh_db.values()])
    vv_limits = [float(np.percentile(vv_values, 2)), float(np.percentile(vv_values, 98))]
    vh_limits = [float(np.percentile(vh_values, 2)), float(np.percentile(vh_values, 98))]
    if not vv_limits[0] < vv_limits[1] or not vh_limits[0] < vh_limits[1]:
        raise RuntimeError("Invalid shared display limits")

    rendered: list[Path] = []

    fig = plt.figure(figsize=(7.0866, 2.45))
    grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.055], wspace=0.035)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    cax = fig.add_subplot(grid[0, 5])
    image = None
    for ax, (phase, label, date_text) in zip(axes, PHASES):
        image = ax.imshow(vv_db[phase], cmap="gray", vmin=vv_limits[0], vmax=vv_limits[1], interpolation="nearest")
        ax.set_title(f"{label}\n{date_text}", fontfamily=chinese, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("VV RTC (dB)", fontfamily=western, fontsize=8)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily(western)
        tick.set_fontsize(7)
    fig.suptitle("五阶段 Sentinel-1 RTC 后向散射（公开数据重建）", fontfamily=chinese, fontsize=10, y=0.985)
    fig.subplots_adjust(left=0.012, right=0.965, top=0.79, bottom=0.035)
    rendered.extend(save_four_formats(fig, "figure03_five_phase_rtc_vv"))
    plt.close(fig)

    # The available Planetary Computer RTC assets contain VV/VH COGs but no
    # OPERA-style quality mask. Therefore this figure is deliberately limited
    # to non-NoData data availability and must not be interpreted as strict
    # sensor-quality coverage or as a replica of the original HPC mask result.
    coverage = [float(phase_masks[p].mean()) for p, _, _ in PHASES]
    common_fraction = float(common_valid.mean())
    fig, ax = plt.subplots(figsize=(7.0866, 3.35))
    x = np.arange(len(PHASES))
    bars = ax.bar(
        x,
        np.asarray(coverage) * 100.0,
        width=0.58,
        edgecolor="black",
        linewidth=0.5,
        color="#8C8C8C",
    )
    ax.axhline(
        common_fraction * 100.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="Five-phase common data availability",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in PHASES], fontfamily=chinese, fontsize=8)
    ax.set_ylabel("Non-NoData coverage (%)", fontfamily=western, fontsize=9)
    ax.set_title("AOI 内 RTC 数据可用覆盖", fontfamily=chinese, fontsize=10)
    ax.set_ylim(0, 102)
    set_numeric_tick_font(ax, western)
    for bar, value in zip(bars, coverage):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value * 100.0 + 0.8, 100.35),
            f"{value * 100.0:.2f}",
            ha="center",
            va="bottom",
            fontfamily=western,
            fontsize=7,
        )
    ax.text(
        0.01,
        0.025,
        "仅表示 VV/VH 非 NoData；不等同于 OPERA 质量掩膜或原 HPC 严格有效区",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontfamily=chinese,
        fontsize=7,
    )
    ax.legend(frameon=False, prop={"family": western, "size": 8}, loc="lower right")
    fig.tight_layout(pad=0.6)
    rendered.extend(save_four_formats(fig, "figure04_rtc_data_availability"))
    plt.close(fig)

    qa = {
        "source": status.get("source"),
        "phase_count": len(PHASES),
        "expected_phase_count": 5,
        "grid_shape": list(common.shape),
        "crs": common_meta["crs"],
        "resolution_m": 30.0,
        "common_valid_pixels": int(common_valid.sum()),
        "common_data_availability_fraction": common_fraction,
        "phase_data_availability_fraction": {p: float(phase_masks[p].mean()) for p, _, _ in PHASES},
        "phase_mask_definition": "1 where reprojected VV and VH are finite and non-NoData; no OPERA quality mask is available in the Planetary Computer RTC item",
        "strict_quality_mask_available": False,
        "figure04_scope": "Data-availability QA only; not strict quality coverage and not the original HPC valid-area result",
        "vv_shared_display_percentiles_db": vv_limits,
        "vh_shared_display_percentiles_db": vh_limits,
        "all_rasters_real_and_present": bool(status.get("all_phase_rasters_complete")),
        "formal_export_allowed": bool(gate.get("formal_export_allowed")),
        "selected_western_family": western,
        "selected_chinese_family": chinese,
        "visual_qa_requirements": {
            "figure03_colorbar_has_dedicated_axis": True,
            "figure03_five_titles_complete": True,
            "figure04_chinese_ticks_preserved": True,
            "figure04_quality_scope_disclosed": True,
        },
        "scientific_caveat": status.get("scientific_caveat"),
        "rendered_files": [p.name for p in rendered],
    }
    qa["qa_passed"] = bool(
        qa["phase_count"] == 5
        and qa["crs"] == "EPSG:32610"
        and qa["common_valid_pixels"] > 0
        and qa["all_rasters_real_and_present"]
        and qa["formal_export_allowed"]
        and len(rendered) == 8
    )
    if not qa["qa_passed"]:
        raise RuntimeError("RTC figure QA failed: " + json.dumps(qa, ensure_ascii=False))

    (OUT / "figure03_04_qa_status.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = OUT / "sha256_manifest.csv"
    rows = []
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name != manifest_path.name:
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
