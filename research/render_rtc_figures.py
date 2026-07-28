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
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    return outputs


def apply_font(ax: plt.Axes, western: str) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
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
    raster_meta: dict[str, dict] = {}

    for phase, _, _ in PHASES:
        vv, meta_vv = read_raster(RTC / phase / f"{phase}_vv_rtc_30m.tif")
        vh, meta_vh = read_raster(RTC / phase / f"{phase}_vh_rtc_30m.tif")
        mask, meta_mask = read_raster(RTC / phase / f"{phase}_valid_mask_30m.tif")
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
        raster_meta[phase] = {"vv": meta_vv, "vh": meta_vh, "mask": meta_mask}

    vv_values = np.concatenate([a[np.isfinite(a)] for a in vv_db.values()])
    vh_values = np.concatenate([a[np.isfinite(a)] for a in vh_db.values()])
    vv_limits = [float(np.percentile(vv_values, 2)), float(np.percentile(vv_values, 98))]
    vh_limits = [float(np.percentile(vh_values, 2)), float(np.percentile(vh_values, 98))]
    if not vv_limits[0] < vv_limits[1] or not vh_limits[0] < vh_limits[1]:
        raise RuntimeError("Invalid shared display limits")

    rendered: list[Path] = []

    fig, axes = plt.subplots(1, 5, figsize=(7.0866, 2.25), sharex=True, sharey=True)
    image = None
    for ax, (phase, label, date_text) in zip(axes, PHASES):
        image = ax.imshow(vv_db[phase], cmap="gray", vmin=vv_limits[0], vmax=vv_limits[1], interpolation="nearest")
        ax.set_title(f"{label}\n{date_text}", fontfamily=chinese, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(image, ax=list(axes), fraction=0.018, pad=0.015)
    cbar.set_label("VV RTC (dB)", fontfamily=western, fontsize=8)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily(western)
        tick.set_fontsize(7)
    fig.suptitle("五阶段 Sentinel-1 RTC 后向散射", fontfamily=chinese, fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.015, right=0.94, top=0.82, bottom=0.03, wspace=0.03)
    rendered.extend(save_four_formats(fig, "figure03_five_phase_rtc_vv"))
    plt.close(fig)

    coverage = [float(phase_masks[p].mean()) for p, _, _ in PHASES]
    common_fraction = float(common_valid.mean())
    fig, ax = plt.subplots(figsize=(7.0866, 3.8))
    x = np.arange(len(PHASES))
    bars = ax.bar(x, np.asarray(coverage) * 100.0, width=0.62, edgecolor="black", linewidth=0.5)
    ax.axhline(common_fraction * 100.0, color="black", linestyle="--", linewidth=0.9, label="Five-phase common valid")
    ax.set_xticks(x, [label for _, label, _ in PHASES], fontfamily=chinese, fontsize=8)
    ax.set_ylabel("Valid coverage (%)", fontfamily=western, fontsize=9)
    ax.set_title("阶段有效覆盖率与共同有效区", fontfamily=chinese, fontsize=10)
    ax.set_ylim(0, 102)
    apply_font(ax, western)
    for bar, value in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, min(value * 100.0 + 1.0, 100.5), f"{value * 100.0:.2f}", ha="center", va="bottom", fontfamily=western, fontsize=7)
    ax.legend(frameon=False, prop={"family": western, "size": 8}, loc="lower right")
    fig.tight_layout()
    rendered.extend(save_four_formats(fig, "figure04_valid_coverage"))
    plt.close(fig)

    qa = {
        "source": status.get("source"),
        "phase_count": len(PHASES),
        "expected_phase_count": 5,
        "grid_shape": list(common.shape),
        "crs": common_meta["crs"],
        "resolution_m": 30.0,
        "common_valid_pixels": int(common_valid.sum()),
        "common_valid_fraction": common_fraction,
        "phase_valid_fraction": {p: float(phase_masks[p].mean()) for p, _, _ in PHASES},
        "vv_shared_display_percentiles_db": vv_limits,
        "vh_shared_percentiles_db": vh_limits,
        "all_rasters_real_and_present": bool(status.get("all_phase_rasters_complete")),
        "formal_export_allowed": bool(gate.get("formal_export_allowed")),
        "selected_western_family": western,
        "selected_chinese_family": chinese,
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
