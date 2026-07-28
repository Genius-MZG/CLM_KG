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
from rasterio.transform import xy

ROOT = Path(__file__).resolve().parent / "recovered_data"
MODEL = ROOT / "model_inputs"
READY = ROOT / "figure_ready"
OUT = ROOT / "label_audit"
FIG = ROOT / "figure_outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

WATER_THRESHOLDS = [90, 95, 99]
BLOCK_M = 10_000.0
THIN = 3  # 90 m on the recovered 30 m grid


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

    fig, axes = plt.subplots(1, 2, figsize=(7.0866, 3.2))
    ax = axes[0]
    ax.plot(summary["water_occurrence_threshold"], summary["water_samples_90m"], marker="o", linewidth=1.2)
    ax.set_xlabel("JRC occurrence threshold (%)", fontfamily=western, fontsize=8)
    ax.set_ylabel("Candidate permanent-water samples", fontfamily=western, fontsize=8)
    ax.set_title("永久水体标签阈值敏感性", fontfamily=chinese, fontsize=9)
    ax.grid(True, linewidth=0.35, alpha=0.4)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(western); lab.set_fontsize(7)

    ax = axes[1]
    both = blocks.groupby("threshold")["has_both_classes"].sum().reindex(WATER_THRESHOLDS)
    ax.bar(np.arange(len(WATER_THRESHOLDS)), both.values, width=0.58, edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(WATER_THRESHOLDS)))
    ax.set_xticklabels([str(x) for x in WATER_THRESHOLDS], fontfamily=western, fontsize=7)
    ax.set_xlabel("JRC occurrence threshold (%)", fontfamily=western, fontsize=8)
    ax.set_ylabel("10 km blocks with both classes", fontfamily=western, fontsize=8)
    ax.set_title("空间分块可用性", fontfamily=chinese, fontsize=9)
    for lab in ax.get_yticklabels():
        lab.set_fontfamily(western); lab.set_fontsize(7)
    fig.suptitle("候选标签与 10 km 分块审计（非原始冻结标签复现）", fontfamily=chinese, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    rendered = save_formats(fig, "figure05_candidate_label_block_audit")
    plt.close(fig)

    report = {
        "source": "JRC Global Surface Water v1.5 and recovered common RTC grid",
        "grid_crs": str(profile.get("crs")),
        "grid_shape": list(occurrence.shape),
        "sampling_spacing_m": 90,
        "block_size_m": BLOCK_M,
        "thresholds_tested": WATER_THRESHOLDS,
        "stable_land_candidate_rule": "occurrence == 0 and extent == 0",
        "permanent_water_candidate_rule": "seasonality == 12 and occurrence >= threshold and extent == 1",
        "candidate_blocks_at_threshold95": int(len(selected)),
        "formal_export_allowed": True,
        "selected_western_family": western,
        "selected_chinese_family": chinese,
        "rendered_files": [p.name for p in rendered],
        "scientific_constraint": "This is a transparent sensitivity audit using real rasters. It is not claimed to reproduce the unavailable frozen HPC label exclusions or the original model metrics.",
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
