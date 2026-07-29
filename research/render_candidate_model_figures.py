from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent / "recovered_data"
CANDIDATE = ROOT / "candidate_model"
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
THRESHOLDS = [0.35, 0.50, 0.65]
WIDTH_IN = 180.0 / 25.4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
        metadata = {
            "crs": str(src.crs),
            "shape": [src.height, src.width],
            "transform": tuple(src.transform),
            "nodata": src.nodata,
        }
    return array, metadata


def save_four_formats(fig: plt.Figure, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for extension, dpi in (("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)):
        path = OUT / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": None, "pad_inches": 0}
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def set_axis_fonts(ax: plt.Axes, western: str, chinese: str) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(western)
        label.set_fontsize(7)
    ax.xaxis.label.set_fontfamily(western)
    ax.yaxis.label.set_fontfamily(western)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.title.set_fontfamily(chinese)
    ax.title.set_fontsize(9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)


def main() -> None:
    required = [
        READY / "font_gate.json",
        CANDIDATE / "candidate_model_status.json",
        CANDIDATE / "candidate_oof_predictions.csv",
        CANDIDATE / "candidate_spatial_cv_folds.csv",
        CANDIDATE / "candidate_connected_water_area.csv",
        CANDIDATE / "downstream_basin_common_domain_30m.tif",
    ]
    for phase, _, _ in PHASES:
        required.extend(
            [
                CANDIDATE / f"{phase}_candidate_water_probability_30m.tif",
                CANDIDATE / f"{phase}_connected_water_t50_30m.tif",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing candidate-model figure inputs: " + ", ".join(missing))

    gate = json.loads((READY / "font_gate.json").read_text(encoding="utf-8"))
    status = json.loads((CANDIDATE / "candidate_model_status.json").read_text(encoding="utf-8"))
    western = gate.get("selected_western_family")
    chinese = gate.get("selected_chinese_family")
    if (
        not gate.get("formal_export_allowed")
        or western != "Times New Roman"
        or chinese not in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}
    ):
        raise RuntimeError("Formal candidate-model export blocked by exact font gate")
    if not status.get("qa_passed"):
        raise RuntimeError("Candidate-model QA did not pass")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    rendered: list[Path] = []

    oof = pd.read_csv(CANDIDATE / "candidate_oof_predictions.csv")
    folds = pd.read_csv(CANDIDATE / "candidate_spatial_cv_folds.csv")
    area = pd.read_csv(CANDIDATE / "candidate_connected_water_area.csv")
    if oof["probability"].isna().any() or set(oof["label"].unique()) != {0, 1}:
        raise RuntimeError("Invalid out-of-fold prediction table")
    if len(folds) != 5:
        raise RuntimeError(f"Expected five spatial folds; found {len(folds)}")

    labels = oof["label"].to_numpy(dtype="uint8")
    probabilities = oof["probability"].to_numpy(dtype="float64")
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    observed_fraction, mean_probability = calibration_curve(
        labels,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    figure06 = plt.figure(figsize=(WIDTH_IN, 3.05))
    grid06 = figure06.add_gridspec(1, 3, left=0.075, right=0.975, bottom=0.18, top=0.78, wspace=0.34)
    ax_roc = figure06.add_subplot(grid06[0, 0])
    ax_cal = figure06.add_subplot(grid06[0, 1])
    ax_fold = figure06.add_subplot(grid06[0, 2])

    ax_roc.plot(false_positive_rate, true_positive_rate, linewidth=1.4, color="#35618f")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=0.7, color="#666666")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title("空间分块 ROC", fontfamily=chinese)
    ax_roc.text(
        0.96,
        0.07,
        f"AUC = {status['oof_auc']:.3f}",
        ha="right",
        va="bottom",
        fontfamily=western,
        fontsize=7.5,
    )
    set_axis_fonts(ax_roc, western, chinese)

    ax_cal.plot(mean_probability, observed_fraction, marker="o", markersize=3.5, linewidth=1.2, color="#2f7f75")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", linewidth=0.7, color="#666666")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed water fraction")
    ax_cal.set_title("概率校准", fontfamily=chinese)
    ax_cal.text(
        0.96,
        0.07,
        f"Brier = {status['oof_brier']:.3f}",
        ha="right",
        va="bottom",
        fontfamily=western,
        fontsize=7.5,
    )
    set_axis_fonts(ax_cal, western, chinese)

    fold_x = folds["fold"].to_numpy()
    ax_fold.plot(fold_x, folds["auc"], marker="o", markersize=3.5, linewidth=1.2, color="#35618f", label="AUC")
    ax_fold.set_ylim(0.90, 1.005)
    ax_fold.set_xticks(fold_x)
    ax_fold.set_xlabel("Spatial fold")
    ax_fold.set_ylabel("AUC")
    ax_fold.set_title("折间稳定性", fontfamily=chinese)
    set_axis_fonts(ax_fold, western, chinese)
    ax_brier = ax_fold.twinx()
    ax_brier.plot(fold_x, folds["brier"], marker="s", markersize=3.2, linewidth=1.0, color="#b56a3a", label="Brier")
    ax_brier.set_ylabel("Brier score", fontfamily=western, fontsize=8)
    ax_brier.tick_params(width=0.6, length=3)
    for label in ax_brier.get_yticklabels():
        label.set_fontfamily(western)
        label.set_fontsize(7)
    handles1, labels1 = ax_fold.get_legend_handles_labels()
    handles2, labels2 = ax_brier.get_legend_handles_labels()
    ax_fold.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        prop={"family": western, "size": 7},
        loc="lower left",
    )

    figure06.suptitle("候选水体模型：空间交叉验证与校准", fontfamily=chinese, fontsize=10.5, y=0.965)
    figure06.text(
        0.5,
        0.875,
        "公开源五特征重建；非原 HPC 九特征模型",
        ha="center",
        va="center",
        fontfamily=chinese,
        fontsize=7.5,
    )
    rendered.extend(save_four_formats(figure06, "figure06_candidate_model_performance"))
    plt.close(figure06)

    domain, domain_metadata = read_raster(CANDIDATE / "downstream_basin_common_domain_30m.tif")
    domain_mask = domain == 1
    probability_maps: dict[str, np.ndarray] = {}
    connected_maps: dict[str, np.ndarray] = {}
    for phase, _, _ in PHASES:
        probability, metadata = read_raster(CANDIDATE / f"{phase}_candidate_water_probability_30m.tif")
        connected, _ = read_raster(CANDIDATE / f"{phase}_connected_water_t50_30m.tif")
        if metadata["crs"] != "EPSG:32610" or metadata["shape"] != domain_metadata["shape"]:
            raise RuntimeError(f"Candidate probability grid mismatch for {phase}")
        probability_maps[phase] = np.where(domain_mask & (probability >= 0), probability, np.nan)
        connected_maps[phase] = connected == 1

    figure07 = plt.figure(figsize=(WIDTH_IN, 2.65))
    grid07 = figure07.add_gridspec(
        1,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.055],
        left=0.012,
        right=0.965,
        bottom=0.06,
        top=0.76,
        wspace=0.035,
    )
    axes07 = [figure07.add_subplot(grid07[0, index]) for index in range(5)]
    colorbar_axis = figure07.add_subplot(grid07[0, 5])
    image = None
    for axis, (phase, label, date_text) in zip(axes07, PHASES):
        image = axis.imshow(
            probability_maps[phase],
            cmap="cividis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        if connected_maps[phase].any():
            axis.contour(connected_maps[phase].astype("uint8"), levels=[0.5], colors="#ffffff", linewidths=0.45)
        axis.contour(domain_mask.astype("uint8"), levels=[0.5], colors="#222222", linewidths=0.35)
        axis.set_title(f"{label}\n{date_text}", fontfamily=chinese, fontsize=8, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
    colorbar = figure07.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Candidate water probability", fontfamily=western, fontsize=8)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(western)
        label.set_fontsize(7)
    figure07.suptitle("五阶段候选水体概率与 0.50 连通边界", fontfamily=chinese, fontsize=10.5, y=0.97)
    figure07.text(
        0.5,
        0.855,
        "白线：候选连通水体边界；黑线：GRDC 下游流域范围",
        ha="center",
        fontfamily=chinese,
        fontsize=7.3,
    )
    rendered.extend(save_four_formats(figure07, "figure07_candidate_probability_connected_water"))
    plt.close(figure07)

    pivot = area.pivot(index="phase", columns="threshold", values="connected_area_km2").reindex([phase for phase, _, _ in PHASES])
    if pivot.isna().any().any() or list(pivot.columns) != THRESHOLDS:
        pivot = pivot.reindex(columns=THRESHOLDS)
    if pivot.isna().any().any():
        raise RuntimeError("Candidate connected-water area table is incomplete")

    figure09, ax09 = plt.subplots(figsize=(WIDTH_IN, 3.15))
    figure09.subplots_adjust(left=0.09, right=0.975, bottom=0.23, top=0.78)
    x = np.arange(len(PHASES))
    threshold_styles = {
        0.35: ("#6b7f99", 1.0, "o"),
        0.50: ("#2f6f8f", 1.8, "o"),
        0.65: ("#b56a3a", 1.0, "s"),
    }
    for threshold in THRESHOLDS:
        color, linewidth, marker = threshold_styles[threshold]
        values = pivot[threshold].to_numpy()
        ax09.plot(
            x,
            values,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=4,
            label=f"p ≥ {threshold:.2f}",
        )
        if threshold == 0.50:
            for position, value in zip(x, values):
                ax09.text(
                    position,
                    value + max(values) * 0.025,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontfamily=western,
                    fontsize=7,
                )
    ax09.set_xticks(x)
    ax09.set_xticklabels([label for _, label, _ in PHASES], fontfamily=chinese, fontsize=8)
    ax09.set_ylabel("Connected-water area (km²)", fontfamily=western, fontsize=9)
    ax09.set_title("候选连通水体面积及阈值敏感性", fontfamily=chinese, fontsize=10)
    ax09.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax09.legend(frameon=False, prop={"family": western, "size": 8}, ncol=3, loc="upper left")
    set_axis_fonts(ax09, western, chinese)
    for label in ax09.get_xticklabels():
        label.set_fontfamily(chinese)
        label.set_fontsize(8)
    figure09.suptitle("公开源候选重建；面积不得替代原 HPC 结果", fontfamily=chinese, fontsize=7.5, y=0.94)
    rendered.extend(save_four_formats(figure09, "figure09_candidate_connected_water_area"))
    plt.close(figure09)

    main_area = pivot[0.50]
    captions = pd.DataFrame(
        [
            {
                "figure": "Figure 6",
                "file_stem": "figure06_candidate_model_performance",
                "caption_zh": "候选水体模型的空间分块交叉验证、概率校准与折间稳定性。模型使用基线期 Sentinel-1 RTC 的 VV、VH、极化差值及 Copernicus GLO-30 高程和坡度共五项透明特征；标签来自 JRC Global Surface Water v1.5 候选永久水体与稳定陆地。所有指标均为公开源候选重建结果，不代表原 HPC 九特征模型。",
            },
            {
                "figure": "Figure 7",
                "file_stem": "figure07_candidate_probability_connected_water",
                "caption_zh": "五个洪水阶段的候选水体概率及 0.50 阈值连通水体边界。概率仅在 GRDC_4146080 官方下游流域与五期共同 RTC 数据域的交集内计算；连通组件以该域内 JRC 候选永久水体为种子。",
            },
            {
                "figure": "Figure 9",
                "file_stem": "figure09_candidate_connected_water_area",
                "caption_zh": "候选连通水体面积的五阶段变化及 0.35、0.50、0.65 概率阈值敏感性。面积由 30 m 栅格像元直接统计，属于公开源候选重建，不得与冻结 HPC 面积序列混用。",
            },
        ]
    )
    captions.to_csv(OUT / "candidate_figure_captions_zh.csv", index=False)

    qa = {
        "candidate_model_scope": status.get("scope"),
        "candidate_feature_count": status.get("feature_count"),
        "spatial_groups": status.get("spatial_groups"),
        "cv_folds": int(len(folds)),
        "oof_auc": float(status["oof_auc"]),
        "oof_brier": float(status["oof_brier"]),
        "main_threshold": 0.50,
        "main_threshold_area_km2": {phase: float(main_area.loc[phase]) for phase, _, _ in PHASES},
        "main_threshold_maximum_phase": str(main_area.idxmax()),
        "formal_export_allowed": bool(gate.get("formal_export_allowed")),
        "selected_western_family": western,
        "selected_chinese_family": chinese,
        "domain_crs": domain_metadata["crs"],
        "domain_shape": domain_metadata["shape"],
        "rendered_files": [path.name for path in rendered],
        "caption_file": "candidate_figure_captions_zh.csv",
        "scientific_caveats": status.get("scientific_caveats"),
        "prohibited_interpretation": status.get("prohibited_interpretation"),
    }
    qa["qa_passed"] = bool(
        qa["formal_export_allowed"]
        and qa["selected_western_family"] == "Times New Roman"
        and qa["selected_chinese_family"] in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}
        and qa["cv_folds"] == 5
        and qa["domain_crs"] == "EPSG:32610"
        and len(rendered) == 12
        and len(captions) == 3
    )
    (OUT / "candidate_model_figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = OUT / "sha256_manifest.csv"
    manifest_rows = []
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name != manifest_path.name:
            manifest_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    if not qa["qa_passed"]:
        raise RuntimeError("Candidate-model figure QA failed: " + json.dumps(qa, ensure_ascii=False))
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
