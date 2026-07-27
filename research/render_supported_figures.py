from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent / "recovered_data"
READY = ROOT / "figure_ready"
OUT = ROOT / "figure_outputs"
OUT.mkdir(parents=True, exist_ok=True)

PHASE_DATES = {
    "baseline": "2017-11-01",
    "rising": "2017-11-13",
    "peak": "2017-11-25",
    "early_recession": "2017-12-07",
    "late_recession": "2017-12-19",
}
PHASE_LABELS = {
    "baseline": "基线",
    "rising": "上涨期",
    "peak": "峰值期",
    "early_recession": "早退水期",
    "late_recession": "晚退水期",
}
STATION_LABELS = {
    "GRDC_4146080": "GRDC_4146080 · Near Mount Vernon",
    "GRDC_4146081": "GRDC_4146081 · Near Concrete",
}
BASIN_EDGE = ["#4477AA", "#CC6677"]
BASIN_FILL = ["#DDE8F2", "#F2E0E4"]
FLOW_COLORS = {"GRDC_4146080": "#4477AA", "GRDC_4146081": "#CC6677"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exterior_rings(geometry: dict):
    """Yield exterior polygon rings only; interior holes must not be plotted as point-like artefacts."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon" and coords:
        yield coords[0]
    elif gtype == "MultiPolygon":
        for polygon in coords:
            if polygon:
                yield polygon[0]


def apply_tick_font(ax, family: str) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily(family)
        label.set_fontsize(8)


def station_id_from_feature(feature: dict) -> str:
    props = feature.get("properties", {})
    value = props.get("grdc_no")
    try:
        return f"GRDC_{int(float(value))}"
    except (TypeError, ValueError):
        return "unknown"


def render_map(basins: dict, stations: pd.DataFrame, western_font: str, chinese_font: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0866, 4.8))
    features = sorted(basins.get("features", []), key=station_id_from_feature)
    for i, feature in enumerate(features):
        station_id = station_id_from_feature(feature)
        first = True
        for ring in exterior_rings(feature.get("geometry", {})):
            if len(ring) < 4:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            ax.fill(
                xs,
                ys,
                facecolor=BASIN_FILL[i % len(BASIN_FILL)],
                edgecolor=BASIN_EDGE[i % len(BASIN_EDGE)],
                linewidth=0.9,
                alpha=0.65,
                label=STATION_LABELS.get(station_id, station_id) + " basin" if first else None,
                zorder=1,
            )
            first = False

    lon_col = "long_pp" if "long_pp" in stations.columns else "long_org"
    lat_col = "lat_pp" if "lat_pp" in stations.columns else "lat_org"
    for _, row in stations.sort_values("grdc_no").iterrows():
        station_id = f"GRDC_{int(float(row['grdc_no']))}"
        color = FLOW_COLORS.get(station_id, "#333333")
        ax.scatter([row[lon_col]], [row[lat_col]], s=34, marker="o", color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(
            station_id,
            (row[lon_col], row[lat_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontfamily=western_font,
        )

    ax.set_xlabel("Longitude", fontfamily=western_font, fontsize=9)
    ax.set_ylabel("Latitude", fontfamily=western_font, fontsize=9)
    ax.set_title("流域范围与观测站", fontfamily=chinese_font, fontsize=10)
    apply_tick_font(ax, western_font)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, prop={"family": western_font, "size": 7}, loc="upper left")
    fig.tight_layout()
    outputs = []
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)]:
        path = OUT / f"figure01_study_reach.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def render_hydrograph(flow: pd.DataFrame, phase: pd.DataFrame, western_font: str, chinese_font: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0866, 4.8))
    for station, group in flow.groupby("station_id", sort=True):
        ax.plot(
            group["date"],
            group["discharge_m3s"],
            linewidth=1.15,
            color=FLOW_COLORS.get(station),
            label=STATION_LABELS.get(station, station),
        )
    for phase_name, date in PHASE_DATES.items():
        ax.axvline(pd.Timestamp(date), linewidth=0.7, linestyle="--", color="#666666")
        ax.text(
            pd.Timestamp(date),
            0.98,
            PHASE_LABELS[phase_name],
            rotation=90,
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=7,
            fontfamily=chinese_font,
        )
    ax.set_xlabel("Date", fontfamily=western_font, fontsize=9)
    ax.set_ylabel(r"Discharge (m$^3$ s$^{-1}$)", fontfamily=western_font, fontsize=9)
    ax.set_title("双站水文过程与五期卫星观测", fontfamily=chinese_font, fontsize=10)
    ax.legend(frameon=False, prop={"family": western_font, "size": 8})
    apply_tick_font(ax, western_font)
    ax.margins(x=0.01)
    fig.tight_layout()
    outputs = []
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)]:
        path = OUT / f"figure02_hydrographs.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    required = {
        "font_gate": READY / "font_gate.json",
        "unit_conversion": READY / "streamflow_unit_conversion.json",
        "flow": READY / "figure02_hydrograph_long.csv",
        "phase": READY / "figure02_phase_discharge.csv",
        "stations": ROOT / "grdc_pair_station_metadata.csv",
        "basins": ROOT / "grdc_pair_basins_arcgis.geojson",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing render inputs: " + ", ".join(missing))

    gate = json.loads(required["font_gate"].read_text(encoding="utf-8"))
    unit_conversion = json.loads(required["unit_conversion"].read_text(encoding="utf-8"))
    flow = pd.read_csv(required["flow"], parse_dates=["date"])
    phase = pd.read_csv(required["phase"])
    stations = pd.read_csv(required["stations"])
    basins = json.loads(required["basins"].read_text(encoding="utf-8"))

    qa = {
        "flow_station_count": int(flow["station_id"].nunique()),
        "flow_record_count": int(len(flow)),
        "flow_missing_discharge": int(flow["discharge_m3s"].isna().sum()),
        "flow_missing_source_runoff": int(flow["streamflow_mm_day"].isna().sum()),
        "phase_rows": int(len(phase)),
        "phase_missing_discharge": int(phase["discharge_m3s"].isna().sum()),
        "basin_feature_count": int(len(basins.get("features", []))),
        "station_metadata_rows": int(len(stations)),
        "formal_export_allowed": bool(gate.get("formal_export_allowed")),
        "selected_western_family": gate.get("selected_western_family"),
        "selected_chinese_family": gate.get("selected_chinese_family"),
        "unit_conversion_verified": unit_conversion.get("source_units", "").startswith("mm day-1"),
        "peak_discharge_range_m3s": [float(flow["discharge_m3s"].min()), float(flow["discharge_m3s"].max())],
    }
    qa["data_qa_passed"] = bool(
        qa["flow_station_count"] == 2
        and qa["phase_rows"] == 10
        and qa["phase_missing_discharge"] == 0
        and qa["flow_missing_source_runoff"] == 0
        and qa["basin_feature_count"] >= 2
        and qa["station_metadata_rows"] >= 2
        and qa["unit_conversion_verified"]
        and qa["peak_discharge_range_m3s"][1] > 100.0
    )

    specifications = {
        "figure01": {
            "title": "流域范围与观测站",
            "inputs": ["grdc_pair_basins_arcgis.geojson", "grdc_pair_station_metadata.csv"],
            "width_mm": 180,
            "formal_formats": ["svg", "pdf", "png_300dpi", "tiff_600dpi"],
        },
        "figure02": {
            "title": "双站水文过程与五期卫星观测",
            "inputs": ["figure02_hydrograph_long.csv", "figure02_phase_discharge.csv", "streamflow_unit_conversion.json"],
            "width_mm": 180,
            "formal_formats": ["svg", "pdf", "png_300dpi", "tiff_600dpi"],
        },
    }
    (OUT / "figure_specs.json").write_text(json.dumps(specifications, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rendered: list[Path] = []
    if qa["data_qa_passed"] and qa["formal_export_allowed"]:
        western = gate["selected_western_family"]
        chinese = gate["selected_chinese_family"]
        if western != "Times New Roman" or chinese not in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}:
            raise RuntimeError("Font gate reported an unapproved family")
        rendered.extend(render_map(basins, stations, western, chinese))
        rendered.extend(render_hydrograph(flow, phase, western, chinese))

    qa["formal_files_rendered"] = [p.name for p in rendered]
    qa["render_status"] = (
        "formal_exports_complete" if rendered else
        "font_gate_blocked" if not qa["formal_export_allowed"] else
        "data_qa_failed"
    )
    (OUT / "figure_qa_status.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = []
    for path in sorted(OUT.glob("*")):
        if path.is_file():
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(OUT / "sha256_manifest.csv", index=False)
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
