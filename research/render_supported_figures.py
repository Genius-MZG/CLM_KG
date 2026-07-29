from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
STATION_ROLE_LABELS = {
    "GRDC_4146080": "下游 · Near Mount Vernon",
    "GRDC_4146081": "上游 · Near Concrete",
}
BASIN_EDGE = ["#4477AA", "#CC6677"]
BASIN_FILL = ["#DDE8F2", "#F2E0E4"]
FLOW_COLORS = {"GRDC_4146080": "#4477AA", "GRDC_4146081": "#CC6677"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exterior_rings(geometry: dict):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon" and coords:
        yield coords[0]
    elif gtype == "MultiPolygon":
        for polygon in coords:
            if polygon:
                yield polygon[0]


def line_parts(geometry: dict):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "LineString":
        yield coords
    elif gtype == "MultiLineString":
        yield from coords


def apply_tick_font(axis, family: str) -> None:
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_fontfamily(family)
        label.set_fontsize(8)
    for offset_text in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
        offset_text.set_fontfamily(family)
        offset_text.set_fontsize(8)


def station_id_from_feature(feature: dict) -> str:
    try:
        return f"GRDC_{int(float(feature.get('properties', {}).get('grdc_no')))}"
    except (TypeError, ValueError):
        return "unknown"


def add_scale_and_north(axis, western_font: str) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    latitude = (ymin + ymax) / 2.0
    km_per_degree_lon = 111.32 * max(0.2, abs(__import__("math").cos(__import__("math").radians(latitude))))
    target_km = 20.0
    length_deg = target_km / km_per_degree_lon
    x0 = xmin + 0.07 * (xmax - xmin)
    y0 = ymin + 0.07 * (ymax - ymin)
    axis.plot([x0, x0 + length_deg], [y0, y0], color="#222222", linewidth=1.4, zorder=8)
    axis.plot([x0, x0], [y0 - 0.006 * (ymax - ymin), y0 + 0.006 * (ymax - ymin)], color="#222222", linewidth=0.8, zorder=8)
    axis.plot([x0 + length_deg, x0 + length_deg], [y0 - 0.006 * (ymax - ymin), y0 + 0.006 * (ymax - ymin)], color="#222222", linewidth=0.8, zorder=8)
    axis.text(x0 + length_deg / 2, y0 + 0.015 * (ymax - ymin), "20 km", ha="center", va="bottom", fontfamily=western_font, fontsize=7)
    axis.annotate("N", xy=(0.94, 0.91), xytext=(0.94, 0.80), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="bottom", fontfamily=western_font, fontsize=8, arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": "#222222"})


def render_map(basins: dict, stations: pd.DataFrame, rivers: dict | None, western_font: str, chinese_font: str) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.0866, 4.8))
    features = sorted(basins.get("features", []), key=station_id_from_feature)
    for index, feature in enumerate(features):
        station_id = station_id_from_feature(feature)
        first = True
        for ring in exterior_rings(feature.get("geometry", {})):
            if len(ring) < 4:
                continue
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            axis.fill(xs, ys, facecolor=BASIN_FILL[index % 2], edgecolor=BASIN_EDGE[index % 2], linewidth=0.9, alpha=0.55, label=STATION_ROLE_LABELS.get(station_id, station_id) + " basin" if first else None, zorder=1)
            first = False

    river_count = 0
    if rivers:
        for feature in rivers.get("features", []):
            for part in line_parts(feature.get("geometry", {})):
                if len(part) >= 2:
                    axis.plot([p[0] for p in part], [p[1] for p in part], color="#355C7D", linewidth=0.55, alpha=0.85, zorder=2)
                    river_count += 1
        if river_count:
            axis.plot([], [], color="#355C7D", linewidth=0.8, label="Official SWORD v17c reaches")

    lon_col = "long_pp" if "long_pp" in stations.columns else "long_org"
    lat_col = "lat_pp" if "lat_pp" in stations.columns else "lat_org"
    for _, row in stations.sort_values("grdc_no").iterrows():
        station_id = f"GRDC_{int(float(row['grdc_no']))}"
        color = FLOW_COLORS.get(station_id, "#333333")
        axis.scatter([row[lon_col]], [row[lat_col]], s=34, marker="o", color=color, edgecolor="white", linewidth=0.6, zorder=4)
        axis.annotate(
            STATION_ROLE_LABELS.get(station_id, station_id),
            (row[lon_col], row[lat_col]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontfamily=western_font,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
        )

    axis.set_xlabel("Longitude", fontfamily=western_font, fontsize=9)
    axis.set_ylabel("Latitude", fontfamily=western_font, fontsize=9)
    axis.set_title("流域范围、观测站与真实河道", fontfamily=chinese_font, fontsize=10)
    apply_tick_font(axis, western_font)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(False)
    add_scale_and_north(axis, western_font)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False, prop={"family": western_font, "size": 7}, loc="upper left")
    figure.tight_layout()
    outputs = []
    for extension, dpi in [("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)]:
        path = OUT / f"figure01_study_reach.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def render_hydrograph(flow: pd.DataFrame, western_font: str, chinese_font: str) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.0866, 4.8))
    for station, group in flow.groupby("station_id", sort=True):
        axis.plot(group["date"], group["discharge_m3s"], linewidth=1.15, color=FLOW_COLORS.get(station), label=STATION_ROLE_LABELS.get(station, station))
    label_y = {"baseline": 0.985, "rising": 0.925, "peak": 0.985, "early_recession": 0.925, "late_recession": 0.985}
    label_ha = {"baseline": "right", "rising": "right", "peak": "left", "early_recession": "right", "late_recession": "right"}
    for phase_name, date in PHASE_DATES.items():
        timestamp = pd.Timestamp(date)
        axis.axvline(timestamp, linewidth=0.7, linestyle="--", color="#666666")
        axis.text(timestamp, label_y[phase_name], PHASE_LABELS[phase_name], rotation=90, transform=axis.get_xaxis_transform(), ha=label_ha[phase_name], va="top", fontsize=7, fontfamily=chinese_font)
    axis.set_xlabel("Date", fontfamily=western_font, fontsize=9)
    axis.set_ylabel("Discharge (m³ s⁻¹)", fontfamily=western_font, fontsize=9)
    axis.set_title("双站水文过程与五期卫星观测", fontfamily=chinese_font, fontsize=10)
    axis.legend(frameon=False, prop={"family": western_font, "size": 8}, loc="upper left")
    apply_tick_font(axis, western_font)
    axis.margins(x=0.01)
    axis.set_ylim(bottom=0)
    figure.tight_layout()
    outputs = []
    for extension, dpi in [("svg", None), ("pdf", None), ("png", 300), ("tiff", 600)]:
        path = OUT / f"figure02_hydrographs.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
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

    sword_source = ROOT / "river_geometry" / "sword_NA_v17c_reaches.parquet"
    common_mask = ROOT / "planetary_computer_rtc" / "common_valid_mask_30m.tif"
    sword_output = ROOT / "river_geometry" / "sword_event_aoi_reaches.geojson"
    subset_returncode = None
    if sword_source.exists() and common_mask.exists():
        subset_returncode = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "subset_sword_river_geometry.py")], check=False).returncode

    gate = json.loads(required["font_gate"].read_text(encoding="utf-8"))
    unit_conversion = json.loads(required["unit_conversion"].read_text(encoding="utf-8"))
    flow = pd.read_csv(required["flow"], parse_dates=["date"])
    phase = pd.read_csv(required["phase"])
    stations = pd.read_csv(required["stations"])
    basins = json.loads(required["basins"].read_text(encoding="utf-8"))
    rivers = json.loads(sword_output.read_text(encoding="utf-8")) if sword_output.exists() else None

    qa = {
        "flow_station_count": int(flow["station_id"].nunique()),
        "flow_record_count": int(len(flow)),
        "phase_rows": int(len(phase)),
        "phase_missing_discharge": int(phase["discharge_m3s"].isna().sum()),
        "basin_feature_count": int(len(basins.get("features", []))),
        "station_metadata_rows": int(len(stations)),
        "formal_export_allowed": bool(gate.get("formal_export_allowed")),
        "selected_western_family": gate.get("selected_western_family"),
        "selected_chinese_family": gate.get("selected_chinese_family"),
        "unit_conversion_verified": unit_conversion.get("source_units", "").startswith("mm day-1"),
        "sword_subset_returncode": subset_returncode,
        "sword_geometry_ready": bool(rivers and rivers.get("features")),
        "sword_feature_count": 0 if not rivers else len(rivers.get("features", [])),
    }
    qa["data_qa_passed"] = bool(qa["flow_station_count"] == 2 and qa["phase_rows"] == 10 and qa["phase_missing_discharge"] == 0 and qa["basin_feature_count"] >= 2 and qa["station_metadata_rows"] >= 2 and qa["unit_conversion_verified"])

    rendered: list[Path] = []
    if qa["data_qa_passed"] and qa["formal_export_allowed"]:
        western = gate["selected_western_family"]
        chinese = gate["selected_chinese_family"]
        if western != "Times New Roman" or chinese not in {"KaiTi", "STKaiti", "AR PL KaitiM GB"}:
            raise RuntimeError("Font gate reported an unapproved family")
        rendered.extend(render_map(basins, stations, rivers, western, chinese))
        rendered.extend(render_hydrograph(flow, western, chinese))

    qa["formal_files_rendered"] = [path.name for path in rendered]
    qa["render_status"] = "formal_exports_complete" if rendered else "blocked"
    (OUT / "figure_qa_status.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(OUT.glob("*")) if path.is_file()]
    pd.DataFrame(manifest).to_csv(OUT / "sha256_manifest.csv", index=False)
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
