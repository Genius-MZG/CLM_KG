from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = Path(__file__).resolve().parent / "phase_binding"
OUT.mkdir(parents=True, exist_ok=True)
PEAK = pd.Timestamp("2017-11-23", tz="UTC")
STATIONS = {"GRDC_4146081": (-121.77083, 48.52417), "GRDC_4146080": (-122.335, 48.445)}
PHASES = ["baseline", "rising", "peak", "early_recession", "late_recession"]

up = pd.read_csv(ROOT / "GRDC_4146081_20170801_20180301.csv", parse_dates=["date"]).set_index("date")["discharge_m3s"]
down = pd.read_csv(ROOT / "GRDC_4146080_20170801_20180301.csv", parse_dates=["date"]).set_index("date")["discharge_m3s"]
geojson = json.loads((ROOT / "asf_sentinel1_candidates_combined.geojson").read_text(encoding="utf-8"))
points = {key: Point(*coordinates) for key, coordinates in STATIONS.items()}

records = []
for feature in geojson["features"]:
    properties = feature.get("properties", {})
    geometry = shape(feature["geometry"]) if feature.get("geometry") else None
    records.append(
        {
            "scene_name": properties.get("sceneName") or properties.get("fileID") or properties.get("productName"),
            "start_time": properties.get("startTime") or properties.get("start") or properties.get("sensingStart"),
            "stop_time": properties.get("stopTime") or properties.get("stop") or properties.get("sensingStop"),
            "platform": properties.get("platform"),
            "flight_direction": properties.get("flightDirection"),
            "path_number": properties.get("pathNumber"),
            "frame_number": properties.get("frameNumber"),
            "polarization": properties.get("polarization"),
            "processing_level": properties.get("processingLevel"),
            "beam_mode": properties.get("beamModeType") or properties.get("beamMode"),
            "url": properties.get("url") or properties.get("downloadUrl"),
            "covers_upstream": bool(geometry.covers(points["GRDC_4146081"])) if geometry else False,
            "covers_downstream": bool(geometry.covers(points["GRDC_4146080"])) if geometry else False,
        }
    )

all_scenes = pd.DataFrame(records).drop_duplicates("scene_name")
all_scenes["dt"] = pd.to_datetime(all_scenes["start_time"], errors="coerce", utc=True)
eligible = all_scenes[
    (all_scenes.processing_level == "GRD_HD")
    & (all_scenes.polarization == "VV+VH")
    & (all_scenes.beam_mode == "IW")
    & all_scenes.covers_upstream
    & all_scenes.covers_downstream
].copy()
eligible = eligible[(eligible.dt >= PEAK - pd.Timedelta(days=40)) & (eligible.dt <= PEAK + pd.Timedelta(days=40))].sort_values("dt")

sequences = []
for (direction, path, frame), group in eligible.groupby(["flight_direction", "path_number", "frame_number"], dropna=False):
    group = group.sort_values("dt").reset_index(drop=True)
    for index in range(max(0, len(group) - 4)):
        window = group.iloc[index : index + 5]
        if len(window) != 5:
            continue
        dates = list(window.dt)
        intervals = np.diff([date.value for date in dates]) / 86400e9
        score = float(np.mean(np.abs(intervals - 12.0)))
        if not (
            dates[0] < PEAK - pd.Timedelta(days=10)
            and dates[1] < PEAK
            and PEAK - pd.Timedelta(days=3) <= dates[2] <= PEAK + pd.Timedelta(days=5)
        ):
            score += 100
        sequences.append({"direction": direction, "path_number": path, "frame_number": frame, "score": score, "rows": window})

sequences.sort(key=lambda item: item["score"])
if not sequences or sequences[0]["score"] >= 100:
    raise RuntimeError("No valid five-scene same-track sequence brackets the verified peak")

best = sequences[0]
selected = best["rows"].copy().reset_index(drop=True)
selected.insert(0, "phase", PHASES)
selected["offset_days_from_upstream_peak"] = (selected.dt - PEAK).dt.total_seconds() / 86400


def q_at(series: pd.Series, timestamp: pd.Timestamp) -> float:
    day = timestamp.tz_convert(None).normalize()
    return float(series.loc[day]) if day in series.index else float("nan")


selected["upstream_discharge_m3s"] = [q_at(up, timestamp) for timestamp in selected.dt]
selected["downstream_discharge_m3s"] = [q_at(down, timestamp) for timestamp in selected.dt]
selected["binding_status"] = "catalogue-derived reconstruction; requires comparison with original HPC product-plan JSONL"

columns = [
    "phase",
    "scene_name",
    "start_time",
    "stop_time",
    "platform",
    "flight_direction",
    "path_number",
    "frame_number",
    "polarization",
    "processing_level",
    "beam_mode",
    "offset_days_from_upstream_peak",
    "upstream_discharge_m3s",
    "downstream_discharge_m3s",
    "covers_upstream",
    "covers_downstream",
    "url",
    "binding_status",
]
selected[columns].to_csv(OUT / "sentinel1_five_phase_candidate_binding.csv", index=False)

# Pair every selected GRD date with available OPERA RTC burst products on the same path.
rtc = all_scenes[(all_scenes.processing_level == "RTC") & (all_scenes.path_number == best["path_number"])].copy()
rtc["date"] = rtc.dt.dt.strftime("%Y-%m-%d")
selected_dates = set(selected.dt.dt.strftime("%Y-%m-%d"))
rtc = rtc[rtc.date.isin(selected_dates)].sort_values(["date", "scene_name"])
rtc.to_csv(OUT / "opera_rtc_products_for_selected_dates.csv", index=False)

report = {
    "event_id": "pair_event_candidate_b257c320fb40f164e685a6ec",
    "pair_id": "pair_05dd029bcf4d966afa3b7d5f",
    "verified_upstream_peak_date": "2017-11-23",
    "catalogue_records_total": int(len(all_scenes)),
    "eligible_dual_pol_grd_covering_both_stations_in_80d_window": int(len(eligible)),
    "selected_track": {
        "flight_direction": best["direction"],
        "relative_orbit_path": int(best["path_number"]),
        "frame_number": int(best["frame_number"]),
        "selection_score": best["score"],
    },
    "selected_grd_scenes": selected[columns].replace({np.nan: None}).to_dict("records"),
    "opera_rtc_product_count_for_selected_dates": int(len(rtc)),
    "limitation": "The binding is reproducible from public catalogue metadata but is not yet byte-for-byte matched to the inaccessible HPC product-plan JSONL.",
}
(OUT / "phase_binding_qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

with (OUT / "sha256_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["path", "bytes", "sha256"])
    for path in sorted(OUT.glob("*")):
        if path.name == "sha256_manifest.csv" or not path.is_file():
            continue
        writer.writerow([path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()])

print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
