from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "figure_ready"
OUT.mkdir(parents=True, exist_ok=True)

PHASE_DATES = {
    "baseline": "2017-11-01",
    "rising": "2017-11-13",
    "peak": "2017-11-25",
    "early_recession": "2017-12-07",
    "late_recession": "2017-12-19",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fc_match(name: str) -> str:
    try:
        return subprocess.check_output(["fc-match", "-f", "%{family}|%{file}\n", name], text=True).strip()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"


def main() -> None:
    flow_path = ROOT / "grdc_pair_event_window_long.csv"
    meta_path = ROOT / "grdc_pair_station_metadata.csv"
    basin_path = ROOT / "grdc_pair_basins_arcgis.geojson"
    rtc_path = ROOT / "opera_rtc_catalog" / "opera_rtc_expected_track_candidates.csv"

    required = [flow_path, meta_path, basin_path, rtc_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required recovered inputs: " + ", ".join(missing))

    flow = pd.read_csv(flow_path, parse_dates=["date"])
    flow = flow.sort_values(["station_id", "date"]).reset_index(drop=True)
    phase_rows = []
    for phase, date in PHASE_DATES.items():
        day = pd.Timestamp(date)
        for station, group in flow.groupby("station_id"):
            hit = group[group["date"] == day]
            phase_rows.append({
                "phase": phase,
                "date": date,
                "station_id": station,
                "discharge_m3s": None if hit.empty else float(hit.iloc[0]["discharge_m3s"]),
            })
    pd.DataFrame(phase_rows).to_csv(OUT / "figure02_phase_discharge.csv", index=False)

    peak_rows = []
    for station, group in flow.groupby("station_id"):
        valid = group.dropna(subset=["discharge_m3s"])
        row = valid.loc[valid["discharge_m3s"].idxmax()]
        peak_rows.append({
            "station_id": station,
            "peak_date": row["date"].date().isoformat(),
            "peak_discharge_m3s": float(row["discharge_m3s"]),
        })
    pd.DataFrame(peak_rows).to_csv(OUT / "figure02_station_peaks.csv", index=False)
    flow.to_csv(OUT / "figure02_hydrograph_long.csv", index=False)

    rtc = pd.read_csv(rtc_path)
    cols = [c for c in ["phase", "phase_date", "product_name", "burst_id", "platform", "track", "flight_direction", "urls_json"] if c in rtc.columns]
    rtc[cols].to_csv(OUT / "figure03_rtc_product_binding.csv", index=False)

    font_report = {
        "required_western_font": "Times New Roman",
        "required_chinese_font": "KaiTi/STKaiti/AR PL KaitiM GB",
        "times_new_roman_match": fc_match("Times New Roman"),
        "kaiti_match": fc_match("AR PL KaitiM GB"),
    }
    font_report["times_new_roman_exact"] = font_report["times_new_roman_match"].lower().startswith("times new roman|")
    font_report["kaiti_available"] = "kaiti" in font_report["kaiti_match"].lower() or "kai" in font_report["kaiti_match"].lower()
    font_report["formal_export_allowed"] = bool(font_report["times_new_roman_exact"] and font_report["kaiti_available"])
    (OUT / "font_gate.json").write_text(json.dumps(font_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = []
    for path in sorted(OUT.glob("*")):
        if path.is_file():
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(OUT / "sha256_manifest.csv", index=False)

    status = {
        "input_validation_complete": True,
        "figure02_hydrograph_ready": True,
        "figure01_geometry_ready": basin_path.exists() and meta_path.exists(),
        "figure03_catalogue_binding_ready": not rtc.empty,
        "formal_export_allowed": font_report["formal_export_allowed"],
        "formal_export_blocker": None if font_report["formal_export_allowed"] else "Exact Times New Roman and approved KaiTi font are required before formal export.",
        "missing_raster_blocker": "Scientific RTC raster layers remain inaccessible without Earthdata credentials or original HPC outputs.",
    }
    (OUT / "figure_preparation_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
