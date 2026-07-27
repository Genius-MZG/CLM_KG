from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from remotezip import RemoteZip

OUT = Path(__file__).resolve().parent / "recovered_data"
OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ["GRDC_4146081", "GRDC_4146080"]
STATION_NUMBERS = [4146081, 4146080]
ZENODO_RECORD = "15349031"
ZENODO_ZIP_NAME = "GRDC_Caravan_extension_csv.zip"
PEAK_DATE = pd.Timestamp("2017-11-23")
SEARCH_START = "2017-08-01T00:00:00Z"
SEARCH_END = "2018-03-01T00:00:00Z"

status: dict[str, Any] = {
    "stations": STATIONS,
    "zenodo_record": ZENODO_RECORD,
    "steps": {},
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 120) -> Any:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
    r.raise_for_status()
    return r.json()


def save_response_json(path: Path, response: requests.Response) -> None:
    try:
        payload = response.json()
    except Exception:
        payload = {"status_code": response.status_code, "text": response.text[:20000]}
    write_json(path, payload)


def locate_extracted_file(root: Path, basename: str) -> Path | None:
    hits = list(root.rglob(basename))
    return hits[0] if hits else None


def recover_zenodo_members() -> dict[str, Any]:
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    meta = get_json(api_url)
    write_json(OUT / "zenodo_record_metadata.json", meta)

    content_url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZENODO_ZIP_NAME}/content"
    extracted = OUT / "zenodo_extracted"
    extracted.mkdir(exist_ok=True)

    exact_members = {
        f"GRDC_Caravan_extension_csv/timeseries/csv/grdc/{station}.csv"
        for station in STATIONS
    }
    suffixes = {
        "attributes_caravan_grdc.csv",
        "attributes_hydroatlas_grdc.csv",
        "attributes_other_grdc.csv",
        "attributes_grdc.csv",
    }

    with RemoteZip(content_url, initial_buffer_size=2 * 1024 * 1024) as rz:
        names = [z.filename for z in rz.infolist()]
        matching = [
            n for n in names
            if n in exact_members
            or Path(n).name in suffixes
            or any(station in Path(n).name for station in STATIONS)
        ]
        (OUT / "zenodo_archive_matching_members.txt").write_text("\n".join(matching) + "\n", encoding="utf-8")
        for member in matching:
            if member.endswith("/"):
                continue
            try:
                rz.extract(member, path=extracted)
            except Exception as exc:
                with (OUT / "zenodo_extract_errors.txt").open("a", encoding="utf-8") as f:
                    f.write(f"{member}\t{type(exc).__name__}: {exc}\n")

    found = {}
    for station in STATIONS:
        p = locate_extracted_file(extracted, f"{station}.csv")
        found[station] = str(p) if p else None
    return {"content_url": content_url, "matching_members": matching, "station_files": found}


def select_column(columns: list[str], candidates: list[str], contains: list[str] | None = None) -> str | None:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if contains:
        for original in columns:
            lo = original.lower()
            if all(token in lo for token in contains):
                return original
    return None


def process_station_timeseries(zenodo_info: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    merged = []
    for station in STATIONS:
        file_path = zenodo_info["station_files"].get(station)
        if not file_path:
            summaries[station] = {"error": "station CSV was not extracted"}
            continue
        path = Path(file_path)
        df = pd.read_csv(path)
        date_col = select_column(list(df.columns), ["date", "DATE", "datetime", "time"])
        q_col = select_column(
            list(df.columns),
            ["observed_discharge_cms", "observed_discharge_m3s", "discharge", "streamflow"],
            contains=["observed", "discharge"],
        )
        if date_col is None or q_col is None:
            summaries[station] = {"error": "date/discharge columns not found", "columns": list(df.columns)}
            continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[q_col] = pd.to_numeric(df[q_col], errors="coerce")
        window = df[(df[date_col] >= "2017-08-01") & (df[date_col] <= "2018-03-01")][[date_col, q_col]].copy()
        window.columns = ["date", "discharge_m3s"]
        window.insert(0, "station_id", station)
        window.to_csv(OUT / f"{station}_20170801_20180301.csv", index=False)
        merged.append(window)

        event = window[(window["date"] >= "2017-10-01") & (window["date"] <= "2018-01-31")]
        valid = event.dropna(subset=["discharge_m3s"])
        peak = None
        if not valid.empty:
            row = valid.loc[valid["discharge_m3s"].idxmax()]
            peak = {"date": row["date"].date().isoformat(), "discharge_m3s": float(row["discharge_m3s"])}
        q_on_known_peak = event.loc[event["date"] == PEAK_DATE, "discharge_m3s"]
        summaries[station] = {
            "source_file": str(path),
            "columns": list(df.columns),
            "date_column": date_col,
            "discharge_column": q_col,
            "records_total": int(len(df)),
            "window_records": int(len(window)),
            "window_valid_discharge": int(window["discharge_m3s"].notna().sum()),
            "event_window_peak": peak,
            "discharge_on_2017_11_23_m3s": None if q_on_known_peak.empty or pd.isna(q_on_known_peak.iloc[0]) else float(q_on_known_peak.iloc[0]),
        }
    if merged:
        pd.concat(merged, ignore_index=True).to_csv(OUT / "grdc_pair_event_window_long.csv", index=False)
    write_json(OUT / "grdc_timeseries_summary.json", summaries)
    return summaries


def query_grdc_arcgis() -> dict[str, Any]:
    url = "https://geoportal.bafg.de/arcgis3/rest/services/GRDC/GRDC_basins_changeid/FeatureServer/0/query"
    params = {
        "where": "grdc_no IN (4146081,4146080)",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    r = requests.get(url, params=params, timeout=180, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
    r.raise_for_status()
    data = r.json()
    write_json(OUT / "grdc_pair_basins_arcgis.geojson", data)
    rows = []
    coords = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        rows.append(props)
        num = int(float(props.get("grdc_no"))) if props.get("grdc_no") is not None else None
        lon = props.get("long_pp", props.get("long_org"))
        lat = props.get("lat_pp", props.get("lat_org"))
        if num and lon is not None and lat is not None:
            coords[f"GRDC_{num}"] = [float(lon), float(lat)]
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "grdc_pair_station_metadata.csv", index=False)
    return {"feature_count": len(data.get("features", [])), "coordinates": coords, "properties": rows}


def extract_attributes() -> dict[str, Any]:
    extracted = OUT / "zenodo_extracted"
    results = {}
    wanted = {s.lower() for s in STATIONS}
    for path in extracted.rglob("attributes*_grdc.csv"):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            results[str(path)] = {"error": str(exc)}
            continue
        id_col = select_column(list(df.columns), ["gauge_id", "station_id", "grdc_no", "grdc_id", "id"])
        if id_col is None:
            results[str(path)] = {"columns": list(df.columns), "matching_rows": 0}
            continue
        values = df[id_col].astype(str).str.lower()
        mask = values.isin(wanted) | values.str.replace("grdc_", "", regex=False).isin({"4146081", "4146080"})
        subset = df[mask].copy()
        out_name = "selected_" + path.name
        subset.to_csv(OUT / out_name, index=False)
        results[str(path)] = {"id_column": id_col, "matching_rows": int(len(subset)), "output": out_name}
    write_json(OUT / "attribute_extraction_summary.json", results)
    return results


def asf_search(coords: dict[str, list[float]]) -> dict[str, Any]:
    endpoint = "https://api.daac.asf.alaska.edu/services/search/param"
    all_features = {}
    request_log = []
    for station, (lon, lat) in coords.items():
        params = {
            "platform": "Sentinel-1",
            "beamMode": "IW",
            "start": SEARCH_START,
            "end": SEARCH_END,
            "intersectsWith": f"POINT({lon} {lat})",
            "output": "geojson",
            "maxResults": 5000,
        }
        r = requests.get(endpoint, params=params, timeout=240, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
        request_log.append({"station": station, "url": r.url, "status": r.status_code})
        r.raise_for_status()
        payload = r.json()
        write_json(OUT / f"asf_search_{station}.geojson", payload)
        for f in payload.get("features", []):
            p = f.get("properties", {})
            key = p.get("sceneName") or p.get("fileID") or p.get("productName") or f.get("id")
            if key:
                all_features[key] = f

    combined = {"type": "FeatureCollection", "features": list(all_features.values())}
    write_json(OUT / "asf_sentinel1_candidates_combined.geojson", combined)
    rows = []
    for f in combined["features"]:
        p = f.get("properties", {})
        rows.append({
            "id": f.get("id"),
            "scene_name": p.get("sceneName") or p.get("fileID") or p.get("productName"),
            "start_time": p.get("startTime") or p.get("start") or p.get("sensingStart"),
            "stop_time": p.get("stopTime") or p.get("stop") or p.get("sensingStop"),
            "platform": p.get("platform"),
            "flight_direction": p.get("flightDirection"),
            "path_number": p.get("pathNumber"),
            "frame_number": p.get("frameNumber"),
            "polarization": p.get("polarization"),
            "processing_level": p.get("processingLevel"),
            "beam_mode": p.get("beamModeType") or p.get("beamMode"),
            "url": p.get("url") or p.get("downloadUrl"),
        })
    table = pd.DataFrame(rows)
    if not table.empty and "start_time" in table:
        table["start_time_parsed"] = pd.to_datetime(table["start_time"], errors="coerce", utc=True)
        table = table.sort_values("start_time_parsed")
    table.to_csv(OUT / "asf_sentinel1_candidates.csv", index=False)
    write_json(OUT / "asf_search_requests.json", request_log)
    return {"unique_features": len(all_features), "requests": request_log}


def make_manifest() -> None:
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "sha256_manifest.csv":
            rows.append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    with (OUT / "sha256_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)


def run_step(name: str, fn):
    try:
        result = fn()
        status["steps"][name] = {"ok": True, "result": result}
        return result
    except Exception as exc:
        status["steps"][name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return None


def main() -> None:
    zenodo = run_step("zenodo_remote_zip_extraction", recover_zenodo_members)
    if zenodo:
        run_step("station_timeseries", lambda: process_station_timeseries(zenodo))
        run_step("station_attributes", extract_attributes)
    arcgis = run_step("grdc_arcgis", query_grdc_arcgis)
    coords = arcgis.get("coordinates", {}) if arcgis else {}
    if coords:
        run_step("asf_sentinel1_catalogue", lambda: asf_search(coords))
    else:
        status["steps"]["asf_sentinel1_catalogue"] = {"ok": False, "error": "No station coordinates available"}
    write_json(OUT / "recovery_status.json", status)
    make_manifest()
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
