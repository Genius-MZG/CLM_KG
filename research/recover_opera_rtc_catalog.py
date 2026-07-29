from __future__ import annotations

import csv
import hashlib
import json
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "recovered_data" / "opera_rtc_catalog"
OUT.mkdir(parents=True, exist_ok=True)

SEARCH_ENDPOINT = "https://api.daac.asf.alaska.edu/services/search/param"
AOI_WKT = (
    "POLYGON((-122.48 48.34,-121.62 48.34,-121.62 48.66,"
    "-122.48 48.66,-122.48 48.34))"
)
EXPECTED_PLATFORM = "Sentinel-1B"
EXPECTED_TRACK = 13

PHASES = {
    "baseline": "2017-11-01",
    "rising": "2017-11-13",
    "peak": "2017-11-25",
    "early_recession": "2017-12-07",
    "late_recession": "2017-12-19",
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def first_nonempty(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def flatten_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            urls.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            urls.extend(flatten_urls(nested))
    elif isinstance(value, list):
        for nested in value:
            urls.extend(flatten_urls(nested))
    return sorted(set(urls))


def normalize_platform(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    aliases = {
        "S1B": "Sentinel-1B",
        "SENTINEL-1B": "Sentinel-1B",
        "Sentinel-1B": "Sentinel-1B",
    }
    return aliases.get(text, text)


def parse_track(properties: dict[str, Any], product_name: str | None) -> int | None:
    value = first_nonempty(
        properties,
        ["pathNumber", "relativeOrbit", "relativeOrbitNumber", "track", "trackNumber"],
    )
    if value is not None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    if product_name:
        match = re.search(r"_T(\d{3})-", product_name)
        if match:
            return int(match.group(1))
    return None


def parse_burst_id(properties: dict[str, Any], product_name: str | None) -> str | None:
    value = first_nonempty(
        properties,
        ["operaBurstID", "burstID", "burstId", "burst_id", "subswath"],
    )
    if value is not None:
        return str(value)
    if product_name:
        match = re.search(r"_(T\d{3}-\d+-IW[123])_", product_name)
        if match:
            return match.group(1)
    return None


def phase_window(date_text: str) -> tuple[str, str]:
    day = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = day - timedelta(hours=3)
    end = day + timedelta(days=1, hours=3)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def search_phase(phase: str, date_text: str) -> dict[str, Any]:
    start, end = phase_window(date_text)
    params = {
        "dataset": "OPERA-S1",
        "processingLevel": "RTC",
        "start": start,
        "end": end,
        "intersectsWith": AOI_WKT,
        "output": "geojson",
        "maxResults": 5000,
    }
    response = requests.get(
        SEARCH_ENDPOINT,
        params=params,
        timeout=300,
        headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"},
    )
    request_record = {
        "phase": phase,
        "date": date_text,
        "request_url": response.url,
        "status_code": response.status_code,
    }
    response.raise_for_status()
    payload = response.json()
    write_json(OUT / f"opera_rtc_{phase}_{date_text}.geojson", payload)
    request_record["feature_count"] = len(payload.get("features", []))
    return {"request": request_record, "payload": payload}


def feature_to_row(phase: str, phase_date: str, feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties", {}) or {}
    product_name = first_nonempty(
        properties,
        ["sceneName", "fileID", "productName", "granuleName", "fileName"],
    )
    if product_name is None:
        product_name = feature.get("id")
    product_name = None if product_name is None else str(product_name)

    start_time = first_nonempty(
        properties,
        ["startTime", "start", "sensingStart", "acquisitionDate", "beginPosition"],
    )
    stop_time = first_nonempty(
        properties,
        ["stopTime", "stop", "sensingStop", "endPosition"],
    )
    platform = normalize_platform(
        first_nonempty(properties, ["platform", "platformName", "sensor", "mission"])
    )
    track = parse_track(properties, product_name)
    burst_id = parse_burst_id(properties, product_name)
    urls = flatten_urls(properties)
    if feature.get("links"):
        urls.extend(flatten_urls(feature["links"]))
    urls = sorted(set(urls))

    layer_names = []
    for url in urls:
        base = url.rsplit("/", 1)[-1]
        for layer in ["VV", "VH", "mask", "local_incidence_angle", "incidence_angle"]:
            if re.search(rf"_{re.escape(layer)}\.(?:tif|tiff)$", base, flags=re.I):
                layer_names.append(layer)

    return {
        "phase": phase,
        "phase_date": phase_date,
        "feature_id": feature.get("id"),
        "product_name": product_name,
        "start_time": start_time,
        "stop_time": stop_time,
        "platform": platform,
        "track": track,
        "burst_id": burst_id,
        "processing_level": first_nonempty(
            properties, ["processingLevel", "productType", "fileType"]
        ),
        "polarization": first_nonempty(
            properties, ["polarization", "polarizations", "polarisationChannels"]
        ),
        "flight_direction": first_nonempty(
            properties, ["flightDirection", "orbitDirection", "direction"]
        ),
        "url_count": len(urls),
        "urls_json": json.dumps(urls, ensure_ascii=False),
        "detected_layers": "+".join(sorted(set(layer_names))),
        "property_keys": ";".join(sorted(properties.keys())),
        "expected_platform_match": platform == EXPECTED_PLATFORM if platform else None,
        "expected_track_match": track == EXPECTED_TRACK if track is not None else None,
    }


def make_manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    with (OUT / "sha256_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    status: dict[str, Any] = {
        "dataset": "OPERA-S1",
        "processing_level": "RTC",
        "aoi_wkt": AOI_WKT,
        "expected_platform": EXPECTED_PLATFORM,
        "expected_track": EXPECTED_TRACK,
        "phase_dates": PHASES,
        "requests": [],
        "phase_summary": {},
        "errors": [],
    }
    all_rows: list[dict[str, Any]] = []

    for phase, date_text in PHASES.items():
        try:
            result = search_phase(phase, date_text)
            status["requests"].append(result["request"])
            features = result["payload"].get("features", [])
            rows = [feature_to_row(phase, date_text, feature) for feature in features]
            all_rows.extend(rows)

            expected = [
                row for row in rows
                if row["expected_platform_match"] is not False
                and row["expected_track_match"] is not False
            ]
            status["phase_summary"][phase] = {
                "date": date_text,
                "all_features": len(rows),
                "expected_platform_track_candidates": len(expected),
                "unique_burst_ids_all": len({r["burst_id"] for r in rows if r["burst_id"]}),
                "unique_burst_ids_expected": len(
                    {r["burst_id"] for r in expected if r["burst_id"]}
                ),
                "products_with_vv_url": sum(
                    "VV" in (r["detected_layers"] or "").split("+") for r in expected
                ),
                "products_with_vh_url": sum(
                    "VH" in (r["detected_layers"] or "").split("+") for r in expected
                ),
                "products_with_mask_url": sum(
                    "mask" in (r["detected_layers"] or "").split("+") for r in expected
                ),
            }
        except Exception as exc:
            status["errors"].append(
                {
                    "phase": phase,
                    "date": date_text,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    table = pd.DataFrame(all_rows)
    if not table.empty:
        table.to_csv(OUT / "opera_rtc_all_candidates.csv", index=False)
        expected_table = table[
            table["expected_platform_match"].fillna(True)
            & table["expected_track_match"].fillna(True)
        ].copy()
        expected_table.to_csv(OUT / "opera_rtc_expected_track_candidates.csv", index=False)

    status["completed_phases"] = sum(
        1
        for phase in PHASES
        if status["phase_summary"].get(phase, {}).get("all_features", 0) > 0
    )
    status["catalogue_complete"] = status["completed_phases"] == len(PHASES)
    status["download_ready"] = all(
        status["phase_summary"].get(phase, {}).get("unique_burst_ids_expected", 0) > 0
        for phase in PHASES
    )
    status["interpretation"] = (
        "Catalogue discovery only. No RTC raster has been downloaded, mosaicked, "
        "resampled, classified or plotted by this stage."
    )
    write_json(OUT / "opera_rtc_catalog_status.json", status)
    make_manifest()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
