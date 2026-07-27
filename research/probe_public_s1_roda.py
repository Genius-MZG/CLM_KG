from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "source_access_probe"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_DATES = ["2017-11-01", "2017-11-13", "2017-11-25", "2017-12-07", "2017-12-19"]
RODA_BUCKET = "https://roda.sentinel-hub.com/sentinel-s1-l1c"
RANGE_BYTES = 65536
TIMEOUT = (10, 25)
MAX_WORKERS = 8


def select_scenes() -> list[dict[str, str]]:
    df = pd.read_csv(ROOT / "asf_sentinel1_candidates.csv")
    rows: list[dict[str, str]] = []
    for date in TARGET_DATES:
        daily = df[df["start_time"].astype(str).str.startswith(date)]
        grd = daily[
            (daily["processing_level"] == "GRD_HD")
            & (daily["scene_name"].str.contains("_GRDH_1SDV_", na=False))
        ]
        if grd.empty:
            rows.append({"date": date, "error": "GRD scene not found"})
            continue
        scene = str(grd.iloc[0]["scene_name"])
        y, m, d = date.split("-")
        prefix = f"GRD/{y}/{int(m)}/{int(d)}/IW/DV/{scene}/"
        rows.append({"date": date, "scene_name": scene, "prefix": prefix})
    return rows


def list_prefix(prefix: str) -> dict[str, object]:
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    try:
        r = requests.get(RODA_BUCKET, params=params, timeout=TIMEOUT, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
        keys: list[str] = []
        if r.status_code == 200 and r.content:
            try:
                root = ET.fromstring(r.content)
                keys = [node.text for node in root.iter() if node.tag.endswith("Key") and node.text]
            except ET.ParseError:
                pass
        return {
            "status_code": r.status_code,
            "request_url": r.url,
            "content_type": r.headers.get("content-type"),
            "bytes_received": len(r.content),
            "keys": keys,
            "prefix_hex": r.content[:16].hex(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "keys": []}


def probe_tiff(url: str) -> dict[str, object]:
    headers = {
        "Range": f"bytes=0-{RANGE_BYTES - 1}",
        "Accept-Encoding": "identity",
        "User-Agent": "NorthAmericaFloodplainResearch/1.0",
    }
    try:
        with requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True, stream=True) as r:
            body = bytearray()
            for chunk in r.iter_content(16384):
                if not chunk:
                    continue
                body.extend(chunk[: RANGE_BYTES - len(body)])
                if len(body) >= RANGE_BYTES:
                    break
            payload = bytes(body)
            is_tiff = payload.startswith((b"II*\x00", b"MM\x00*"))
            is_html = payload.lstrip().lower().startswith((b"<!doctype html", b"<html")) or "text/html" in (r.headers.get("content-type") or "").lower()
            return {
                "status_code": r.status_code,
                "final_url": r.url,
                "content_type": r.headers.get("content-type"),
                "content_range": r.headers.get("content-range"),
                "bytes_received": len(payload),
                "prefix_hex": payload[:16].hex(),
                "is_tiff": is_tiff,
                "is_html": is_html,
                "ok": r.status_code in (200, 206) and is_tiff and not is_html,
            }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def polarization_for_key(key: str) -> str | None:
    name = key.lower()
    if not name.endswith((".tif", ".tiff")) or "/measurement/" not in name:
        return None
    if "-vv-" in name or name.endswith("_vv.tif") or name.endswith("_vv.tiff"):
        return "VV"
    if "-vh-" in name or name.endswith("_vh.tif") or name.endswith("_vh.tiff"):
        return "VH"
    return None


def main() -> None:
    scenes = select_scenes()
    phase_reports: dict[str, object] = {}
    targets: list[dict[str, str]] = []

    for scene in scenes:
        date = scene["date"]
        if "error" in scene:
            phase_reports[date] = scene
            continue
        listing = list_prefix(scene["prefix"])
        keys = list(listing.get("keys", []))
        pol_keys: dict[str, list[str]] = {"VV": [], "VH": []}
        for key in keys:
            pol = polarization_for_key(key)
            if pol:
                pol_keys[pol].append(key)
        selected = {pol: sorted(values)[0] if values else None for pol, values in pol_keys.items()}
        phase_reports[date] = {
            **scene,
            "listing": {k: v for k, v in listing.items() if k != "keys"},
            "key_count": len(keys),
            "measurement_key_count": sum(len(v) for v in pol_keys.values()),
            "selected_keys": selected,
        }
        for pol, key in selected.items():
            if key:
                targets.append({
                    "date": date,
                    "scene_name": scene["scene_name"],
                    "polarization": pol,
                    "key": key,
                    "url": f"{RODA_BUCKET}/{quote(key, safe='/')}",
                })

    probes: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(probe_tiff, target["url"]): target for target in targets}
        for future in as_completed(future_map):
            target = future_map[future]
            probes.append({**target, **future.result()})
    probes.sort(key=lambda r: (str(r["date"]), str(r["polarization"])))

    pd.DataFrame(probes).to_csv(OUT / "public_s1_roda_probe.csv", index=False)
    per_date: dict[str, dict[str, bool]] = {date: {"VV": False, "VH": False} for date in TARGET_DATES}
    for record in probes:
        per_date[str(record["date"])][str(record["polarization"])] = bool(record.get("ok"))

    report = {
        "source": "Sentinel-1 GRD public RODA HTTP proxy",
        "bucket": RODA_BUCKET,
        "phase_reports": phase_reports,
        "probes": probes,
        "per_date_access": per_date,
        "all_dates_have_public_vv_vh": all(all(v.values()) for v in per_date.values()),
        "interpretation": "This stage only proves anonymous byte access to the exact GRD measurement COGs. It does not perform calibration, terrain correction, mosaicking, classification or plotting.",
    }
    (OUT / "public_s1_roda_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
