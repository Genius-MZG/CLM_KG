from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "source_access_probe"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_DATES = ["2017-11-01", "2017-11-13", "2017-11-25", "2017-12-07", "2017-12-19"]
RANGE_BYTES = 65536
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20
MAX_WORKERS = 8


def classify_payload(kind: str, status_code: int, content_type: str, body: bytes, final_url: str) -> dict[str, object]:
    prefix = body[:16]
    lower_type = (content_type or "").lower()
    lower_url = (final_url or "").lower()
    is_html = prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")) or "text/html" in lower_type
    is_tiff = prefix.startswith((b"II*\x00", b"MM\x00*"))
    is_zip = prefix.startswith(b"PK\x03\x04")
    auth_challenge = "urs.earthdata.nasa.gov" in lower_url or status_code in (401, 403) or is_html
    expected_signature = is_tiff if kind.startswith("BURST_") else is_zip
    return {
        "is_html": is_html,
        "is_tiff": is_tiff,
        "is_zip": is_zip,
        "auth_challenge": auth_challenge,
        "expected_signature": expected_signature,
        "ok": status_code in (200, 206) and len(body) > 0 and expected_signature and not auth_challenge,
    }


def probe(kind: str, url: str) -> dict[str, object]:
    headers = {
        "Range": f"bytes=0-{RANGE_BYTES - 1}",
        "User-Agent": "NorthAmericaFloodplainResearch/1.0",
        "Accept-Encoding": "identity",
    }
    try:
        with requests.get(
            url,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
            stream=True,
        ) as r:
            body = bytearray()
            for chunk in r.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                need = RANGE_BYTES - len(body)
                body.extend(chunk[:need])
                if len(body) >= RANGE_BYTES:
                    break
            payload = bytes(body)
            classification = classify_payload(
                kind,
                r.status_code,
                r.headers.get("content-type", ""),
                payload,
                r.url,
            )
            return {
                "status_code": r.status_code,
                "final_url": r.url,
                "content_type": r.headers.get("content-type"),
                "content_range": r.headers.get("content-range"),
                "content_length_header": r.headers.get("content-length"),
                "accept_ranges": r.headers.get("accept-ranges"),
                "bytes_received": len(payload),
                "redirect_history": [x.status_code for x in r.history],
                "prefix_hex": payload[:16].hex(),
                "range_honored": r.status_code == 206 or r.headers.get("content-range") is not None,
                **classification,
            }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def selected_targets(df: pd.DataFrame) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for date in TARGET_DATES:
        daily = df[df["start_time"].astype(str).str.startswith(date)].copy()
        grd = daily[
            (daily["processing_level"] == "GRD_HD")
            & (daily["scene_name"].str.contains("_GRDH_1SDV_", na=False))
        ]
        bursts = daily[daily["processing_level"] == "BURST"]
        selected: list[tuple[str, pd.Series]] = []
        if not grd.empty:
            selected.append(("GRD_HD", grd.iloc[0]))
        for pol in ["VV", "VH"]:
            rows = bursts[bursts["polarization"] == pol]
            if not rows.empty:
                selected.append((f"BURST_{pol}", rows.iloc[0]))
        for kind, row in selected:
            targets.append(
                {
                    "date": date,
                    "kind": kind,
                    "scene_name": str(row["scene_name"]),
                    "url": str(row["url"]),
                }
            )
    return targets


def main() -> None:
    df = pd.read_csv(ROOT / "asf_sentinel1_candidates.csv")
    targets = selected_targets(df)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(probe, target["kind"], target["url"]): target for target in targets}
        for future in as_completed(future_map):
            target = future_map[future]
            result = future.result()
            records.append({**target, **result})

    records.sort(key=lambda r: (str(r["date"]), str(r["kind"])))
    pd.DataFrame(records).to_csv(OUT / "asf_source_access_probe.csv", index=False)
    grd_records = [r for r in records if r["kind"] == "GRD_HD"]
    burst_records = [r for r in records if str(r["kind"]).startswith("BURST_")]
    report = {
        "probe_type": "bounded unauthenticated HTTP range GET with payload signature validation",
        "range_bytes": RANGE_BYTES,
        "connect_timeout_seconds": CONNECT_TIMEOUT,
        "read_timeout_seconds": READ_TIMEOUT,
        "max_workers": MAX_WORKERS,
        "records": records,
        "all_grd_accessible": bool(grd_records) and all(bool(r.get("ok")) for r in grd_records),
        "all_bursts_accessible": bool(burst_records) and all(bool(r.get("ok")) for r in burst_records),
        "authentication_blocked_records": sum(bool(r.get("auth_challenge")) for r in records),
        "interpretation": "HTTP 200 is not sufficient: Earthdata login HTML is rejected. BURST targets must begin with a TIFF signature and GRD targets with a ZIP signature.",
    }
    (OUT / "asf_source_access_probe.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
