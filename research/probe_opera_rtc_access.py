from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
CAT = ROOT / "opera_rtc_catalog"
OUT = CAT / "access_probe"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = CAT / "opera_rtc_expected_track_candidates.csv"
RANGE_BYTES = 65536
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 15
MAX_WORKERS = 10


def layer_urls(urls: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for url in urls:
        lower = url.lower()
        if lower.endswith("_vv.tif"):
            result["VV"] = url
        elif lower.endswith("_vh.tif"):
            result["VH"] = url
        elif lower.endswith("_mask.tif"):
            result["mask"] = url
        elif lower.endswith("_browse_low-res.png"):
            result["browse_low_res"] = url
        elif lower.endswith(".h5"):
            result["h5"] = url
    return result


def classify_signature(body: bytes, content_type: str | None) -> tuple[bool, str]:
    prefix = body[:16]
    ctype = (content_type or "").lower()
    if prefix.startswith((b"II*\x00", b"MM\x00*")):
        return True, "tiff"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return True, "png"
    if prefix.startswith(b"\x89HDF\r\n\x1a\n"):
        return True, "hdf5"
    if "html" in ctype or prefix.lstrip().lower().startswith((b"<!doctype", b"<html")):
        return False, "html_or_login_page"
    return False, "unknown_signature"


def probe(url: str, *, save_path: Path | None = None) -> dict[str, object]:
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
            valid_signature, detected_type = classify_signature(payload, r.headers.get("content-type"))
            ok = r.status_code in (200, 206) and bool(payload) and valid_signature
            if save_path is not None and ok:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(payload)
            return {
                "ok": ok,
                "status_code": r.status_code,
                "final_url": r.url,
                "content_type": r.headers.get("content-type"),
                "content_length_header": r.headers.get("content-length"),
                "content_range": r.headers.get("content-range"),
                "accept_ranges": r.headers.get("accept-ranges"),
                "bytes_received": len(payload),
                "redirect_history": [x.status_code for x in r.history],
                "first_16_bytes_hex": payload[:16].hex(),
                "range_honored": r.status_code == 206 or r.headers.get("content-range") is not None,
                "detected_type": detected_type,
            }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    df = pd.read_csv(CANDIDATES)
    targets: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for phase, group in df.groupby("phase", sort=False):
        row = group.sort_values(["burst_id", "product_name"]).iloc[0]
        urls = layer_urls(json.loads(row["urls_json"]))
        summaries[str(phase)] = {
            "phase_date": row["phase_date"],
            "product_name": row["product_name"],
            "burst_id": row["burst_id"],
            "layers": {},
        }
        for layer in ["VV", "VH", "mask", "h5", "browse_low_res"]:
            targets.append({
                "phase": str(phase),
                "phase_date": row["phase_date"],
                "product_name": row["product_name"],
                "burst_id": row["burst_id"],
                "layer": layer,
                "url": urls.get(layer),
            })

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for target in targets:
            url = target["url"]
            if not url:
                result = {"ok": False, "error": "URL missing"}
                record = {**target, **result}
                records.append(record)
                summaries[target["phase"]]["layers"][target["layer"]] = result
                continue
            suffix = ".png" if target["layer"] == "browse_low_res" else ".bin"
            save = OUT / f"{target['phase']}_{target['layer']}_prefix{suffix}"
            futures[pool.submit(probe, str(url), save_path=save)] = target
        for future in as_completed(futures):
            target = futures[future]
            result = future.result()
            record = {**target, **result}
            records.append(record)
            summaries[target["phase"]]["layers"][target["layer"]] = result

    records.sort(key=lambda r: (str(r["phase"]), str(r["layer"])))
    with (OUT / "opera_rtc_access_probe.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in records for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    all_raster_layers_accessible = all(
        bool(summaries[p]["layers"][layer].get("ok"))
        for p in summaries
        for layer in ["VV", "VH", "mask"]
    )
    report = {
        "probe_type": "bounded parallel unauthenticated HTTP range GET with file-signature validation",
        "range_bytes": RANGE_BYTES,
        "connect_timeout_seconds": CONNECT_TIMEOUT,
        "read_timeout_seconds": READ_TIMEOUT,
        "max_workers": MAX_WORKERS,
        "phase_count": len(summaries),
        "all_raster_layers_accessible": all_raster_layers_accessible,
        "phase_results": summaries,
        "interpretation": (
            "Only responses with valid TIFF/PNG/HDF5 signatures count as accessible. HTML login pages and redirects do not."
        ),
    }
    (OUT / "opera_rtc_access_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
