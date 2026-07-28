from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "source_access_probe"
OUT.mkdir(parents=True, exist_ok=True)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
BBOX = [-122.48, 48.34, -121.62, 48.66]
DATES = ["2017-11-01", "2017-11-13", "2017-11-25", "2017-12-07", "2017-12-19"]
COLLECTIONS = ["sentinel-1-grd", "sentinel-1-rtc"]
RANGE_BYTES = 65536


def payload_kind(body: bytes, content_type: str | None) -> str:
    head = body[:16]
    ctype = (content_type or "").lower()
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
    if b"html" in ctype or body[:512].lower().find(b"<html") >= 0:
        return "html"
    if "json" in ctype:
        return "json"
    return "other"


def probe(url: str) -> dict[str, object]:
    try:
        with requests.get(
            url,
            headers={
                "Range": f"bytes=0-{RANGE_BYTES - 1}",
                "Accept-Encoding": "identity",
                "User-Agent": "NorthAmericaFloodplainResearch/1.0",
            },
            timeout=(8, 20),
            allow_redirects=True,
            stream=True,
        ) as response:
            body = bytearray()
            for chunk in response.iter_content(16384):
                if not chunk:
                    continue
                body.extend(chunk[: RANGE_BYTES - len(body)])
                if len(body) >= RANGE_BYTES:
                    break
            kind = payload_kind(bytes(body), response.headers.get("content-type"))
            return {
                "status_code": response.status_code,
                "final_url": response.url,
                "content_type": response.headers.get("content-type"),
                "bytes_received": len(body),
                "payload_kind": kind,
                "valid_science_payload": kind in {"tiff", "zip", "hdf5"},
                "prefix_hex": bytes(body[:16]).hex(),
                "redirect_history": [x.status_code for x in response.history],
            }
    except Exception as exc:
        return {"valid_science_payload": False, "error": f"{type(exc).__name__}: {exc}"}


def search_items(collection: str, date: str) -> dict[str, object]:
    payload = {
        "collections": [collection],
        "bbox": BBOX,
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
        "limit": 100,
    }
    try:
        response = requests.post(f"{STAC}/search", json=payload, timeout=(8, 30))
        result: dict[str, object] = {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
        }
        if response.status_code != 200:
            result["body_prefix"] = response.text[:1000]
            return result
        data = response.json()
        features = data.get("features", [])
        result["item_count"] = len(features)
        result["items"] = features
        return result
    except Exception as exc:
        return {"item_count": 0, "error": f"{type(exc).__name__}: {exc}"}


def candidate_assets(feature: dict[str, object]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    assets = feature.get("assets", {}) if isinstance(feature, dict) else {}
    if not isinstance(assets, dict):
        return result
    for key, asset in assets.items():
        if not isinstance(asset, dict):
            continue
        href = asset.get("href")
        if not isinstance(href, str):
            continue
        roles = asset.get("roles") or []
        media = str(asset.get("type") or "")
        lowered = f"{key} {href} {media}".lower()
        if any(token in lowered for token in ["vv", "vh", ".tif", ".tiff", ".zip", ".h5"]):
            result.append({"asset_key": str(key), "href": href, "media_type": media, "roles": json.dumps(roles)})
    return result


def token_url_for(href: str) -> str | None:
    parsed = urlparse(href)
    parts = [p for p in parsed.path.split("/") if p]
    host = parsed.netloc
    if ".blob.core.windows.net" not in host or not parts:
        return None
    account = host.split(".")[0]
    container = parts[0]
    return f"{TOKEN}/{account}/{container}"


def main() -> None:
    searches: dict[str, object] = {}
    targets: list[dict[str, str]] = []
    token_results: dict[str, object] = {}

    for collection in COLLECTIONS:
        for date in DATES:
            key = f"{collection}:{date}"
            search = search_items(collection, date)
            items = search.pop("items", []) if isinstance(search, dict) else []
            searches[key] = search
            if not isinstance(items, list):
                continue
            for feature in items:
                if not isinstance(feature, dict):
                    continue
                item_id = str(feature.get("id"))
                for asset in candidate_assets(feature):
                    href = asset["href"]
                    targets.append({
                        "collection": collection,
                        "date": date,
                        "item_id": item_id,
                        **asset,
                        "access_mode": "raw_href",
                        "url": href,
                    })
                    token_url = token_url_for(href)
                    if token_url and token_url not in token_results:
                        try:
                            r = requests.get(token_url, timeout=(8, 20))
                            entry: dict[str, object] = {
                                "status_code": r.status_code,
                                "content_type": r.headers.get("content-type"),
                                "body_prefix": r.text[:1000],
                            }
                            if r.status_code == 200:
                                data = r.json()
                                entry["has_token"] = bool(data.get("token"))
                                entry["expiry"] = data.get("msft:expiry")
                            token_results[token_url] = entry
                        except Exception as exc:
                            token_results[token_url] = {"error": f"{type(exc).__name__}: {exc}"}

    # Limit duplicate probes while retaining both collections and all dates.
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for target in targets:
        key = (target["collection"], target["date"], target["url"])
        unique[key] = target
    probe_targets = list(unique.values())[:120]

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_map = {pool.submit(probe, target["url"]): target for target in probe_targets}
        for future in as_completed(future_map):
            records.append({**future_map[future], **future.result()})
    records.sort(key=lambda x: (str(x.get("collection")), str(x.get("date")), str(x.get("asset_key"))))

    per_collection_date: dict[str, bool] = {}
    for collection in COLLECTIONS:
        for date in DATES:
            per_collection_date[f"{collection}:{date}"] = any(
                r.get("collection") == collection
                and r.get("date") == date
                and bool(r.get("valid_science_payload"))
                for r in records
            )

    report = {
        "source": "Microsoft Planetary Computer STAC and SAS token service",
        "bbox": BBOX,
        "dates": DATES,
        "collections": COLLECTIONS,
        "searches": searches,
        "token_results": token_results,
        "records": records,
        "per_collection_date_access": per_collection_date,
        "all_dates_have_public_grd": all(per_collection_date[f"sentinel-1-grd:{d}"] for d in DATES),
        "all_dates_have_public_rtc": all(per_collection_date[f"sentinel-1-rtc:{d}"] for d in DATES),
        "interpretation": "Only valid TIFF, ZIP or HDF5 signatures count as scientific data access. STAC metadata presence alone is insufficient.",
    }
    (OUT / "planetary_computer_s1_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "searches", "token_results"}}, indent=2))


if __name__ == "__main__":
    main()
