from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "river_geometry"
OUT.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "21415370"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_FILE_ENDPOINT = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
USER_AGENT = "NorthAmericaFloodplainResearch/1.0"
MAX_EXTRACT_BYTES = 800 * 1024 * 1024


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resilient_session() -> requests.Session:
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET", "HEAD"}), raise_on_status=False)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    return session


def canonical_content_url(key: str, links: dict[str, Any]) -> str:
    linked = str(links.get("content") or links.get("self") or "")
    if linked.rstrip("/").endswith("/content"):
        return linked
    return f"{ZENODO_FILE_ENDPOINT}/{quote(key, safe='')}/content"


def is_na(filename: str) -> bool:
    name = Path(filename).name.lower()
    return name.startswith("na_") or name.startswith("sword_na_") or "_na_" in name or "north_america" in name


def inventory_and_extract(row: dict[str, Any]) -> dict[str, Any]:
    url = str(row["content_url"])
    inspection: dict[str, Any] = {**row}
    try:
        with RemoteZip(url, initial_buffer_size=8 * 1024 * 1024) as archive:
            infos = archive.infolist()
            members = [info.filename for info in infos]
            reach_infos = [info for info in infos if "reach" in info.filename.lower() and info.filename.lower().endswith((".parquet", ".geoparquet"))]
            na_reach_infos = [info for info in reach_infos if is_na(info.filename)]
            na_infos = [info for info in infos if is_na(info.filename)]
            selected = sorted(na_reach_infos, key=lambda info: info.file_size)[0] if na_reach_infos else None
            verified = [{"filename": info.filename, "file_size": info.file_size, "compress_size": info.compress_size, "crc": info.CRC} for info in sorted(set(reach_infos + na_infos), key=lambda info: info.filename)]
            inspection.update({"member_count": len(members), "reach_members": [info.filename for info in reach_infos], "north_america_members": [info.filename for info in na_infos[:200]], "verified_member_metadata": verified[:400], "all_members_sample": members[:100], "selected_north_america_reaches_member": selected.filename if selected else None})
            if selected is not None and selected.file_size <= MAX_EXTRACT_BYTES:
                destination = OUT / Path(selected.filename).name
                temporary = destination.with_suffix(destination.suffix + ".part")
                if temporary.exists():
                    temporary.unlink()
                with archive.open(selected.filename) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                temporary.replace(destination)
                inspection["extracted_member"] = {"archive_member": selected.filename, "output": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size, "sha256": sha256(destination)}
            elif selected is not None:
                inspection["extraction_skipped"] = {"reason": "verified member exceeds bounded extraction size", "file_size": selected.file_size, "max_extract_bytes": MAX_EXTRACT_BYTES}
    except Exception as exc:
        inspection["error"] = f"{type(exc).__name__}: {exc}"
        inspection["traceback"] = traceback.format_exc()
    return inspection


def main() -> None:
    started = time.time()
    report: dict[str, Any] = {"source": "Official SWORD v17c Zenodo record", "record_id": ZENODO_RECORD, "archive_member_inventory_recovered": False, "geometry_member_extracted": False, "geometry_recovered": False, "scientific_rule": "No centerline is drawn until an official SWORD member and its real geometry are extracted and spatially verified."}
    try:
        session = resilient_session()
        response = session.get(ZENODO_API, timeout=(10, 60))
        response.raise_for_status()
        metadata = response.json()
        write_json(OUT / "sword_v17c_zenodo_metadata.json", metadata)
        candidates: list[dict[str, Any]] = []
        for entry in metadata.get("files", []):
            key = str(entry.get("key", ""))
            if "geoparquet" in key.lower() or "parquet" in key.lower():
                links = entry.get("links", {}) or {}
                candidates.append({"key": key, "size": entry.get("size"), "checksum": entry.get("checksum"), "content_url": canonical_content_url(key, links), "api_links": links})
        preferred = [row for row in candidates if row["key"].lower() == "sword_v17c_parquet.zip"]
        inspections = [inventory_and_extract(row) for row in (preferred or candidates[:1])]
        report.update({"record_version": metadata.get("metadata", {}).get("version"), "record_doi": metadata.get("metadata", {}).get("doi"), "candidate_geoparquet_archives": candidates, "inspections": inspections, "archive_member_inventory_recovered": any(bool(item.get("reach_members")) for item in inspections), "geometry_member_extracted": any(bool(item.get("extracted_member")) for item in inspections), "geometry_recovered": False, "next_gate": "Read the extracted official North America reaches GeoParquet, spatially subset by the fixed AOI and station corridor, verify topology and coordinates, then render Figure 1."})
    except Exception as exc:
        report["fatal_probe_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["next_gate"] = "Retry the official Zenodo metadata and bounded archive-member probe; the independent RTC/model pipeline must continue meanwhile."
    finally:
        report["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(OUT / "sword_geometry_probe.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
