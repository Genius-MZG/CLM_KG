from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "river_geometry"
OUT.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "21415370"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
USER_AGENT = "NorthAmericaFloodplainResearch/1.0"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    response = requests.get(ZENODO_API, timeout=(10, 60), headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    metadata = response.json()
    write_json(OUT / "sword_v17c_zenodo_metadata.json", metadata)

    files = metadata.get("files", [])
    candidates = []
    for entry in files:
        key = str(entry.get("key", ""))
        lower = key.lower()
        if "geoparquet" in lower or "parquet" in lower:
            links = entry.get("links", {}) or {}
            candidates.append({
                "key": key,
                "size": entry.get("size"),
                "checksum": entry.get("checksum"),
                # Zenodo's current records API exposes the downloadable content endpoint
                # as links.self for files. Retain links.content as a compatibility fallback.
                "content_url": links.get("content") or links.get("self"),
            })

    preferred = [row for row in candidates if "na" in row["key"].lower() or "north" in row["key"].lower()]
    inspection_targets = preferred or candidates[:1]
    inspections = []
    for row in inspection_targets:
        url = row.get("content_url")
        if not url:
            inspections.append({**row, "error": "missing content URL"})
            continue
        try:
            with RemoteZip(url, initial_buffer_size=4 * 1024 * 1024) as archive:
                infos = archive.infolist()
            members = [info.filename for info in infos]
            reach_members = [name for name in members if "reach" in name.lower() and name.lower().endswith((".parquet", ".geoparquet"))]
            na_members = [name for name in members if ("/na" in name.lower() or name.lower().startswith("na") or "north_america" in name.lower())]
            member_rows = [
                {
                    "filename": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": info.CRC,
                }
                for info in infos
                if info.filename in set(reach_members + na_members)
            ]
            inspections.append({
                **row,
                "member_count": len(members),
                "reach_members": reach_members,
                "north_america_members": na_members[:100],
                "verified_member_metadata": member_rows[:200],
                "all_members_sample": members[:100],
            })
        except Exception as exc:
            inspections.append({**row, "error": f"{type(exc).__name__}: {exc}"})

    geometry_recovered = any(bool(item.get("reach_members")) for item in inspections)
    report = {
        "source": "Official SWORD v17c Zenodo record",
        "record_id": ZENODO_RECORD,
        "record_version": metadata.get("metadata", {}).get("version"),
        "record_doi": metadata.get("metadata", {}).get("doi"),
        "candidate_geoparquet_archives": candidates,
        "inspections": inspections,
        "archive_member_inventory_recovered": geometry_recovered,
        "geometry_recovered": False,
        "next_gate": "Extract only the verified North America reaches GeoParquet member, then spatially subset by the fixed AOI and station corridor before Figure 1 rendering.",
        "scientific_rule": "No centerline is drawn until an official SWORD member and its real geometry are extracted and spatially verified.",
    }
    write_json(OUT / "sword_geometry_probe.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
