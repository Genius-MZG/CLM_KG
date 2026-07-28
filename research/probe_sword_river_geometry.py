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
            candidates.append({
                "key": key,
                "size": entry.get("size"),
                "checksum": entry.get("checksum"),
                "content_url": entry.get("links", {}).get("content"),
            })

    preferred = [row for row in candidates if "na" in row["key"].lower() or "north" in row["key"].lower()]
    inspection_targets = preferred or candidates[:3]
    inspections = []
    for row in inspection_targets:
        url = row.get("content_url")
        if not url:
            inspections.append({**row, "error": "missing content URL"})
            continue
        try:
            with RemoteZip(url, initial_buffer_size=2 * 1024 * 1024) as archive:
                members = [info.filename for info in archive.infolist()]
            reach_members = [name for name in members if "reach" in name.lower() and name.lower().endswith((".parquet", ".geoparquet"))]
            na_members = [name for name in members if ("/na" in name.lower() or name.lower().startswith("na") or "north_america" in name.lower())]
            inspections.append({
                **row,
                "member_count": len(members),
                "reach_members": reach_members,
                "north_america_members": na_members[:100],
                "all_members_sample": members[:100],
            })
        except Exception as exc:
            inspections.append({**row, "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "source": "Official SWORD v17c Zenodo record",
        "record_id": ZENODO_RECORD,
        "record_version": metadata.get("metadata", {}).get("version"),
        "record_doi": metadata.get("metadata", {}).get("doi"),
        "candidate_geoparquet_archives": candidates,
        "inspections": inspections,
        "geometry_recovered": False,
        "next_gate": "Extract only the verified North America reaches GeoParquet member, then spatially subset by the fixed AOI and station corridor before Figure 1 rendering.",
        "scientific_rule": "No centerline is drawn until an official SWORD member and its real geometry are extracted and spatially verified.",
    }
    write_json(OUT / "sword_geometry_probe.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
