from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "model_inputs"
OUT.mkdir(parents=True, exist_ok=True)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [-122.48, 48.34, -121.62, 48.66]
KEYWORDS = ("jrc", "water", "surface-water", "dem", "elevation", "cop-dem")


def get_json(url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.get(url, timeout=60, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"}, **kwargs)
    response.raise_for_status()
    return response.json()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=90, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
    response.raise_for_status()
    return response.json()


def main() -> None:
    collections = get_json(f"{STAC}/collections").get("collections", [])
    candidates = []
    for collection in collections:
        text = " ".join(str(collection.get(k, "")) for k in ("id", "title", "description")).lower()
        if any(keyword in text for keyword in KEYWORDS):
            candidates.append({
                "id": collection.get("id"),
                "title": collection.get("title"),
                "description": collection.get("description"),
                "item_assets": collection.get("item_assets", {}),
                "extent": collection.get("extent"),
            })

    probes: list[dict[str, Any]] = []
    for candidate in candidates:
        collection_id = candidate["id"]
        try:
            result = post_json(f"{STAC}/search", {
                "collections": [collection_id],
                "bbox": BBOX,
                "limit": 5,
            })
            features = result.get("features", [])
            probes.append({
                "collection_id": collection_id,
                "feature_count_returned": len(features),
                "sample_items": [
                    {
                        "id": feature.get("id"),
                        "datetime": feature.get("properties", {}).get("datetime"),
                        "asset_keys": sorted(feature.get("assets", {}).keys()),
                        "assets": {
                            key: {
                                "href": value.get("href"),
                                "type": value.get("type"),
                                "roles": value.get("roles"),
                            }
                            for key, value in feature.get("assets", {}).items()
                        },
                    }
                    for feature in features[:2]
                ],
                "error": None,
            })
        except Exception as exc:
            probes.append({"collection_id": collection_id, "feature_count_returned": 0, "sample_items": [], "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "source": "Microsoft Planetary Computer STAC",
        "aoi_bbox_wgs84": BBOX,
        "candidate_collection_count": len(candidates),
        "candidate_collections": candidates,
        "probes": probes,
        "water_candidates": [p["collection_id"] for p in probes if any(token in p["collection_id"].lower() for token in ("jrc", "water")) and p["feature_count_returned"] > 0],
        "dem_candidates": [p["collection_id"] for p in probes if any(token in p["collection_id"].lower() for token in ("dem", "elevation")) and p["feature_count_returned"] > 0],
        "ready_for_extraction": any(any(token in p["collection_id"].lower() for token in ("jrc", "water")) and p["feature_count_returned"] > 0 for p in probes) and any(any(token in p["collection_id"].lower() for token in ("dem", "elevation")) and p["feature_count_returned"] > 0 for p in probes),
        "scientific_constraint": "This stage only inventories real public datasets and asset keys. It does not create labels, terrain variables, model metrics, or figures.",
    }
    (OUT / "model_input_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
