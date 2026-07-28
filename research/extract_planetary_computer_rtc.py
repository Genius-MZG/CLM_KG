from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "planetary_computer_rtc"
OUT.mkdir(parents=True, exist_ok=True)

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_ENDPOINT = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
COLLECTION = "sentinel-1-rtc"
BBOX_WGS84 = (-122.48, 48.34, -121.62, 48.66)
DST_CRS = CRS.from_epsg(32610)
RESOLUTION = 30.0
NODATA = -9999.0
PHASES = {
    "baseline": "2017-11-01",
    "rising": "2017-11-13",
    "peak": "2017-11-25",
    "early_recession": "2017-12-07",
    "late_recession": "2017-12-19",
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def aligned_grid() -> tuple[Any, int, int, tuple[float, float, float, float]]:
    left, bottom, right, top = transform_bounds("EPSG:4326", DST_CRS, *BBOX_WGS84, densify_pts=21)
    left = math.floor(left / RESOLUTION) * RESOLUTION
    bottom = math.floor(bottom / RESOLUTION) * RESOLUTION
    right = math.ceil(right / RESOLUTION) * RESOLUTION
    top = math.ceil(top / RESOLUTION) * RESOLUTION
    width = int(round((right - left) / RESOLUTION))
    height = int(round((top - bottom) / RESOLUTION))
    transform = from_origin(left, top, RESOLUTION, RESOLUTION)
    return transform, width, height, (left, bottom, right, top)


def get_item(date_text: str) -> dict[str, Any]:
    payload = {
        "collections": [COLLECTION],
        "bbox": list(BBOX_WGS84),
        "datetime": f"{date_text}T00:00:00Z/{date_text}T23:59:59Z",
        "limit": 20,
    }
    r = requests.post(STAC_SEARCH, json=payload, timeout=90, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
    r.raise_for_status()
    features = r.json().get("features", [])
    if not features:
        raise RuntimeError(f"No {COLLECTION} item for {date_text}")
    dual = [f for f in features if "vv" in f.get("assets", {}) and "vh" in f.get("assets", {})]
    if len(dual) != 1:
        raise RuntimeError(f"Expected exactly one dual-pol RTC item for {date_text}; found {len(dual)}")
    return dual[0]


def sign_href(href: str) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(href)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) < 2:
        raise RuntimeError(f"Cannot identify Azure container for {href}")
    account = parsed.netloc.split(".")[0]
    container = parts[0]
    token_url = f"{SAS_ENDPOINT}/{account}/{container}"
    r = requests.get(token_url, timeout=60, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
    r.raise_for_status()
    payload = r.json()
    token = payload.get("token")
    if not token:
        raise RuntimeError(f"No SAS token returned for {account}/{container}")
    sep = "&" if "?" in href else "?"
    return href + sep + token, {"token_endpoint": token_url, "expiry": payload.get("msft:expiry")}


def source_metadata(url: str) -> dict[str, Any]:
    with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(url) as src:
            return {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": str(src.crs),
                "transform": tuple(src.transform),
                "nodata": src.nodata,
                "bounds": tuple(src.bounds),
                "block_shapes": src.block_shapes,
            }


def reproject_asset(url: str, dst_transform: Any, width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    dst = np.full((height, width), NODATA, dtype="float32")
    with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(url) as src:
            src_nodata = src.nodata
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=DST_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
                num_threads=2,
            )
            meta = source_metadata(url)
    return dst, meta


def write_raster(path: Path, array: np.ndarray, transform: Any, *, dtype: str, nodata: float | int, tags: dict[str, str]) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": DST_CRS,
        "transform": transform,
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 3 if dtype.startswith("float") else 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.update_tags(**tags)
        dst.build_overviews([2, 4, 8, 16], Resampling.average if dtype.startswith("float") else Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="average" if dtype.startswith("float") else "nearest")


def main() -> None:
    transform, width, height, bounds = aligned_grid()
    phase_reports: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []

    for phase, date_text in PHASES.items():
        item = get_item(date_text)
        item_id = item["id"]
        phase_dir = OUT / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        assets_report: dict[str, Any] = {}

        for pol in ("vv", "vh"):
            asset = item["assets"][pol]
            signed_url, token_meta = sign_href(asset["href"])
            array, src_meta = reproject_asset(signed_url, transform, width, height)
            arrays[pol] = array
            out_path = phase_dir / f"{phase}_{pol}_rtc_30m.tif"
            tags = {
                "source_collection": COLLECTION,
                "source_item_id": item_id,
                "source_asset": pol,
                "phase": phase,
                "acquisition_date": date_text,
                "processing_note": "Reprojected from Microsoft Planetary Computer Sentinel-1 RTC COG to EPSG:32610 at 30 m",
            }
            write_raster(out_path, array, transform, dtype="float32", nodata=NODATA, tags=tags)
            assets_report[pol] = {
                "source_href": asset["href"],
                "token_metadata": token_meta,
                "source_metadata": src_meta,
                "output": str(out_path.relative_to(ROOT)),
                "sha256": sha256(out_path),
            }
            manifest_rows.append({"phase": phase, "date": date_text, "layer": pol, "path": str(out_path.relative_to(ROOT)), "sha256": sha256(out_path), "bytes": out_path.stat().st_size})

        valid = np.isfinite(arrays["vv"]) & np.isfinite(arrays["vh"]) & (arrays["vv"] != NODATA) & (arrays["vh"] != NODATA)
        mask = valid.astype("uint8")
        mask_path = phase_dir / f"{phase}_valid_mask_30m.tif"
        write_raster(mask_path, mask, transform, dtype="uint8", nodata=0, tags={"phase": phase, "acquisition_date": date_text, "definition": "1 where both reprojected VV and VH RTC pixels are valid; 0 otherwise"})
        manifest_rows.append({"phase": phase, "date": date_text, "layer": "valid_mask", "path": str(mask_path.relative_to(ROOT)), "sha256": sha256(mask_path), "bytes": mask_path.stat().st_size})

        phase_reports[phase] = {
            "date": date_text,
            "item_id": item_id,
            "platform": item.get("properties", {}).get("platform"),
            "orbit_state": item.get("properties", {}).get("sat:orbit_state"),
            "relative_orbit": item.get("properties", {}).get("sat:relative_orbit"),
            "assets": assets_report,
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
            "valid_fraction": float(valid.mean()),
            "vv_valid_min": float(np.nanmin(np.where(valid, arrays["vv"], np.nan))),
            "vv_valid_max": float(np.nanmax(np.where(valid, arrays["vv"], np.nan))),
            "vh_valid_min": float(np.nanmin(np.where(valid, arrays["vh"], np.nan))),
            "vh_valid_max": float(np.nanmax(np.where(valid, arrays["vh"], np.nan))),
        }

    common = np.ones((height, width), dtype=bool)
    for phase in PHASES:
        with rasterio.open(OUT / phase / f"{phase}_valid_mask_30m.tif") as src:
            common &= src.read(1).astype(bool)
    common_path = OUT / "common_valid_mask_30m.tif"
    write_raster(common_path, common.astype("uint8"), transform, dtype="uint8", nodata=0, tags={"definition": "Intersection of five phase-specific dual-polarization valid masks", "phase_count": str(len(PHASES))})
    manifest_rows.append({"phase": "all", "date": "", "layer": "common_valid_mask", "path": str(common_path.relative_to(ROOT)), "sha256": sha256(common_path), "bytes": common_path.stat().st_size})

    with (OUT / "sha256_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "date", "layer", "path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "source": "Microsoft Planetary Computer sentinel-1-rtc",
        "aoi_bbox_wgs84": BBOX_WGS84,
        "output_crs": str(DST_CRS),
        "resolution_m": RESOLUTION,
        "grid_width": width,
        "grid_height": height,
        "grid_bounds": bounds,
        "phase_count": len(phase_reports),
        "phase_reports": phase_reports,
        "common_valid_pixels": int(common.sum()),
        "common_valid_fraction": float(common.mean()),
        "all_phase_rasters_complete": len(phase_reports) == 5 and all(v["valid_pixels"] > 0 for v in phase_reports.values()),
        "scientific_caveat": "These are Planetary Computer Sentinel-1 RTC assets, not the frozen original HPC OPERA RTC files. They are a transparent public-source reconstruction and must not be described as exact replicas of the HPC products.",
    }
    write_json(OUT / "rtc_extraction_status.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
