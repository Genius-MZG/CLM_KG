from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import traceback
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
MAX_ASSET_ATTEMPTS = 4
PHASES = {
    "baseline": "2017-11-01",
    "rising": "2017-11-13",
    "peak": "2017-11-25",
    "early_recession": "2017-12-07",
    "late_recession": "2017-12-19",
}
GDAL_ENV = {
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_USE_HEAD": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": 67108864,
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


def aligned_grid() -> tuple[Any, int, int, tuple[float, float, float, float]]:
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", DST_CRS, *BBOX_WGS84, densify_pts=21
    )
    left = math.floor(left / RESOLUTION) * RESOLUTION
    bottom = math.floor(bottom / RESOLUTION) * RESOLUTION
    right = math.ceil(right / RESOLUTION) * RESOLUTION
    top = math.ceil(top / RESOLUTION) * RESOLUTION
    width = int(round((right - left) / RESOLUTION))
    height = int(round((top - bottom) / RESOLUTION))
    return (
        from_origin(left, top, RESOLUTION, RESOLUTION),
        width,
        height,
        (left, bottom, right, top),
    )


def get_item(date_text: str) -> dict[str, Any]:
    payload = {
        "collections": [COLLECTION],
        "bbox": list(BBOX_WGS84),
        "datetime": f"{date_text}T00:00:00Z/{date_text}T23:59:59Z",
        "limit": 20,
    }
    response = requests.post(
        STAC_SEARCH,
        json=payload,
        timeout=(15, 90),
        headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"},
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    dual = [
        feature
        for feature in features
        if "vv" in feature.get("assets", {}) and "vh" in feature.get("assets", {})
    ]
    if len(dual) != 1:
        raise RuntimeError(
            f"Expected exactly one dual-pol RTC item for {date_text}; found {len(dual)}"
        )
    return dual[0]


def sign_href(href: str) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(href)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) < 2:
        raise RuntimeError(f"Cannot identify Azure container for {href}")
    account = parsed.netloc.split(".")[0]
    container = parts[0]
    token_url = f"{SAS_ENDPOINT}/{account}/{container}"
    response = requests.get(
        token_url,
        timeout=(15, 60),
        headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("token")
    if not token:
        raise RuntimeError(f"No SAS token returned for {account}/{container}")
    separator = "&" if "?" in href else "?"
    return href + separator + token, {
        "token_endpoint": token_url,
        "expiry": payload.get("msft:expiry"),
    }


def dataset_metadata(src: rasterio.io.DatasetReader) -> dict[str, Any]:
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


def reproject_signed_asset(
    signed_url: str,
    dst_transform: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    destination = np.full((height, width), NODATA, dtype="float32")
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(signed_url) as src:
            metadata = dataset_metadata(src)
            if src.count < 1 or src.crs is None:
                raise RuntimeError("RTC source is missing a raster band or CRS")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=DST_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
                num_threads=2,
            )
    valid = np.isfinite(destination) & (destination != NODATA)
    if not valid.any():
        raise RuntimeError("Reprojected RTC asset contains no valid pixels in the AOI")
    return destination, metadata


def read_asset_with_retry(
    href: str,
    dst_transform: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    for attempt in range(1, MAX_ASSET_ATTEMPTS + 1):
        try:
            signed_url, token_metadata = sign_href(href)
            array, source_metadata = reproject_signed_asset(
                signed_url, dst_transform, width, height
            )
            return array, source_metadata, token_metadata, errors
        except Exception as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            if attempt < MAX_ASSET_ATTEMPTS:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"RTC asset failed after {MAX_ASSET_ATTEMPTS} attempts: {errors[-1]['error']}"
    )


def write_raster(
    path: Path,
    array: np.ndarray,
    transform: Any,
    *,
    dtype: str,
    nodata: float | int,
    tags: dict[str, str],
) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
            dst.update_tags(**tags)
            overview_resampling = (
                Resampling.average if dtype.startswith("float") else Resampling.nearest
            )
            dst.build_overviews([2, 4, 8, 16], overview_resampling)
            dst.update_tags(
                ns="rio_overview",
                resampling="average" if dtype.startswith("float") else "nearest",
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_stats(array: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    values = array[valid]
    if values.size == 0:
        raise RuntimeError("No valid values available for statistics")
    return float(values.min()), float(values.max())


def main() -> None:
    transform, width, height, bounds = aligned_grid()
    phase_reports: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []
    progress: dict[str, Any] = {
        "status": "running",
        "grid": {
            "crs": str(DST_CRS),
            "resolution_m": RESOLUTION,
            "width": width,
            "height": height,
            "bounds": bounds,
        },
        "completed_assets": [],
        "phase_reports": {},
    }
    write_json(OUT / "rtc_extraction_progress.json", progress)

    try:
        for phase, date_text in PHASES.items():
            item = get_item(date_text)
            item_id = item["id"]
            phase_dir = OUT / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, np.ndarray] = {}
            assets_report: dict[str, Any] = {}

            for polarization in ("vv", "vh"):
                asset = item["assets"][polarization]
                array, source_metadata, token_metadata, retry_errors = (
                    read_asset_with_retry(asset["href"], transform, width, height)
                )
                arrays[polarization] = array
                output_path = phase_dir / f"{phase}_{polarization}_rtc_30m.tif"
                write_raster(
                    output_path,
                    array,
                    transform,
                    dtype="float32",
                    nodata=NODATA,
                    tags={
                        "source_collection": COLLECTION,
                        "source_item_id": item_id,
                        "source_asset": polarization,
                        "phase": phase,
                        "acquisition_date": date_text,
                        "processing_note": (
                            "Reprojected from Microsoft Planetary Computer Sentinel-1 "
                            "RTC COG to EPSG:32610 at 30 m"
                        ),
                    },
                )
                digest = sha256(output_path)
                assets_report[polarization] = {
                    "source_href": asset["href"],
                    "token_metadata": token_metadata,
                    "source_metadata": source_metadata,
                    "retry_errors_before_success": retry_errors,
                    "output": str(output_path.relative_to(ROOT)),
                    "sha256": digest,
                }
                manifest_rows.append(
                    {
                        "phase": phase,
                        "date": date_text,
                        "layer": polarization,
                        "path": str(output_path.relative_to(ROOT)),
                        "sha256": digest,
                        "bytes": output_path.stat().st_size,
                    }
                )
                progress["completed_assets"].append(
                    {
                        "phase": phase,
                        "date": date_text,
                        "layer": polarization,
                        "path": str(output_path.relative_to(ROOT)),
                        "sha256": digest,
                    }
                )
                write_json(OUT / "rtc_extraction_progress.json", progress)

            valid = (
                np.isfinite(arrays["vv"])
                & np.isfinite(arrays["vh"])
                & (arrays["vv"] != NODATA)
                & (arrays["vh"] != NODATA)
            )
            mask_path = phase_dir / f"{phase}_valid_mask_30m.tif"
            write_raster(
                mask_path,
                valid.astype("uint8"),
                transform,
                dtype="uint8",
                nodata=0,
                tags={
                    "phase": phase,
                    "acquisition_date": date_text,
                    "definition": (
                        "1 where both reprojected VV and VH RTC pixels are valid; "
                        "0 otherwise"
                    ),
                },
            )
            mask_digest = sha256(mask_path)
            manifest_rows.append(
                {
                    "phase": phase,
                    "date": date_text,
                    "layer": "valid_mask",
                    "path": str(mask_path.relative_to(ROOT)),
                    "sha256": mask_digest,
                    "bytes": mask_path.stat().st_size,
                }
            )
            vv_min, vv_max = valid_stats(arrays["vv"], valid)
            vh_min, vh_max = valid_stats(arrays["vh"], valid)
            phase_report = {
                "date": date_text,
                "item_id": item_id,
                "platform": item.get("properties", {}).get("platform"),
                "orbit_state": item.get("properties", {}).get("sat:orbit_state"),
                "relative_orbit": item.get("properties", {}).get(
                    "sat:relative_orbit"
                ),
                "assets": assets_report,
                "valid_pixels": int(valid.sum()),
                "total_pixels": int(valid.size),
                "valid_fraction": float(valid.mean()),
                "vv_valid_min": vv_min,
                "vv_valid_max": vv_max,
                "vh_valid_min": vh_min,
                "vh_valid_max": vh_max,
            }
            phase_reports[phase] = phase_report
            progress["phase_reports"][phase] = phase_report
            write_json(OUT / "rtc_extraction_progress.json", progress)

        common = np.ones((height, width), dtype=bool)
        for phase in PHASES:
            with rasterio.open(
                OUT / phase / f"{phase}_valid_mask_30m.tif"
            ) as src:
                common &= src.read(1).astype(bool)
        if not common.any():
            raise RuntimeError("Five-phase common valid mask contains no valid pixels")

        common_path = OUT / "common_valid_mask_30m.tif"
        write_raster(
            common_path,
            common.astype("uint8"),
            transform,
            dtype="uint8",
            nodata=0,
            tags={
                "definition": (
                    "Intersection of five phase-specific dual-polarization valid masks"
                ),
                "phase_count": str(len(PHASES)),
            },
        )
        common_digest = sha256(common_path)
        manifest_rows.append(
            {
                "phase": "all",
                "date": "",
                "layer": "common_valid_mask",
                "path": str(common_path.relative_to(ROOT)),
                "sha256": common_digest,
                "bytes": common_path.stat().st_size,
            }
        )

        with (OUT / "sha256_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["phase", "date", "layer", "path", "sha256", "bytes"],
            )
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
            "all_phase_rasters_complete": len(phase_reports) == 5
            and all(v["valid_pixels"] > 0 for v in phase_reports.values()),
            "scientific_caveat": (
                "These are Planetary Computer Sentinel-1 RTC assets, not the "
                "frozen original HPC OPERA RTC files. They are a transparent "
                "public-source reconstruction and must not be described as exact "
                "replicas of the HPC products."
            ),
        }
        write_json(OUT / "rtc_extraction_status.json", report)
        progress["status"] = "completed"
        progress["common_valid_fraction"] = report["common_valid_fraction"]
        write_json(OUT / "rtc_extraction_progress.json", progress)
        (OUT / "rtc_extraction_failure.json").unlink(missing_ok=True)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "progress": progress,
        }
        write_json(OUT / "rtc_extraction_failure.json", failure)
        progress["status"] = "failed"
        progress["error"] = failure["error"]
        write_json(OUT / "rtc_extraction_progress.json", progress)
        raise


if __name__ == "__main__":
    main()
