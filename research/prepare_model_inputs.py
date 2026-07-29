from __future__ import annotations

import csv
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
RTC = ROOT / "planetary_computer_rtc"
OUT = ROOT / "model_inputs"
OUT.mkdir(parents=True, exist_ok=True)

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_ENDPOINT = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
JRC_VERSION = "1.5-2024"
JRC_TILE = "130W_50N"
JRC_BASE = "https://storage.googleapis.com/water-world/download2024/VER1-5"
JRC_LAYERS = ("occurrence", "seasonality", "extent")
DEM_COLLECTION = "cop-dem-glo-30"
BBOX_WGS84 = (-122.48, 48.34, -121.62, 48.66)
UINT8_NODATA = 255
FLOAT_NODATA = -9999.0


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def target_grid() -> dict[str, Any]:
    common = RTC / "common_valid_mask_30m.tif"
    if not common.exists():
        raise FileNotFoundError(f"Missing target-grid raster: {common}")
    with rasterio.open(common) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": tuple(src.bounds),
        }


def sign_href(href: str) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(href)
    if ".blob.core.windows.net" not in parsed.netloc:
        return href, {"access_mode": "raw_public_href"}
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
    return href + sep + token, {
        "access_mode": "planetary_computer_sas",
        "token_endpoint": token_url,
        "expiry": payload.get("msft:expiry"),
    }


def gdal_env() -> rasterio.Env:
    return rasterio.Env(
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY="4",
        GDAL_HTTP_RETRY_DELAY="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="33554432",
    )


def reproject_one(
    href: str,
    grid: dict[str, Any],
    *,
    resampling: Resampling,
    dst_nodata: float | int,
    dst_dtype: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    last_error: str | None = None
    for attempt in range(1, 5):
        try:
            signed, access = sign_href(href)
            with gdal_env():
                with rasterio.open(signed) as src:
                    dst = np.full((grid["height"], grid["width"]), dst_nodata, dtype=dst_dtype)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=grid["transform"],
                        dst_crs=grid["crs"],
                        dst_nodata=dst_nodata,
                        resampling=resampling,
                        num_threads=2,
                    )
                    meta = {
                        "driver": src.driver,
                        "width": src.width,
                        "height": src.height,
                        "dtype": src.dtypes[0],
                        "crs": str(src.crs),
                        "nodata": src.nodata,
                        "bounds": tuple(src.bounds),
                        "access": access,
                        "attempt": attempt,
                    }
                    return dst, meta
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 4:
                break
            time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to read/reproject {href}: {last_error}")


def write_raster(
    path: Path,
    array: np.ndarray,
    grid: dict[str, Any],
    *,
    dtype: str,
    nodata: float | int,
    tags: dict[str, str],
    categorical: bool,
) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 2 if categorical else 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.update_tags(**tags)
        overview_resampling = Resampling.nearest if categorical else Resampling.average
        dst.build_overviews([2, 4, 8, 16], overview_resampling)
        dst.update_tags(ns="rio_overview", resampling="nearest" if categorical else "average")
    tmp.replace(path)


def recover_jrc(grid: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for layer in JRC_LAYERS:
        href = f"{JRC_BASE}/{layer}/{layer}_{JRC_TILE}_v1_5_2024.tif"
        array, src_meta = reproject_one(
            href,
            grid,
            resampling=Resampling.nearest,
            dst_nodata=UINT8_NODATA,
            dst_dtype="uint8",
        )
        valid = array != UINT8_NODATA
        if not valid.any():
            raise RuntimeError(f"JRC {layer} has no valid pixels in target grid")
        path = OUT / f"jrc_gsw_v1_5_{layer}_30m.tif"
        write_raster(
            path,
            array,
            grid,
            dtype="uint8",
            nodata=UINT8_NODATA,
            categorical=True,
            tags={
                "source": "JRC Global Surface Water v1.5 (2024 public tile)",
                "source_layer": layer,
                "source_tile": JRC_TILE,
                "source_href": href,
                "processing_note": "Nearest-neighbour reprojection to the Sentinel-1 RTC EPSG:32610 30 m grid",
            },
        )
        reports[layer] = {
            "source_href": href,
            "source_metadata": src_meta,
            "output": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "valid_pixels": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "valid_min": int(array[valid].min()),
            "valid_max": int(array[valid].max()),
        }
        manifest.append({
            "dataset": "jrc_gsw_v1_5",
            "layer": layer,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return reports, manifest


def search_dem_items() -> list[dict[str, Any]]:
    payload = {"collections": [DEM_COLLECTION], "bbox": list(BBOX_WGS84), "limit": 100}
    r = requests.post(STAC_SEARCH, json=payload, timeout=90, headers={"User-Agent": "NorthAmericaFloodplainResearch/1.0"})
    r.raise_for_status()
    items = r.json().get("features", [])
    items = [item for item in items if "data" in item.get("assets", {})]
    if not items:
        raise RuntimeError("No Copernicus DEM GLO-30 items found for AOI")
    return items


def recover_dem(grid: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    items = search_dem_items()
    mosaic = np.full((grid["height"], grid["width"]), FLOAT_NODATA, dtype="float32")
    item_reports: list[dict[str, Any]] = []
    for item in items:
        href = item["assets"]["data"]["href"]
        tile, src_meta = reproject_one(
            href,
            grid,
            resampling=Resampling.bilinear,
            dst_nodata=FLOAT_NODATA,
            dst_dtype="float32",
        )
        valid = np.isfinite(tile) & (tile != FLOAT_NODATA)
        mosaic[valid] = tile[valid]
        item_reports.append({"item_id": item.get("id"), "source_href": href, "source_metadata": src_meta, "contributed_pixels": int(valid.sum())})
    valid = np.isfinite(mosaic) & (mosaic != FLOAT_NODATA)
    if not valid.any():
        raise RuntimeError("Copernicus DEM mosaic has no valid pixels")
    path = OUT / "cop_dem_glo30_30m.tif"
    write_raster(
        path,
        mosaic,
        grid,
        dtype="float32",
        nodata=FLOAT_NODATA,
        categorical=False,
        tags={
            "source": "Microsoft Planetary Computer cop-dem-glo-30",
            "source_collection": DEM_COLLECTION,
            "processing_note": "Bilinear reprojection and mosaic to the Sentinel-1 RTC EPSG:32610 30 m grid; DSM, not bare-earth DEM",
        },
    )
    report = {
        "collection": DEM_COLLECTION,
        "item_count": len(items),
        "items": item_reports,
        "output": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "elevation_min_m": float(mosaic[valid].min()),
        "elevation_max_m": float(mosaic[valid].max()),
        "scientific_caveat": "Copernicus DEM GLO-30 is a DSM and includes vegetation and built structures; it is not a bare-earth terrain model.",
    }
    manifest = {
        "dataset": DEM_COLLECTION,
        "layer": "data",
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    return report, manifest


def main() -> None:
    failure_path = OUT / "model_input_failure.json"
    if failure_path.exists():
        failure_path.unlink()
    try:
        grid = target_grid()
        jrc_reports, manifest = recover_jrc(grid)
        dem_report, dem_manifest = recover_dem(grid)
        manifest.append(dem_manifest)
        manifest_path = OUT / "sha256_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "layer", "path", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(manifest)
        status = {
            "target_crs": str(grid["crs"]),
            "target_resolution_m": 30.0,
            "target_width": grid["width"],
            "target_height": grid["height"],
            "target_bounds": grid["bounds"],
            "jrc_version": JRC_VERSION,
            "jrc_tile": JRC_TILE,
            "jrc_layers": jrc_reports,
            "copernicus_dem": dem_report,
            "inputs_complete": len(jrc_reports) == len(JRC_LAYERS) and dem_report["valid_pixels"] > 0,
            "model_recompute_allowed": False,
            "model_recompute_blockers": [
                "The exact frozen JRC label thresholds and exclusion rules are not present in the recovered public files.",
                "The exact original nine-feature definitions and preprocessing constants are not present in the recovered public files.",
                "The Planetary Computer RTC reconstruction is not the frozen original HPC OPERA RTC stack.",
            ],
            "next_action": "Recover the frozen implementation policy and feature definitions before creating labels, fitting a classifier, or comparing metrics.",
        }
        write_json(OUT / "model_input_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
