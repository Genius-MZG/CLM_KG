from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent / "recovered_data"
RTC = ROOT / "planetary_computer_rtc"
MODEL = ROOT / "model_inputs"
OUT = MODEL / "qa"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = ("baseline", "rising", "peak", "early_recession", "late_recession")
EXPECTED_CRS = "EPSG:32610"
EXPECTED_RESOLUTION = 30.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = {
            "crs": str(src.crs),
            "shape": [src.height, src.width],
            "transform": list(src.transform),
            "bounds": list(src.bounds),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "res": [abs(src.transform.a), abs(src.transform.e)],
        }
    return arr, meta


def finite_stats(arr: np.ndarray, nodata: float | int | None) -> dict:
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    values = arr[valid]
    if values.size == 0:
        return {"valid_pixels": 0, "valid_fraction": 0.0}
    q = np.percentile(values.astype("float64"), [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "valid_pixels": int(values.size),
        "valid_fraction": float(values.size / arr.size),
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p25": float(q[3]),
        "median": float(q[4]),
        "p75": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "max": float(q[8]),
    }


def main() -> None:
    required = {
        "jrc_occurrence": MODEL / "jrc_gsw_v1_5_occurrence_30m.tif",
        "jrc_seasonality": MODEL / "jrc_gsw_v1_5_seasonality_30m.tif",
        "jrc_extent": MODEL / "jrc_gsw_v1_5_extent_30m.tif",
        "cop_dem": MODEL / "cop_dem_glo30_30m.tif",
        "common_valid": RTC / "common_valid_mask_30m.tif",
    }
    for phase in PHASES:
        required[f"{phase}_vv"] = RTC / phase / f"{phase}_vv_rtc_30m.tif"
        required[f"{phase}_vh"] = RTC / phase / f"{phase}_vh_rtc_30m.tif"

    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing recovered model inputs: " + ", ".join(missing))

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for key, path in required.items():
        arr, meta = read(path)
        arrays[key] = arr
        metadata[key] = meta
        stats[key] = finite_stats(arr, meta["nodata"])
        stats[key]["sha256"] = sha256(path)
        stats[key]["bytes"] = path.stat().st_size

    ref = metadata["common_valid"]
    grid_checks = {}
    for key, meta in metadata.items():
        grid_checks[key] = {
            "crs_ok": meta["crs"] == EXPECTED_CRS,
            "shape_ok": meta["shape"] == ref["shape"],
            "transform_ok": np.allclose(meta["transform"], ref["transform"], atol=1e-9),
            "resolution_ok": np.allclose(meta["res"], [EXPECTED_RESOLUTION, EXPECTED_RESOLUTION], atol=1e-9),
        }

    common = arrays["common_valid"] == 1
    jrc_domain = {
        "occurrence_unique": {str(int(v)): int(c) for v, c in zip(*np.unique(arrays["jrc_occurrence"], return_counts=True))},
        "seasonality_unique": {str(int(v)): int(c) for v, c in zip(*np.unique(arrays["jrc_seasonality"], return_counts=True))},
        "extent_unique": {str(int(v)): int(c) for v, c in zip(*np.unique(arrays["jrc_extent"], return_counts=True))},
        "occurrence_range_ok": bool(arrays["jrc_occurrence"].min() >= 0 and arrays["jrc_occurrence"].max() <= 100),
        "seasonality_range_ok": bool(arrays["jrc_seasonality"].min() >= 0 and arrays["jrc_seasonality"].max() <= 12),
        "extent_range_ok": bool(set(np.unique(arrays["jrc_extent"]).tolist()).issubset({0, 1})),
    }

    rtc_checks = {}
    for phase in PHASES:
        vv = arrays[f"{phase}_vv"]
        vh = arrays[f"{phase}_vh"]
        valid = common & np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
        rtc_checks[phase] = {
            "common_positive_dual_pol_pixels": int(valid.sum()),
            "common_positive_dual_pol_fraction": float(valid.mean()),
            "vv_positive_on_common": bool(np.all(vv[common] > 0)),
            "vh_positive_on_common": bool(np.all(vh[common] > 0)),
        }

    dem = arrays["cop_dem"].astype("float64")
    dem_valid = np.isfinite(dem)
    dem_checks = {
        "finite_fraction": float(dem_valid.mean()),
        "min_m": float(np.nanmin(dem)),
        "max_m": float(np.nanmax(dem)),
        "range_plausible_for_aoi": bool(np.nanmin(dem) > -100 and np.nanmax(dem) < 5000),
        "scientific_caveat": "Copernicus GLO-30 is a DSM, not a bare-earth DEM.",
    }

    qa_passed = bool(
        all(all(v.values()) for v in grid_checks.values())
        and jrc_domain["occurrence_range_ok"]
        and jrc_domain["seasonality_range_ok"]
        and jrc_domain["extent_range_ok"]
        and all(v["common_positive_dual_pol_pixels"] > 0 for v in rtc_checks.values())
        and dem_checks["range_plausible_for_aoi"]
    )

    report = {
        "qa_passed": qa_passed,
        "reference_grid": ref,
        "grid_checks": grid_checks,
        "raster_statistics": stats,
        "jrc_domain_checks": jrc_domain,
        "rtc_common_domain_checks": rtc_checks,
        "dem_checks": dem_checks,
        "model_recompute_allowed": False,
        "model_recompute_blockers": [
            "Exact frozen JRC label thresholds and exclusion rules remain unavailable.",
            "Exact original nine-feature definitions and preprocessing constants remain unavailable.",
            "Recovered Sentinel-1 RTC is a public reconstruction, not the frozen original HPC stack.",
        ],
        "scientific_constraint": "This audit verifies real raster integrity and alignment only. It does not create labels, train a model, reproduce metrics, or generate model figures.",
    }
    (OUT / "model_input_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (OUT / "sha256_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "path", "bytes", "sha256"])
        writer.writeheader()
        for key, path in sorted(required.items()):
            writer.writerow({"name": key, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not qa_passed:
        raise RuntimeError("Recovered model-input QA failed")


if __name__ == "__main__":
    main()
