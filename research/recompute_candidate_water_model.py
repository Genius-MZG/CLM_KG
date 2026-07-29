from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import transform_geom
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent / "recovered_data"
RTC = ROOT / "planetary_computer_rtc"
MODEL = ROOT / "model_inputs"
OUT = ROOT / "candidate_model"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = [
    ("baseline", "2017-11-01"),
    ("rising", "2017-11-13"),
    ("peak", "2017-11-25"),
    ("early_recession", "2017-12-07"),
    ("late_recession", "2017-12-19"),
]
FEATURE_NAMES = ["VV_dB", "VH_dB", "VV_minus_VH_dB", "DEM_m", "slope_deg"]
THRESHOLDS = [0.35, 0.50, 0.65]
DOWNSTREAM_GRDC_NO = 4146080
NODATA = -9999.0
RANDOM_SEED = 20260728


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any], Any]:
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile.copy(), src.transform


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tif(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    dtype: str,
    nodata: float | int,
    tags: dict[str, str],
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
        predictor=3 if dtype.startswith("float") else 2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with rasterio.open(temporary, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.update_tags(**tags)
        dst.build_overviews(
            [2, 4, 8, 16],
            rasterio.enums.Resampling.average if dtype.startswith("float") else rasterio.enums.Resampling.nearest,
        )
        dst.update_tags(
            ns="rio_overview",
            resampling="average" if dtype.startswith("float") else "nearest",
        )
    temporary.replace(path)


def make_features(vv: np.ndarray, vh: np.ndarray, dem: np.ndarray, slope: np.ndarray) -> np.ndarray:
    vv_db = 10.0 * np.log10(np.clip(vv, 1.0e-8, None))
    vh_db = 10.0 * np.log10(np.clip(vh, 1.0e-8, None))
    return np.stack([vv_db, vh_db, vv_db - vh_db, dem, slope], axis=-1).astype("float32")


def downstream_basin_mask(profile: dict[str, Any], transform: Any, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    geometry_path = ROOT / "grdc_pair_basins_arcgis.geojson"
    if not geometry_path.exists():
        raise FileNotFoundError(
            "Official GRDC basin geometry is required for the candidate model domain; no substitute geometry is permitted"
        )
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    matches = [
        feature
        for feature in payload.get("features", [])
        if int(float(feature.get("properties", {}).get("grdc_no", -1))) == DOWNSTREAM_GRDC_NO
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official downstream GRDC basin feature; found {len(matches)}")
    feature = matches[0]
    source_geometry = feature.get("geometry")
    if not source_geometry:
        raise RuntimeError("Official downstream GRDC basin feature has no geometry")
    projected = transform_geom("EPSG:4326", profile["crs"], source_geometry, precision=3)
    mask = rasterize(
        [(projected, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)
    if not mask.any():
        raise RuntimeError("Rasterized downstream GRDC basin is empty on the RTC grid")
    provenance = {
        "source_file": str(geometry_path.relative_to(ROOT)),
        "source_sha256": sha256(geometry_path),
        "downstream_grdc_no": DOWNSTREAM_GRDC_NO,
        "river": feature.get("properties", {}).get("river"),
        "station": feature.get("properties", {}).get("station"),
        "area_calc_km2": feature.get("properties", {}).get("area_calc"),
        "rasterized_pixels": int(mask.sum()),
        "rasterized_area_km2": float(mask.sum() * 0.0009),
        "rasterization_rule": "official GRDC polygon, pixel center rule (all_touched=False)",
    }
    return mask, provenance


def main() -> None:
    occurrence, profile, transform = read_raster(MODEL / "jrc_gsw_v1_5_occurrence_30m.tif")
    seasonality, _, _ = read_raster(MODEL / "jrc_gsw_v1_5_seasonality_30m.tif")
    extent, _, _ = read_raster(MODEL / "jrc_gsw_v1_5_extent_30m.tif")
    dem, _, _ = read_raster(MODEL / "cop_dem_glo30_30m.tif")
    common, _, _ = read_raster(RTC / "common_valid_mask_30m.tif")
    baseline_vv, _, _ = read_raster(RTC / "baseline/baseline_vv_rtc_30m.tif")
    baseline_vh, _, _ = read_raster(RTC / "baseline/baseline_vh_rtc_30m.tif")

    shape = occurrence.shape
    arrays = [seasonality, extent, dem, common, baseline_vv, baseline_vh]
    if any(array.shape != shape for array in arrays):
        raise RuntimeError("Candidate-model inputs do not share one grid")
    if str(profile.get("crs")) != "EPSG:32610":
        raise RuntimeError(f"Unexpected model grid CRS: {profile.get('crs')}")

    basin, basin_provenance = downstream_basin_mask(profile, transform, shape)
    domain = basin & (common == 1)
    if not domain.any():
        raise RuntimeError("Downstream-basin common RTC domain is empty")

    write_tif(
        OUT / "downstream_basin_common_domain_30m.tif",
        domain.astype("uint8"),
        profile,
        "uint8",
        0,
        {
            "definition": "Intersection of official GRDC 4146080 basin and five-phase common RTC data domain",
            "source_geometry": "grdc_pair_basins_arcgis.geojson",
            "source_geometry_sha256": basin_provenance["source_sha256"],
        },
    )

    gradient_y, gradient_x = np.gradient(dem.astype("float64"), 30.0, 30.0)
    slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y))).astype("float32")

    water = (
        (seasonality == 12)
        & (occurrence >= 95)
        & (extent == 1)
        & domain
    )
    land = (occurrence == 0) & (extent == 0) & domain
    if not water.any() or not land.any():
        raise RuntimeError("Candidate water/land labels are empty inside the official downstream basin")

    rows = np.arange(0, shape[0], 3)
    cols = np.arange(0, shape[1], 3)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    sample_rows = row_grid.ravel()
    sample_cols = col_grid.ravel()
    keep = water[sample_rows, sample_cols] | land[sample_rows, sample_cols]
    sample_rows = sample_rows[keep]
    sample_cols = sample_cols[keep]
    labels = water[sample_rows, sample_cols].astype("uint8")

    x_coordinates, y_coordinates = xy(transform, sample_rows, sample_cols, offset="center")
    groups = np.asarray(
        [f"{int(x_coord // 10000)}_{int(y_coord // 10000)}" for x_coord, y_coord in zip(x_coordinates, y_coordinates)]
    )

    baseline_features = make_features(baseline_vv, baseline_vh, dem, slope)
    sampled_features = baseline_features[sample_rows, sample_cols]
    finite = np.all(np.isfinite(sampled_features), axis=1)
    sampled_features = sampled_features[finite]
    labels = labels[finite]
    groups = groups[finite]
    sample_rows = sample_rows[finite]
    sample_cols = sample_cols[finite]
    x_coordinates = np.asarray(x_coordinates)[finite]
    y_coordinates = np.asarray(y_coordinates)[finite]

    rng = np.random.default_rng(RANDOM_SEED)
    selected: list[int] = []
    dual_class_groups: list[str] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        water_indices = indices[labels[indices] == 1]
        land_indices = indices[labels[indices] == 0]
        if len(water_indices) == 0 or len(land_indices) == 0:
            continue
        dual_class_groups.append(str(group))
        selected.extend(water_indices.tolist())
        selected_land_count = min(len(land_indices), max(100, 4 * len(water_indices)))
        selected.extend(rng.choice(land_indices, size=selected_land_count, replace=False).tolist())

    selected_indices = np.asarray(sorted(set(selected)), dtype="int64")
    if selected_indices.size == 0:
        raise RuntimeError("No dual-class spatial blocks were available for candidate-model fitting")

    X = sampled_features[selected_indices]
    y = labels[selected_indices]
    group_ids = groups[selected_indices]
    selected_rows = sample_rows[selected_indices]
    selected_cols = sample_cols[selected_indices]
    selected_x = x_coordinates[selected_indices]
    selected_y = y_coordinates[selected_indices]

    unique_groups = np.unique(group_ids)
    if len(unique_groups) < 5:
        raise RuntimeError(f"Fewer than five dual-class spatial groups: {len(unique_groups)}")

    cross_validator = GroupKFold(n_splits=5)
    oof_probability = np.full(len(y), np.nan, dtype="float64")
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(cross_validator.split(X, y, group_ids), 1):
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        )
        candidate.fit(X[train_indices], y[train_indices])
        probability = candidate.predict_proba(X[test_indices])[:, 1]
        oof_probability[test_indices] = probability
        fold_rows.append(
            {
                "fold": fold,
                "train_samples": int(len(train_indices)),
                "test_samples": int(len(test_indices)),
                "test_groups": int(len(np.unique(group_ids[test_indices]))),
                "water_test": int(y[test_indices].sum()),
                "land_test": int((y[test_indices] == 0).sum()),
                "auc": float(roc_auc_score(y[test_indices], probability)),
                "brier": float(brier_score_loss(y[test_indices], probability)),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("Spatial cross-validation left missing out-of-fold probabilities")

    pd.DataFrame(fold_rows).to_csv(OUT / "candidate_spatial_cv_folds.csv", index=False)
    pd.DataFrame(
        {
            "label": y,
            "probability": oof_probability,
            "block_id": group_ids,
            "row": selected_rows,
            "col": selected_cols,
            "x_utm10n_m": selected_x,
            "y_utm10n_m": selected_y,
        }
    ).to_csv(OUT / "candidate_oof_predictions.csv", index=False)

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )
    final_model.fit(X, y)
    scaler: StandardScaler = final_model.named_steps["standardscaler"]
    logistic: LogisticRegression = final_model.named_steps["logisticregression"]
    pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "training_mean": scaler.mean_,
            "training_scale": scaler.scale_,
            "standardized_logistic_coefficient": logistic.coef_[0],
        }
    ).to_csv(OUT / "candidate_model_coefficients.csv", index=False)

    area_rows: list[dict[str, Any]] = []
    permanent_water_seed = water
    connectivity = ndimage.generate_binary_structure(2, 2)
    for phase, date_text in PHASES:
        vv, _, _ = read_raster(RTC / phase / f"{phase}_vv_rtc_30m.tif")
        vh, _, _ = read_raster(RTC / phase / f"{phase}_vh_rtc_30m.tif")
        features = make_features(vv, vh, dem, slope)
        valid = domain & np.all(np.isfinite(features), axis=-1) & (vv > 0) & (vh > 0)
        probability = np.full(shape, NODATA, dtype="float32")
        probability[valid] = final_model.predict_proba(features[valid])[:, 1].astype("float32")
        write_tif(
            OUT / f"{phase}_candidate_water_probability_30m.tif",
            probability,
            profile,
            "float32",
            NODATA,
            {
                "phase": phase,
                "acquisition_date": date_text,
                "model_scope": "transparent public-source five-feature candidate reconstruction",
                "domain": "official downstream GRDC 4146080 basin intersected with common RTC data domain",
                "prohibited_interpretation": "not the frozen original HPC nine-feature model",
            },
        )

        for threshold in THRESHOLDS:
            binary = (probability >= threshold) & valid
            components, component_count = ndimage.label(binary, structure=connectivity)
            touching_components = np.unique(components[permanent_water_seed & (components > 0)])
            connected = (
                np.isin(components, touching_components)
                if touching_components.size
                else np.zeros_like(binary, dtype=bool)
            )
            output_name = f"{phase}_connected_water_t{int(round(threshold * 100)):02d}_30m.tif"
            write_tif(
                OUT / output_name,
                connected.astype("uint8"),
                profile,
                "uint8",
                0,
                {
                    "phase": phase,
                    "acquisition_date": date_text,
                    "probability_threshold": f"{threshold:.2f}",
                    "connectivity": "8-neighbour",
                    "seed_rule": "components intersecting candidate JRC permanent-water labels inside official downstream basin",
                    "domain": "official downstream GRDC 4146080 basin intersected with common RTC data domain",
                },
            )
            area_rows.append(
                {
                    "phase": phase,
                    "date": date_text,
                    "threshold": threshold,
                    "connected_pixels": int(connected.sum()),
                    "connected_area_km2": float(connected.sum() * 0.0009),
                    "components_before_seed_filter": int(component_count),
                    "seeded_components": int(len(touching_components)),
                }
            )

    area_table = pd.DataFrame(area_rows)
    area_table.to_csv(OUT / "candidate_connected_water_area.csv", index=False)

    main_area = area_table[area_table["threshold"] == 0.50].copy()
    main_maximum = main_area.loc[main_area["connected_area_km2"].idxmax()]
    report = {
        "scope": "transparent public-source candidate reconstruction; not frozen HPC model",
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "label_rule": "water: JRC seasonality=12, occurrence>=95, extent=1; land: occurrence=0, extent=0",
        "training_phase": "baseline (2017-11-01)",
        "sampling_m": 90,
        "block_m": 10000,
        "cv_folds": 5,
        "training_samples": int(len(y)),
        "water_samples": int(y.sum()),
        "land_samples": int((y == 0).sum()),
        "spatial_groups": int(len(unique_groups)),
        "dual_class_groups": dual_class_groups,
        "oof_auc": float(roc_auc_score(y, oof_probability)),
        "oof_brier": float(brier_score_loss(y, oof_probability)),
        "thresholds": THRESHOLDS,
        "domain": basin_provenance,
        "domain_common_pixels": int(domain.sum()),
        "domain_common_area_km2": float(domain.sum() * 0.0009),
        "connectivity_rule": "8-neighbour components intersecting candidate permanent-water seeds inside the official downstream basin",
        "main_threshold_maximum_phase": str(main_maximum["phase"]),
        "main_threshold_maximum_area_km2": float(main_maximum["connected_area_km2"]),
        "qa_passed": bool(
            len(area_rows) == 15
            and len(FEATURE_NAMES) == 5
            and len(unique_groups) >= 5
            and not np.isnan(oof_probability).any()
            and domain.any()
            and all(area_table["connected_area_km2"] >= 0)
        ),
        "scientific_caveats": [
            "Recovered Sentinel-1 RTC is a public Planetary Computer reconstruction, not the frozen original HPC RTC stack.",
            "The exact original nine-feature formulas, preprocessing constants and label exclusions remain unavailable.",
            "The connectivity seed uses candidate JRC permanent-water labels throughout the official downstream basin; it is not the original SWORD-plus-permanent-water seed implementation.",
            "Candidate areas and metrics must not be substituted for the verified original HPC values.",
        ],
        "prohibited_interpretation": "Do not report as reproducing the original nine-feature HPC model or its metrics, probability surfaces, connected-water masks or areas.",
    }
    (OUT / "candidate_model_status.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_rows = []
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            manifest_rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(OUT / "sha256_manifest.csv", index=False)

    if not report["qa_passed"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
