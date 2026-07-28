from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent / "recovered_data"
RIVER = ROOT / "river_geometry"
SOURCE = RIVER / "sword_NA_v17c_reaches.parquet"
COMMON_MASK = ROOT / "planetary_computer_rtc" / "common_valid_mask_30m.tif"
STATIONS = ROOT / "grdc_pair_station_metadata.csv"
OUT_GEOJSON = RIVER / "sword_event_aoi_reaches.geojson"
OUT_QA = RIVER / "sword_event_aoi_qa.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    required = [SOURCE, COMMON_MASK, STATIONS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing verified SWORD subset inputs: " + ", ".join(missing))

    reaches = gpd.read_parquet(SOURCE)
    if reaches.empty or reaches.geometry.isna().all():
        raise RuntimeError("Official North America SWORD reaches file contains no usable geometry")
    if reaches.crs is None:
        raise RuntimeError("Official SWORD GeoParquet has no CRS metadata")

    with rasterio.open(COMMON_MASK) as dataset:
        mask_crs = dataset.crs
        bounds = dataset.bounds
        width = dataset.width
        height = dataset.height
        transform = dataset.transform

    if mask_crs is None:
        raise RuntimeError("Common-valid-mask raster has no CRS")

    transformer = Transformer.from_crs(mask_crs, reaches.crs, always_xy=True)
    minx, miny = transformer.transform(bounds.left, bounds.bottom)
    maxx, maxy = transformer.transform(bounds.right, bounds.top)
    aoi = box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

    subset = reaches.loc[reaches.geometry.intersects(aoi)].copy()
    subset = subset.loc[~subset.geometry.is_empty & subset.geometry.notna()].copy()
    if subset.empty:
        raise RuntimeError("No official SWORD reaches intersect the fixed RTC common-valid AOI")

    subset = subset.to_crs("EPSG:4326")
    keep_candidates = [
        "reach_id", "x", "y", "facc", "dist_out", "rch_id_up", "rch_id_dn",
        "n_rch_up", "n_rch_dn", "river_name", "width", "width_var", "geometry",
    ]
    keep = [column for column in keep_candidates if column in subset.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    subset = subset[keep]
    subset.to_file(OUT_GEOJSON, driver="GeoJSON")

    stations = pd.read_csv(STATIONS)
    lon_col = "long_pp" if "long_pp" in stations.columns else "long_org"
    lat_col = "lat_pp" if "lat_pp" in stations.columns else "lat_org"
    station_points = [
        {"grdc_no": int(float(row["grdc_no"])), "longitude": float(row[lon_col]), "latitude": float(row[lat_col])}
        for _, row in stations.iterrows()
    ]

    geometry_types = subset.geometry.geom_type.value_counts().to_dict()
    invalid_count = int((~subset.geometry.is_valid).sum())
    qa = {
        "source": "Official SWORD v17c North America reaches GeoParquet",
        "source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "source_crs": str(reaches.crs),
        "source_reach_count": int(len(reaches)),
        "fixed_aoi_source": COMMON_MASK.name,
        "fixed_aoi_crs": str(mask_crs),
        "fixed_aoi_bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "fixed_aoi_shape": [height, width],
        "fixed_aoi_transform": list(transform)[:6],
        "subset_reach_count": int(len(subset)),
        "subset_geometry_types": geometry_types,
        "subset_invalid_geometry_count": invalid_count,
        "subset_bounds_wgs84": list(subset.total_bounds),
        "station_points_wgs84": station_points,
        "output": OUT_GEOJSON.name,
        "output_sha256": sha256(OUT_GEOJSON),
        "geometry_recovered": True,
        "scientific_scope": "All official SWORD reaches intersecting the fixed five-phase RTC common-valid AOI; no synthetic or hand-drawn centerline.",
    }
    OUT_QA.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
