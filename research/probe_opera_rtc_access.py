from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
CAT = ROOT / "opera_rtc_catalog"
OUT = CAT / "access_probe"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = CAT / "opera_rtc_expected_track_candidates.csv"
RANGE_BYTES = 65536
TIMEOUT = 90


def layer_urls(urls: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for url in urls:
        lower = url.lower()
        if lower.endswith("_vv.tif"):
            result["VV"] = url
        elif lower.endswith("_vh.tif"):
            result["VH"] = url
        elif lower.endswith("_mask.tif"):
            result["mask"] = url
        elif lower.endswith("_browse_low-res.png"):
            result["browse_low_res"] = url
        elif lower.endswith(".h5"):
            result["h5"] = url
    return result


def probe(url: str, *, save_path: Path | None = None) -> dict[str, object]:
    headers = {"Range": f"bytes=0-{RANGE_BYTES - 1}", "User-Agent": "NorthAmericaFloodplainResearch/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        body = r.content
        if save_path is not None and r.status_code in (200, 206) and body:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(body)
        return {
            "ok": r.status_code in (200, 206),
            "status_code": r.status_code,
            "final_url": r.url,
            "content_type": r.headers.get("content-type"),
            "content_length_header": r.headers.get("content-length"),
            "content_range": r.headers.get("content-range"),
            "accept_ranges": r.headers.get("accept-ranges"),
            "bytes_received": len(body),
            "redirect_history": [x.status_code for x in r.history],
            "first_16_bytes_hex": body[:16].hex(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    df = pd.read_csv(CANDIDATES)
    records: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for phase, group in df.groupby("phase", sort=False):
        row = group.sort_values(["burst_id", "product_name"]).iloc[0]
        urls = layer_urls(json.loads(row["urls_json"]))
        phase_results: dict[str, object] = {
            "phase_date": row["phase_date"],
            "product_name": row["product_name"],
            "burst_id": row["burst_id"],
            "layers": {},
        }
        for layer in ["VV", "VH", "mask", "h5", "browse_low_res"]:
            url = urls.get(layer)
            if not url:
                result = {"ok": False, "error": "URL missing"}
            else:
                suffix = ".png" if layer == "browse_low_res" else ".bin"
                save = OUT / f"{phase}_{layer}_prefix{suffix}"
                result = probe(url, save_path=save)
            phase_results["layers"][layer] = result
            records.append({
                "phase": phase,
                "phase_date": row["phase_date"],
                "product_name": row["product_name"],
                "burst_id": row["burst_id"],
                "layer": layer,
                "url": url,
                **result,
            })
        summaries[phase] = phase_results

    with (OUT / "opera_rtc_access_probe.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in records for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    all_raster_layers_accessible = all(
        bool(summaries[p]["layers"][layer].get("ok"))
        for p in summaries
        for layer in ["VV", "VH", "mask"]
    )
    report = {
        "probe_type": "unauthenticated HTTP range GET",
        "range_bytes": RANGE_BYTES,
        "phase_count": len(summaries),
        "all_raster_layers_accessible": all_raster_layers_accessible,
        "phase_results": summaries,
        "interpretation": (
            "Access probe only. Successful range responses establish that the published COG endpoints can be read "
            "from the workflow environment; they do not establish full-raster integrity, common coverage, or scientific suitability."
        ),
    }
    (OUT / "opera_rtc_access_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
