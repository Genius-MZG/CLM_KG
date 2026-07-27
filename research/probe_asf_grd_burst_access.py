from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "source_access_probe"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_DATES = ["2017-11-01", "2017-11-13", "2017-11-25", "2017-12-07", "2017-12-19"]


def probe(url: str) -> dict[str, object]:
    try:
        r = requests.get(
            url,
            headers={"Range": "bytes=0-65535", "User-Agent": "NorthAmericaFloodplainResearch/1.0"},
            timeout=120,
            allow_redirects=True,
        )
        return {
            "status_code": r.status_code,
            "ok": r.status_code in (200, 206),
            "final_url": r.url,
            "content_type": r.headers.get("content-type"),
            "content_range": r.headers.get("content-range"),
            "bytes_received": len(r.content),
            "redirect_history": [x.status_code for x in r.history],
            "prefix_hex": r.content[:16].hex(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    df = pd.read_csv(ROOT / "asf_sentinel1_candidates.csv")
    records = []
    for date in TARGET_DATES:
        daily = df[df["start_time"].astype(str).str.startswith(date)].copy()
        grd = daily[(daily["processing_level"] == "GRD_HD") & (daily["scene_name"].str.contains("_GRDH_1SDV_", na=False))]
        bursts = daily[daily["processing_level"] == "BURST"]
        selected = []
        if not grd.empty:
            selected.append(("GRD_HD", grd.iloc[0]))
        for pol in ["VV", "VH"]:
            rows = bursts[bursts["polarization"] == pol]
            if not rows.empty:
                selected.append((f"BURST_{pol}", rows.iloc[0]))
        for kind, row in selected:
            result = probe(str(row["url"]))
            records.append({
                "date": date,
                "kind": kind,
                "scene_name": row["scene_name"],
                "url": row["url"],
                **result,
            })
    pd.DataFrame(records).to_csv(OUT / "asf_source_access_probe.csv", index=False)
    report = {
        "probe_type": "unauthenticated HTTP range GET",
        "records": records,
        "all_grd_accessible": all(r.get("ok") for r in records if r["kind"] == "GRD_HD"),
        "all_bursts_accessible": all(r.get("ok") for r in records if r["kind"].startswith("BURST_")),
    }
    (OUT / "asf_source_access_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
