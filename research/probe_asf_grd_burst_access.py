from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "recovered_data"
OUT = ROOT / "source_access_probe"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_DATES = ["2017-11-01", "2017-11-13", "2017-11-25", "2017-12-07", "2017-12-19"]
RANGE_BYTES = 65536
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20
MAX_WORKERS = 8


def probe(url: str) -> dict[str, object]:
    headers = {
        "Range": f"bytes=0-{RANGE_BYTES - 1}",
        "User-Agent": "NorthAmericaFloodplainResearch/1.0",
        "Accept-Encoding": "identity",
    }
    try:
        with requests.get(
            url,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
            stream=True,
        ) as r:
            body = bytearray()
            for chunk in r.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                need = RANGE_BYTES - len(body)
                body.extend(chunk[:need])
                if len(body) >= RANGE_BYTES:
                    break
            return {
                "status_code": r.status_code,
                "ok": r.status_code in (200, 206) and len(body) > 0,
                "final_url": r.url,
                "content_type": r.headers.get("content-type"),
                "content_range": r.headers.get("content-range"),
                "content_length_header": r.headers.get("content-length"),
                "accept_ranges": r.headers.get("accept-ranges"),
                "bytes_received": len(body),
                "redirect_history": [x.status_code for x in r.history],
                "prefix_hex": bytes(body[:16]).hex(),
                "range_honored": r.status_code == 206 or r.headers.get("content-range") is not None,
            }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def selected_targets(df: pd.DataFrame) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for date in TARGET_DATES:
        daily = df[df["start_time"].astype(str).str.startswith(date)].copy()
        grd = daily[
            (daily["processing_level"] == "GRD_HD")
            & (daily["scene_name"].str.contains("_GRDH_1SDV_", na=False))
        ]
        bursts = daily[daily["processing_level"] == "BURST"]
        selected: list[tuple[str, pd.Series]] = []
        if not grd.empty:
            selected.append(("GRD_HD", grd.iloc[0]))
        for pol in ["VV", "VH"]:
            rows = bursts[bursts["polarization"] == pol]
            if not rows.empty:
                selected.append((f"BURST_{pol}", rows.iloc[0]))
        for kind, row in selected:
            targets.append(
                {
                    "date": date,
                    "kind": kind,
                    "scene_name": str(row["scene_name"]),
                    "url": str(row["url"]),
                }
            )
    return targets


def main() -> None:
    df = pd.read_csv(ROOT / "asf_sentinel1_candidates.csv")
    targets = selected_targets(df)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(probe, target["url"]): target for target in targets}
        for future in as_completed(future_map):
            target = future_map[future]
            result = future.result()
            records.append({**target, **result})

    records.sort(key=lambda r: (str(r["date"]), str(r["kind"])))
    pd.DataFrame(records).to_csv(OUT / "asf_source_access_probe.csv", index=False)
    report = {
        "probe_type": "bounded unauthenticated HTTP range GET",
        "range_bytes": RANGE_BYTES,
        "connect_timeout_seconds": CONNECT_TIMEOUT,
        "read_timeout_seconds": READ_TIMEOUT,
        "max_workers": MAX_WORKERS,
        "records": records,
        "all_grd_accessible": bool(records) and all(
            bool(r.get("ok")) for r in records if r["kind"] == "GRD_HD"
        ),
        "all_bursts_accessible": bool(records) and all(
            bool(r.get("ok")) for r in records if str(r["kind"]).startswith("BURST_")
        ),
    }
    (OUT / "asf_source_access_probe.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
