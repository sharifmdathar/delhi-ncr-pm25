#!/usr/bin/env python
"""
Download every station file from OpenCity's "Delhi Hourly Air Quality Reports"
(2017-2023 hourly + 2024-25 15-minute, 39 stations -> 78 CSVs) and write a
stations.csv skeleton that pairs each station's two files.

Usage:
    python download_opencity.py                # download everything
    python download_opencity.py --dry-run      # just list what it found

Output:
    data/raw/opencity/<station-slug>_2017-23.csv
    data/raw/opencity/<station-slug>_2024-25.csv
    data/raw/stations.csv   (or stations_skeleton.csv if stations.csv already exists)
                            -> fill in the lat / lon columns yourself

Standard library only. Re-running skips files that are already downloaded.
"""
import argparse
import csv
import html as htmllib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

PAGE = "https://data.opencity.in/dataset/delhi-hourly-air-quality-reports"
API = "https://data.opencity.in/api/3/action/package_show?id=delhi-hourly-air-quality-reports"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic data download script)"}

# "Alipur DPCC AQI Data 2017-2023" / "Alipur DPCC 15 minute AQI Data for 2024-25" -> "Alipur DPCC"
NAME_RE = re.compile(r"\s*(?:15 minute\s+)?AQI Data(?: for)?\s*\d{4}-\d{2,4}\s*$", re.I)

# The 2024-25 files label a few stations differently from the 2017-2023 files.
# normalized 2024-25 key -> normalized 2017-2023 key
ALIASES = {
    "dwarkasector8": "dwarkasector8dpcc",
    "okhlaphase2": "okhlaphase2dpcc",
    "ihbascpcb": "ihbasdilshadgardencpcb",
    "northcampusimd": "northcampusduimd",
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_resources_api():
    data = json.loads(fetch(API))
    return [{"name": r["name"], "url": r["url"]} for r in data["result"]["resources"]]


def list_resources_html():
    """Fallback: scrape the dataset page. Works whatever order the <a> attributes are in."""
    page = fetch(PAGE).decode("utf-8", "replace")
    titles, downloads = {}, {}
    for tag in re.findall(r"<a\b[^>]*>", page):
        href = re.search(r'href="([^"]+)"', tag)
        if not href:
            continue
        link = htmllib.unescape(href.group(1))
        m = re.search(r"/resource/([0-9a-f]{8}-[0-9a-f-]{27})", link)
        if not m:
            continue
        rid = m.group(1)
        if "/download/" in link:
            downloads[rid] = urljoin(PAGE, link)
        else:
            t = re.search(r'title="([^"]*)"', tag)
            if t:
                titles[rid] = htmllib.unescape(t.group(1))
    return [{"name": titles[r], "url": downloads[r]} for r in downloads if r in titles]


def pair_stations(resources):
    """Group resources into {key: {label, '2017-23': url, '2024-25': url}}."""
    stations = {}
    for r in resources:
        n = r["name"]
        if "2017-2023" in n:
            period = "2017-23"
        elif "2024-25" in n:
            period = "2024-25"
        else:
            print(f"  (skipping unrecognised resource: {n})")
            continue
        label = NAME_RE.sub("", n).strip()
        key = norm(label)
        key = ALIASES.get(key, key)
        s = stations.setdefault(key, {"label": label})
        if period == "2017-23":
            s["label"] = label            # prefer the 2017-2023 file's station name
        s[period] = r["url"]
    return stations


def download(url, dest, retries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        try:
            body = fetch(url)
            if body[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
                raise ValueError("got an HTML page instead of CSV")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(dest)
            return "ok"
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(2 * attempt)
    return f"FAILED: {last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/opencity")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        resources = list_resources_api()
        print(f"CKAN API: found {len(resources)} resources")
    except Exception as e:                           # noqa: BLE001
        print(f"CKAN API failed ({e}); falling back to scraping the page")
        resources = list_resources_html()
        print(f"Page scrape: found {len(resources)} resources")
    if not resources:
        sys.exit("No resources found. Check your internet connection / the site.")

    stations = pair_stations(resources)
    print(f"{len(stations)} stations (expected 39)")
    incomplete = [s["label"] for s in stations.values()
                  if "2017-23" not in s or "2024-25" not in s]
    if incomplete:
        print("WARNING - missing one of the two files for:", ", ".join(incomplete))
    if args.dry_run:
        for k in sorted(stations, key=lambda k: stations[k]["label"]):
            s = stations[k]
            print(f"  {s['label']:<45} 2017-23:{'yes' if '2017-23' in s else 'NO '} "
                  f"2024-25:{'yes' if '2024-25' in s else 'NO'}")
        return

    out = Path(args.out)
    jobs = []
    for s in stations.values():
        slug = slugify(s["label"])
        for period in ("2017-23", "2024-25"):
            if period in s:
                jobs.append((s["label"], period, s[period], out / f"{slug}_{period}.csv"))

    results = {"ok": 0, "skipped": 0, "failed": []}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, url, dest): (label, period, dest)
                for label, period, url, dest in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            label, period, dest = futs[fut]
            status = fut.result()
            if status.startswith("FAILED"):
                results["failed"].append((dest.name, status))
            else:
                results[status] += 1
            print(f"[{i:>2}/{len(jobs)}] {dest.name}: {status}")

    # stations.csv skeleton (never overwrite one you've already filled in)
    raw = Path(args.raw)
    target = raw / "stations.csv"
    if target.exists():
        target = raw / "stations_skeleton.csv"
    rows = []
    for n, s in enumerate(sorted(stations.values(), key=lambda s: s["label"]), 1):
        slug = slugify(s["label"])
        files = ";".join(f"{slug}_{p}.csv" for p in ("2017-23", "2024-25") if p in s)
        rows.append([f"DL{n:03d}", s["label"], "", "", files])
    with open(target, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "name", "lat", "lon", "files"])
        w.writerows(rows)

    print(f"\nDownloaded {results['ok']}, already present {results['skipped']}, "
          f"failed {len(results['failed'])}")
    for name, status in results["failed"]:
        print("  ", name, status)
    print(f"Wrote {target}  -> fill in lat/lon for each station.")

    # show the header of one file so you can check column detection
    sample = next(iter(sorted(out.glob("*_2017-23.csv"))), None)
    if sample:
        print(f"\nHeader of {sample.name}:")
        print(sample.read_text(errors="replace").splitlines()[:3])
    sample = next(iter(sorted(out.glob("*_2024-25.csv"))), None)
    if sample:
        print(f"Header of {sample.name}:")
        print(sample.read_text(errors="replace").splitlines()[:3])


if __name__ == "__main__":
    main()
