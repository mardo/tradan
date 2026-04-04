#!/usr/bin/env python3
import os
import argparse
import datetime as dt
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from typing import Iterable


DATA_HOST = "https://data.binance.vision"
BASE_PREFIX = "data/futures/um/monthly"

ALL_INTERVALS = [
    "12h", "15m", "1d", "1h", "1m", "1mo", "1w",
    "2h", "30m", "3d", "3m", "4h", "5m", "6h", "8h"
]


def build_klines_url(
    symbol: str, interval: str, year: int, month: int, checksum=False
) -> str:
    month_str = f"{year:04d}-{month:02d}"
    suffix = ".CHECKSUM" if checksum else ""
    return (
        f"{DATA_HOST}/{BASE_PREFIX}/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{month_str}.zip{suffix}"
    )


def monthrange(start: dt.date, end: dt.date) -> Iterable[tuple[int, int]]:
    """Yield (year, month) for each calendar month from start's month through end's month."""
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y, m) <= (end_y, end_m):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def check_url(url: str):
    """Send HEAD request to verify existence"""
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as resp:
            print(f"[OK] {url}")
    except HTTPError as e:
        if e.code == 404:
            print(f"[MISSING] {url}")
        else:
            print(f"[HTTP {e.code}] {url}")
    except Exception as e:
        print(f"[ERROR] {url} ({e})")


def download_file(url: str, out_path: str, overwrite=False, dry_run=False):
    if dry_run:
        print(f"[DRY] {url}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not overwrite:
        print(f"Skip: {out_path}")
        return

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as resp, open(out_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        print(f"Downloaded: {out_path}")

    except HTTPError as e:
        if e.code == 404:
            print(f"Missing: {url}")
        else:
            print(f"HTTP {e.code}: {url}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance USDT-M futures monthly kline zips from data.binance.vision"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", nargs="+", help="Override intervals")
    parser.add_argument(
        "--start",
        required=True,
        help="Start (YYYY-MM-DD or YYYY-MM); first month is taken from this date",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End (YYYY-MM-DD or YYYY-MM); last month is taken from this date",
    )
    parser.add_argument("--outdir", default="downloads")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="Check URLs with HEAD request (no download)")
    args = parser.parse_args()

    intervals = args.interval if args.interval else ALL_INTERVALS

    start_raw = args.start.strip()
    end_raw = args.end.strip()
    if len(start_raw) == 7 and start_raw[4] == "-":
        start_date = dt.date.fromisoformat(f"{start_raw}-01")
    else:
        start_date = dt.date.fromisoformat(start_raw)
    if len(end_raw) == 7 and end_raw[4] == "-":
        end_date = dt.date.fromisoformat(f"{end_raw}-01")
    else:
        end_date = dt.date.fromisoformat(end_raw)

    print(f"Intervals: {intervals}")

    for interval in intervals:
        for year, month in monthrange(start_date, end_date):
            url = build_klines_url(args.symbol, interval, year, month)
            month_label = f"{year:04d}-{month:02d}"

            out_path = os.path.join(
                args.outdir,
                args.symbol,
                interval,
                f"{args.symbol}-{interval}-{month_label}.zip",
            )

            # Priority: verify > dry-run > download
            if args.verify:
                check_url(url)
            else:
                download_file(url, out_path,
                              overwrite=args.overwrite,
                              dry_run=args.dry_run)

            if args.checksum:
                cs_url = build_klines_url(
                    args.symbol, interval, year, month, checksum=True
                )

                if args.verify:
                    check_url(cs_url)
                else:
                    download_file(cs_url, out_path + ".CHECKSUM",
                                  overwrite=args.overwrite,
                                  dry_run=args.dry_run)


if __name__ == "__main__":
    main()