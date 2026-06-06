"""
Batch-fetch split+dividend-adjusted performance ratios from Yahoo Finance
for all ideas in the vic-pitches Supabase database, then upsert results.

Performance format: ratio = close_at_T / nextDayOpen
  1.05 = +5%, 0.70 = -30%

Timeframes computed:
  nextDayOpen, nextDayClose,
  oneWeekClosePerf, twoWeekClosePerf,
  oneMonthPerf, threeMonthPerf, sixMonthPerf,
  oneYearPerf, twoYearPerf, threeYearPerf, fiveYearPerf
"""
import requests
import yfinance as yf
import pandas as pd
import time
import json
import logging
import os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
PROJECT = "aqfpldvpcoyipkyihuea"
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not KEY:
    raise SystemExit("Set SUPABASE_SERVICE_ROLE_KEY before running fetch_perf_yfinance.py")
BASE = "https://" + PROJECT + ".supabase.co/rest/v1"
HEADERS = {
    "Authorization": "Bearer " + KEY,
    "apikey": KEY,
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates,return=minimal",
}

UPSERT_BATCH = 50    # rows per upsert call — flush early and often
FETCH_PAGE   = 1000  # ideas per page
SLEEP_TICKER = 0.3   # seconds between ticker downloads
SLEEP_BATCH  = 1.0   # seconds between upsert batches

TIMEFRAMES = {
    "oneWeekClosePerf":   timedelta(weeks=1),
    "twoWeekClosePerf":   timedelta(weeks=2),
    "oneMonthPerf":       relativedelta(months=1),
    "threeMonthPerf":     relativedelta(months=3),
    "sixMonthPerf":       relativedelta(months=6),
    "oneYearPerf":        relativedelta(years=1),
    "twoYearPerf":        relativedelta(years=2),
    "threeYearPerf":      relativedelta(years=3),
    "fiveYearPerf":       relativedelta(years=5),
}

# Bloomberg exchange code → Yahoo Finance suffix
# VIC stores tickers in Bloomberg convention: "TICKER EXCHANGE" e.g. "005930 KS"
BLOOMBERG_TO_YF = {
    # Asia-Pacific
    "KS":  ".KS",   # Korea Stock Exchange
    "KQ":  ".KQ",   # KOSDAQ
    "KP":  ".KS",   # Korea (alt)
    "JT":  ".T",    # Japan Tokyo
    "JP":  ".T",    # Japan (alt)
    "HK":  ".HK",   # Hong Kong
    "AU":  ".AX",   # Australia ASX
    "NZ":  ".NZ",   # New Zealand
    "SP":  ".SI",   # Singapore
    "MK":  ".KL",   # Malaysia Bursa
    "IJ":  ".JK",   # Indonesia Jakarta
    "PM":  ".PS",   # Philippines
    "TB":  ".BK",   # Thailand Bangkok
    "TT":  ".TW",   # Taiwan
    "CH":  ".SS",   # China Shanghai
    "CZ":  ".SZ",   # China Shenzhen
    "IN":  ".NS",   # India NSE
    "IB":  ".BO",   # India BSE
    # Europe
    "LN":  ".L",    # London LSE
    "LI":  ".L",    # London (alt)
    "GR":  ".DE",   # Germany Xetra
    "GY":  ".DE",   # Germany (alt)
    "FP":  ".PA",   # France Paris Euronext
    "NA":  ".AS",   # Netherlands Amsterdam
    "BB":  ".BR",   # Belgium Brussels
    "SW":  ".SW",   # Switzerland
    "SE":  ".ST",   # Sweden Stockholm
    "SS":  ".ST",   # Sweden (alt)
    "NO":  ".OL",   # Norway Oslo
    "DC":  ".CO",   # Denmark Copenhagen
    "FH":  ".HE",   # Finland Helsinki
    "SM":  ".MC",   # Spain Madrid
    "IM":  ".MI",   # Italy Milan
    "IT":  ".MI",   # Italy (alt)
    "PW":  ".LS",   # Portugal Lisbon
    "PL":  ".WA",   # Poland Warsaw
    "RU":  ".ME",   # Russia Moscow
    "IR":  ".IR",   # Ireland
    "ID":  ".IR",   # Ireland (alt)
    "AT":  ".VI",   # Austria Vienna
    # Americas
    "CN":  ".TO",   # Canada Toronto
    "CT":  ".TO",   # Canada Toronto (alt)
    "CV":  ".V",    # Canada TSX Venture
    "CF":  ".CN",   # Canada NEO/CSE
    "MX":  ".MX",   # Mexico
    "BZ":  ".SA",   # Brazil Sao Paulo
    # Middle East / Africa
    "IS":  ".IS",   # Turkey Istanbul
    "SA":  ".SR",   # Saudi Arabia
    "DU":  ".DU",   # Dubai
    "AD":  ".AE",   # Abu Dhabi
    "EY":  ".CA",   # Egypt Cairo
    "SJ":  ".JO",   # South Africa Johannesburg
}

# Tickers that are definitely not equities — skip entirely
SKIP_PATTERNS = ("MUNI", "CORP", " MTN ", " TBD", "GOVT")


def normalize_ticker(raw: str) -> str | None:
    """
    Convert VIC/Bloomberg ticker format to Yahoo Finance format.
    "005930 KS" → "005930.KS"
    "AAPL"      → "AAPL"
    Returns None if the ticker looks like a non-equity instrument.
    """
    raw = raw.strip()

    # Skip non-equity instruments
    for pat in SKIP_PATTERNS:
        if pat in raw:
            return None

    if " " not in raw:
        return raw  # Already plain — US ticker or already Yahoo format

    parts = raw.rsplit(" ", 1)
    base, exch = parts[0].strip(), parts[1].strip().upper()

    if exch in BLOOMBERG_TO_YF:
        return base + BLOOMBERG_TO_YF[exch]

    # Unknown exchange suffix — return as-is and let yfinance try
    return raw


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("fetch_perf_yfinance.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Supabase helpers ──────────────────────────────────────────────────────

def fetch_all_ideas() -> list[dict]:
    """Paginate through the ideas table and return id, company_id, date."""
    ideas = []
    offset = 0
    while True:
        r = requests.get(
            BASE + f"/ideas?select=id,company_id,date&order=id.asc"
            f"&limit={FETCH_PAGE}&offset={offset}",
            headers={**HEADERS, "Prefer": "count=exact"},
            timeout=30,
        )
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        ideas.extend(page)
        offset += len(page)
        log.info(f"  fetched {len(ideas)} ideas so far...")
        if len(page) < FETCH_PAGE:
            break
    return ideas


# All columns the performance table accepts — every row must carry every key
PERF_COLUMNS = [
    "idea_id", "nextDayOpen", "nextDayClose",
    "oneWeekClosePerf", "twoWeekClosePerf",
    "oneMonthPerf", "threeMonthPerf", "sixMonthPerf",
    "oneYearPerf", "twoYearPerf", "threeYearPerf", "fiveYearPerf",
]


def normalize_row(row: dict) -> dict:
    """Ensure every row has exactly the same keys (None for missing)."""
    return {col: row.get(col) for col in PERF_COLUMNS}


def upsert_batch(rows: list[dict]) -> bool:
    normalized = [normalize_row(r) for r in rows]
    r = requests.post(
        BASE + "/performance",
        headers=HEADERS,
        json=normalized,
        timeout=60,
    )
    if r.status_code not in (200, 201):
        log.error(f"Upsert failed {r.status_code}: {r.text[:300]}")
        return False
    return True


# ── Price helpers ─────────────────────────────────────────────────────────

def download_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download adjusted OHLC history. Returns empty df on error."""
    try:
        hist = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        log.warning(f"{ticker}: download error — {e}")
        return pd.DataFrame()

    if hist.empty:
        return hist

    # Flatten MultiIndex columns (happens when downloading a single ticker sometimes)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    # Require at least Open and Close columns with real data
    if "Open" not in hist.columns or "Close" not in hist.columns:
        return pd.DataFrame()
    if hist["Open"].dropna().empty:
        return pd.DataFrame()

    return hist


def get_price_on_or_after(hist: pd.DataFrame, target: date, col: str) -> float | None:
    ts = pd.Timestamp(target)
    candidates = hist.index[hist.index >= ts]
    if candidates.empty:
        return None
    raw = hist.loc[candidates[0], col]
    # loc can return a Series when the index has duplicate timestamps
    if isinstance(raw, pd.Series):
        raw = raw.iloc[0]
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if not pd.isna(val) else None


def compute_perf_row(idea: dict, hist: pd.DataFrame) -> dict | None:
    """
    For one idea, compute all performance ratios using adjusted prices.
    Returns None if no base price can be determined.
    """
    pitch_date = date.fromisoformat(idea["date"][:10])
    next_day = pitch_date + timedelta(days=1)

    base_open  = get_price_on_or_after(hist, next_day, "Open")
    base_close = get_price_on_or_after(hist, next_day, "Close")

    if base_open is None or base_open <= 0:
        return None

    row: dict = {
        "idea_id":     idea["id"],
        "nextDayOpen": base_open,
    }
    if base_close is not None and base_close > 0:
        row["nextDayClose"] = base_close

    for field, delta in TIMEFRAMES.items():
        target = pitch_date + delta
        close = get_price_on_or_after(hist, target, "Close")
        if close is not None and close > 0:
            row[field] = close / base_open

    return row


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log.info("=== fetch_perf_yfinance starting ===")

    # 1. Load all ideas
    log.info("Fetching all ideas from Supabase...")
    ideas = fetch_all_ideas()
    log.info(f"Total ideas: {len(ideas)}")

    # 2. Group ideas by normalized Yahoo Finance ticker
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    skipped_non_equity = []
    for idea in ideas:
        raw_ticker = idea.get("company_id", "")
        yf_ticker = normalize_ticker(raw_ticker)
        if yf_ticker is None:
            skipped_non_equity.append(raw_ticker)
            continue
        idea["_yf_ticker"] = yf_ticker
        by_ticker[yf_ticker].append(idea)

    if skipped_non_equity:
        log.info(f"Skipped {len(skipped_non_equity)} non-equity instruments: {skipped_non_equity[:10]}")

    tickers = sorted(by_ticker.keys())
    log.info(f"Unique tickers: {len(tickers)}")

    # 3. Process per ticker
    today = date.today()
    skipped_tickers = []
    total_rows = 0
    pending_upsert: list[dict] = []

    def flush(force=False):
        nonlocal total_rows
        while len(pending_upsert) >= UPSERT_BATCH or (force and pending_upsert):
            batch = pending_upsert[:UPSERT_BATCH]
            del pending_upsert[:UPSERT_BATCH]
            if upsert_batch(batch):
                total_rows += len(batch)
                log.info(f"  upserted {total_rows} rows total")
            else:
                log.error(f"  batch upsert failed — {len(batch)} rows lost")
            if pending_upsert:
                time.sleep(SLEEP_BATCH)

    for i, ticker in enumerate(tickers):
        ticker_ideas = by_ticker[ticker]
        dates = [date.fromisoformat(x["date"][:10]) for x in ticker_ideas]
        earliest = min(dates) + timedelta(days=1)
        # Download up to 6 years past the latest pitch (max timeframe is 5y)
        latest_end = min(max(dates) + relativedelta(years=6), today + timedelta(days=1))

        # Show original VIC ticker for context if it was remapped
        raw_sample = ticker_ideas[0].get("company_id", ticker)
        display = ticker if ticker == raw_sample else f"{raw_sample} → {ticker}"
        log.info(f"[{i+1}/{len(tickers)}] {display} — {len(ticker_ideas)} ideas, "
                 f"range {earliest} → {latest_end}")

        hist = download_ticker(ticker, earliest, latest_end)

        if hist.empty:
            log.warning(f"  {ticker}: no data (delisted or unknown) — skipping {len(ticker_ideas)} ideas")
            skipped_tickers.append(ticker)
            continue  # no sleep for failed tickers — tear through them fast

        for idea in ticker_ideas:
            row = compute_perf_row(idea, hist)
            if row is None:
                log.debug(f"  {ticker} idea {idea['id']}: no base open price")
                continue
            pending_upsert.append(row)

        flush()
        time.sleep(SLEEP_TICKER)  # only sleep after a successful download

    flush(force=True)

    log.info("=== Done ===")
    log.info(f"Total performance rows upserted: {total_rows}")
    log.info(f"Skipped tickers ({len(skipped_tickers)}): {skipped_tickers[:30]}")

    # 4. Final row count
    r = requests.get(
        BASE + "/performance?select=idea_id",
        headers={**HEADERS, "Prefer": "count=exact", "Range": "0-0"},
        timeout=15,
    )
    count = r.headers.get("content-range", "?").split("/")[-1]
    log.info(f"performance table now has {count} rows")


if __name__ == "__main__":
    main()
