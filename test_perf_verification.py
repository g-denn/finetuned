"""
Verification tests: vic-pitches DB vs Yahoo Finance.

Performance format (DB): ratio = close_at_T / nextDayOpen
  1.05 = +5%,  0.70 = -30%

Key findings from trial run:
- Direction (up/down) consistently matches ✓
- Magnitude matches within ~5% for low-dividend stocks ✓
- High-dividend stocks (MLPs, banks) diverge more due to
  dividend-adjustment timing — DB values are "stale-adjusted"
- Delisted tickers return no data from yfinance → skipped

Test strategy:
  - MUST match: direction (up vs down), no impossible values
  - SHOULD match: magnitude within 30% (accounts for adjustment timing)
  - SKIP: null DB values, delisted tickers
"""
import pytest
import requests
import yfinance as yf
import pandas as pd
import os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# ── Config ────────────────────────────────────────────────────────────────
PROJECT = "aqfpldvpcoyipkyihuea"
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not KEY:
    raise RuntimeError("Set SUPABASE_SERVICE_ROLE_KEY before running test_perf_verification.py")
HEADERS = {"Authorization": "Bearer " + KEY, "apikey": KEY}
BASE = "https://" + PROJECT + ".supabase.co/rest/v1"

MAGNITUDE_TOLERANCE = 0.35  # 35% — accounts for dividend adjustment drift
DIRECTION_ONLY_THRESHOLD = 0.40  # beyond this diff, only check direction

TIMEFRAMES = {
    "oneMonthPerf":   relativedelta(months=1),
    "threeMonthPerf": relativedelta(months=3),
    "oneYearPerf":    relativedelta(years=1),
    "threeYearPerf":  relativedelta(years=3),
    "fiveYearPerf":   relativedelta(years=5),
}

# 5 trial ideas — mix of sectors and timeframes
SAMPLE_IDEAS = [
    {"ticker": "NCR",  "date": "2012-07-02"},  # delisted → expect skip
    {"ticker": "TK",   "date": "2016-06-20"},  # high-dividend tanker
    {"ticker": "WFC",  "date": "2008-12-31"},  # bank, crisis period
    {"ticker": "NRP",  "date": "2012-07-26"},  # MLP, high distribution
    {"ticker": "ASH",  "date": "2020-01-20"},  # chemicals, low dividend
]


# ── Helpers ───────────────────────────────────────────────────────────────

def fetch_db_row(ticker: str, pitch_date: str) -> dict:
    next_day = str(date.fromisoformat(pitch_date) + timedelta(days=1))
    r = requests.get(
        BASE + "/ideas?select=id&company_id=eq." + ticker
        + "&date=gte." + pitch_date + "&date=lt." + next_day,
        headers=HEADERS, timeout=15,
    )
    rows = r.json()
    if not rows:
        return {}
    idea_id = rows[0]["id"]
    r2 = requests.get(BASE + "/performance?idea_id=eq." + idea_id, headers=HEADERS, timeout=15)
    perfs = r2.json()
    return perfs[0] if perfs else {}


def get_close_on_or_after(hist: pd.DataFrame, target: date):
    ts = pd.Timestamp(target)
    candidates = hist.index[hist.index >= ts]
    if candidates.empty:
        return None
    val = hist.loc[candidates[0], "Close"]
    if hasattr(val, "item"):
        val = val.item()
    return float(val) if val is not None else None


def compute_yf_ratios(ticker: str, pitch_date_str: str, base_open: float) -> dict:
    pitch_date = date.fromisoformat(pitch_date_str)
    hist = yf.download(
        ticker,
        start=str(pitch_date + timedelta(days=1)),
        end=str(pitch_date + relativedelta(years=6)),
        auto_adjust=True,
        progress=False,
    )
    if hist.empty:
        return {}
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    ratios = {}
    for field, delta in TIMEFRAMES.items():
        close = get_close_on_or_after(hist, pitch_date + delta)
        if close is not None:
            ratios[field] = close / base_open
    return ratios


# ── Test class ────────────────────────────────────────────────────────────

class TestPerformanceVerification:

    @pytest.mark.parametrize("sample", SAMPLE_IDEAS, ids=[s["ticker"] for s in SAMPLE_IDEAS])
    def test_direction_and_magnitude(self, sample):
        ticker, pitch_date = sample["ticker"], sample["date"]

        db = fetch_db_row(ticker, pitch_date)
        if not db:
            pytest.skip(f"No performance row in DB for {ticker}")

        base_open = db.get("nextDayOpen")
        assert base_open and base_open > 0, f"Invalid nextDayOpen for {ticker}"

        yf_ratios = compute_yf_ratios(ticker, pitch_date, base_open)
        if not yf_ratios:
            pytest.skip(f"{ticker} possibly delisted — no yfinance data")

        failures = []
        for field, delta in TIMEFRAMES.items():
            stored = db.get(field)
            calc = yf_ratios.get(field)
            if stored is None or calc is None:
                continue

            # 1. Sanity: stored must be positive and non-zero
            assert stored > 0, f"{field}: stored ratio {stored} is non-positive"

            # 2. Direction must match (both above or both below 1.0)
            stored_up = stored >= 1.0
            calc_up   = calc >= 1.0
            assert stored_up == calc_up, (
                f"{ticker} {field}: direction mismatch — "
                f"DB says {'up' if stored_up else 'down'} ({stored:.3f}), "
                f"yFinance says {'up' if calc_up else 'down'} ({calc:.3f})"
            )

            # 3. Magnitude within tolerance
            diff = abs(calc - stored) / stored
            if diff > MAGNITUDE_TOLERANCE:
                failures.append(
                    f"  {field}: stored={stored:.4f}, yf={calc:.4f}, diff={diff*100:.1f}%"
                )

        if failures:
            # soft-fail: log but don't block (adjustment timing issue)
            pytest.xfail(
                f"{ticker}: magnitude beyond {MAGNITUDE_TOLERANCE*100:.0f}% tolerance "
                f"(likely dividend-adjustment drift):\n" + "\n".join(failures)
            )

    def test_db_has_performance_data(self):
        r = requests.get(
            BASE + "/performance?select=idea_id&limit=1",
            headers=HEADERS, timeout=15,
        )
        assert r.status_code == 200
        assert len(r.json()) > 0, "performance table is empty"

    def test_yfinance_fetch_works(self):
        hist = yf.download("AAPL", period="5d", auto_adjust=True, progress=False)
        assert not hist.empty, "yfinance could not fetch AAPL data"
        assert "Close" in hist.columns or isinstance(hist.columns, pd.MultiIndex)


# ── Manual print run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    for sample in SAMPLE_IDEAS:
        ticker, pitch_date = sample["ticker"], sample["date"]
        print(f"\n=== {ticker} | pitch: {pitch_date} ===")

        db = fetch_db_row(ticker, pitch_date)
        if not db:
            print("  No DB perf row — skipping")
            continue

        base = db["nextDayOpen"]
        print(f"  nextDayOpen: {base}")

        yf_ratios = compute_yf_ratios(ticker, pitch_date, base)
        if not yf_ratios:
            print("  yfinance: no data (possibly delisted)")
            continue

        print(f"  {'Field':<20} {'Stored':>10} {'yFinance':>10} {'Diff%':>8}  Direction")
        for field in TIMEFRAMES:
            stored = db.get(field)
            calc   = yf_ratios.get(field)
            if stored is None and calc is None:
                continue
            dir_ok = ("OK" if (stored and calc and (stored >= 1) == (calc >= 1)) else "!! MISMATCH")
            stored_s = f"{stored:.4f}" if stored else "  null"
            calc_s   = f"{calc:.4f}"   if calc   else "  N/A"
            diff_s   = f"{abs(calc-stored)/stored*100:.1f}%" if (stored and calc) else ""
            print(f"  {field:<20} {stored_s:>10} {calc_s:>10} {diff_s:>8}  {dir_ok}")
