from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
DATASET_DIR = ROOT / "eodhd_output" / "japan_korea_fundamentals_fixture"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fixture_payload(code: str, name: str, exchange: str, currency: str, country_iso: str) -> dict:
    yearly_income = {}
    yearly_balance = {}
    yearly_cash = {}
    for index, year in enumerate(range(2019, 2025), start=1):
        date_key = f"{year}-12-31"
        yearly_income[date_key] = {
            "date": date_key,
            "totalRevenue": 1000 + index * 100,
            "netIncome": 100 + index * 20,
        }
        yearly_balance[date_key] = {
            "date": date_key,
            "totalStockholderEquity": 800 + index * 50,
        }
        yearly_cash[date_key] = {
            "date": date_key,
            "totalCashFromOperatingActivities": 160 + index * 20,
            "capitalExpenditures": -(40 + index * 5),
            "repurchaseOfCapitalStock": -(20 + index * 3),
            "dividendsPaid": -(10 + index * 2),
        }
    return {
        "General": {
            "Code": code,
            "Name": name,
            "Exchange": exchange,
            "CurrencyCode": currency,
            "CountryISO": country_iso,
            "Sector": "Consumer Cyclical",
            "Industry": "Auto Manufacturers",
        },
        "Highlights": {
            "MarketCapitalization": 10000,
        },
        "Financials": {
            "Income_Statement": {"yearly": yearly_income, "quarterly": {}},
            "Balance_Sheet": {"yearly": yearly_balance, "quarterly": {}},
            "Cash_Flow": {"yearly": yearly_cash, "quarterly": {}},
        },
        "Earnings": {
            "Annual": {
                "2024-12-31": {
                    "date": "2024-12-31",
                    "epsActual": 12.3,
                }
            }
        },
    }


def main() -> int:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    manifest = [
        {
            "symbol": "7203.TSE",
            "code": "7203",
            "exchange": "TSE",
            "name": "Fixture Motors",
            "country": "Japan",
            "country_key": "JP",
            "currency": "JPY",
            "type": "Common Stock",
            "isin": "JP0000000000",
            "is_delisted_symbol_list": False,
            "exchange_name": "Tokyo Stock Exchange",
            "exchange_country": "Japan",
            "exchange_country_iso2": "JP",
            "exchange_country_iso3": "JPN",
        },
        {
            "symbol": "005930.KO",
            "code": "005930",
            "exchange": "KO",
            "name": "Fixture Electronics",
            "country": "South Korea",
            "country_key": "KR",
            "currency": "KRW",
            "type": "Common Stock",
            "isin": "KR0000000000",
            "is_delisted_symbol_list": False,
            "exchange_name": "Korea Stock Exchange",
            "exchange_country": "South Korea",
            "exchange_country_iso2": "KR",
            "exchange_country_iso3": "KOR",
        }
    ]
    write_json(DATASET_DIR / "stock_pull_manifest.json", manifest)
    write_json(DATASET_DIR / "raw" / "7203.TSE" / "fundamentals.json", fixture_payload("7203", "Fixture Motors", "TSE", "JPY", "JP"))
    write_json(DATASET_DIR / "raw" / "005930.KO" / "fundamentals.json", fixture_payload("005930", "Fixture Electronics", "KO", "KRW", "KR"))

    env = dict(os.environ)
    env["EODHD_JK_OUT_DIR"] = str(DATASET_DIR)
    subprocess.run([PYTHON, str(ROOT / "eodhd_japan_korea_fundamentals.py"), "--normalize-only"], check=True, env=env)
    subprocess.run([PYTHON, str(ROOT / "screen_japan_korea_shareholder_yield.py"), "--top", "10"], check=True, env=env)
    subprocess.run(
        [PYTHON, str(ROOT / "audit_japan_korea_eodhd_dataset.py"), "--require-screening"],
        check=True,
        env=env,
    )

    income_5y = read_csv(DATASET_DIR / "normalized" / "income_statement_latest_5y.csv")
    balance_5y = read_csv(DATASET_DIR / "normalized" / "balance_sheet_latest_5y.csv")
    cash_5y = read_csv(DATASET_DIR / "normalized" / "cash_flow_latest_5y.csv")
    screen_rows = read_csv(DATASET_DIR / "screening" / "shareholder_yield_screen_top.csv")
    audit = json.loads((DATASET_DIR / "dataset_audit.json").read_text(encoding="utf-8"))

    assert len(income_5y) == 10, len(income_5y)
    assert len(balance_5y) == 10, len(balance_5y)
    assert len(cash_5y) == 10, len(cash_5y)
    assert income_5y[0]["date"] == "2024-12-31"
    assert screen_rows[0]["symbol"] == "7203.TSE"
    assert abs(float(screen_rows[0]["shareholder_yield"]) - 0.006) < 0.000001
    assert float(screen_rows[0]["average_roe_5y"]) > 0
    assert float(screen_rows[0]["free_cash_flow_cagr_5y"]) > 0
    assert audit["complete"] is True

    print(
        json.dumps(
            {
                "status": "ok",
                "income_statement_latest_5y_rows": len(income_5y),
                "balance_sheet_latest_5y_rows": len(balance_5y),
                "cash_flow_latest_5y_rows": len(cash_5y),
                "top_symbol": screen_rows[0]["symbol"],
                "shareholder_yield": screen_rows[0]["shareholder_yield"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
