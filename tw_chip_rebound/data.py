"""CSV data loading and normalization helpers.

The screener keeps its strategy logic independent from data vendors.  Each
loader accepts common English column names and a few Chinese aliases, then
normalizes them into one internal schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ColumnMap = dict[str, tuple[str, ...]]


DAILY_COLUMNS: ColumnMap = {
    "date": ("date", "日期"),
    "stock_id": ("stock_id", "symbol", "code", "股票代號", "證券代號"),
    "stock_name": ("stock_name", "name", "股票名稱", "證券名稱"),
    "open": ("open", "開盤價", "開盤"),
    "high": ("high", "最高價", "最高"),
    "low": ("low", "最低價", "最低"),
    "close": ("close", "收盤價", "收盤"),
    "volume": ("volume", "成交量", "成交股數", "成交張數"),
    "avg_price": ("avg_price", "均價", "今日均價"),
}

MAIN_COLUMNS: ColumnMap = {
    "date": ("date", "日期"),
    "stock_id": ("stock_id", "symbol", "code", "股票代號", "證券代號"),
    "main_buy_sell": ("main_buy_sell", "主力買賣超", "主力買賣超張數", "主力買超"),
    "buyer_count": ("buyer_count", "買進家數", "買方家數"),
    "seller_count": ("seller_count", "賣出家數", "賣方家數"),
    "count_diff": ("count_diff", "家數差"),
    "concentration_5d": ("concentration_5d", "5日集中度", "五日集中度"),
    "concentration_20d": ("concentration_20d", "20日集中度", "二十日集中度"),
}

BRANCH_COLUMNS: ColumnMap = {
    "date": ("date", "日期"),
    "stock_id": ("stock_id", "symbol", "code", "股票代號", "證券代號"),
    "top15_avg_price": ("top15_avg_price", "買方Top15均價", "買方TOP15均價"),
    "top15_brokers": ("top15_brokers", "買方Top15分點", "買方TOP15分點", "買方分點"),
}

CUSTODY_COLUMNS: ColumnMap = {
    "date": ("date", "日期"),
    "stock_id": ("stock_id", "symbol", "code", "股票代號", "證券代號"),
    "holder_count": ("holder_count", "集保戶數", "持股人數"),
}


def load_csv(path: str | Path, column_map: ColumnMap, required: Iterable[str]) -> pd.DataFrame:
    """Load one CSV file and normalize vendor-specific column names."""

    df = pd.read_csv(path)
    rename: dict[str, str] = {}
    for standard_name, aliases in column_map.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = standard_name
                break

    df = df.rename(columns=rename)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["stock_id"] = df["stock_id"].astype(str).str.strip()

    for col in df.columns:
        if col not in {"date", "stock_id", "stock_name", "top15_brokers"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_daily_k(path: str | Path) -> pd.DataFrame:
    """Load daily OHLCV data: open, high, low, close and volume."""

    return load_csv(
        path,
        DAILY_COLUMNS,
        required=("date", "stock_id", "open", "high", "low", "close", "volume"),
    )


def load_main_chip(path: str | Path) -> pd.DataFrame:
    """Load main-force buy/sell, house-count difference and concentration data."""

    df = load_csv(path, MAIN_COLUMNS, required=("date", "stock_id", "main_buy_sell"))
    if "count_diff" not in df.columns and {"buyer_count", "seller_count"}.issubset(df.columns):
        # 家數差用買方家數減賣方家數；小於 0 代表買盤集中、賣方較分散。
        df["count_diff"] = df["buyer_count"] - df["seller_count"]
    return df


def load_branch_chip(path: str | Path) -> pd.DataFrame:
    """Load top-15 branch average cost and branch names."""

    return load_csv(path, BRANCH_COLUMNS, required=("date", "stock_id"))


def load_custody(path: str | Path | None) -> pd.DataFrame | None:
    """Load TDCC holder-count data when available.

    This is optional because many free sources do not provide daily holder
    counts.  When omitted, custody-related bonus points are simply not applied.
    """

    if path is None:
        return None
    return load_csv(path, CUSTODY_COLUMNS, required=("date", "stock_id", "holder_count"))
