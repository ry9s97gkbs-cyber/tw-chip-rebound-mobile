"""Public-data fetchers for the mobile stock screener.

FinMind is used for market-wide price and chip data.  Broker branch data is
fetched only for candidates so the app stays fast on mobile and preserves API
quota.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import re
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .screener import (
    ScreenConfig,
    chip_strong_conditions,
    is_excluded,
    is_price_weak,
    merge_inputs,
    rank_stocks,
    score_stock,
    screen_stocks,
)


FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_BRANCH_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report"
KNOWN_BRANCH_KEYWORDS = ("凱基台北", "富邦", "元大", "統一", "國票", "港商野村")


def clean_token(token: str) -> str:
    """Normalize pasted FinMind tokens from mobile browsers.

    iOS paste can include spaces, newlines, zero-width characters, or the
    copied prefix "Bearer ".  FinMind rejects those as an illegal token.
    """

    value = (token or "").strip()
    value = re.sub(r"^Bearer\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[\s\u200b\u200c\u200d\ufeff]", "", value)
    return value


@dataclass(frozen=True)
class FetchResult:
    """Result object returned to the mobile API."""

    rows: list[dict[str, Any]]
    meta: dict[str, Any]


class FinMindError(RuntimeError):
    """Raised when FinMind returns an error response."""


class FinMindClient:
    """Small stdlib-only FinMind HTTP client."""

    def __init__(self, token: str = "", timeout: int = 30) -> None:
        self.token = clean_token(token)
        self.timeout = timeout

    def get(self, url: str, params: dict[str, Any], label: str = "FinMind") -> dict[str, Any]:
        query = dict(params)
        if self.token:
            query["token"] = self.token
        request_url = f"{url}?{urlencode(query)}"
        request = Request(request_url, headers={"User-Agent": "tw-chip-rebound/1.0"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
                message = payload.get("msg") or payload.get("error") or body
            except json.JSONDecodeError:
                message = body or str(exc)
            raise FinMindError(f"{label} 回傳 {exc.code}: {message}") from exc
        except URLError as exc:
            raise FinMindError(f"{label} 連線失敗: {exc.reason}") from exc
        if payload.get("status") not in (None, 200):
            raise FinMindError(f"{label} 回傳 {payload.get('status')}: {payload.get('msg') or payload}")
        return payload

    def dataset(self, name: str, **params: Any) -> list[dict[str, Any]]:
        payload = self.get(FINMIND_DATA_URL, {"dataset": name, **params}, label=name)
        return payload.get("data", [])

    def branch_report(self, stock_id: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
        payload = self.get(
            FINMIND_BRANCH_URL,
            {"data_id": stock_id, "start_date": start_date, "end_date": end_date},
            label=f"TaiwanStockTradingDailyReport {stock_id}",
        )
        return payload.get("data", [])


def _to_date_string(value: str | None) -> str:
    if value:
        return value
    return date.today().isoformat()


def _range_dates(end_date: str | None, days: int) -> tuple[str, str]:
    end = datetime.strptime(_to_date_string(end_date), "%Y-%m-%d").date()
    start = end - timedelta(days=max(days, 25) * 2)
    return start.isoformat(), end.isoformat()


def _to_number(value: Any) -> float:
    if value in (None, "", "-"):
        return math.nan
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return math.nan


def _shares_to_lots(value: Any) -> float:
    number = _to_number(value)
    if not math.isfinite(number):
        return math.nan
    # FinMind buy/sell and trading volume fields are published in shares.
    # Strategy thresholds are in lots, so normalize once at the data boundary.
    return number / 1000


def fetch_stock_info(client: FinMindClient) -> pd.DataFrame:
    """Fetch common TWSE/TPEx stock metadata."""

    rows = client.dataset("TaiwanStockInfo")
    df = pd.DataFrame(rows)
    if df.empty:
        raise FinMindError("FinMind TaiwanStockInfo returned no rows")
    df = df[df["type"].isin(["twse", "tpex"])].copy()
    df = df[df["stock_id"].astype(str).str.fullmatch(r"\d{4}")]
    df = df.sort_values("date").drop_duplicates("stock_id", keep="last")
    return df[["stock_id", "stock_name", "type", "industry_category"]]


def fetch_prices(client: FinMindClient, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch market-wide daily OHLCV and normalize volume to lots."""

    rows = client.dataset("TaiwanStockPrice", start_date=start_date, end_date=end_date)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FinMindError("FinMind TaiwanStockPrice returned no rows")
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]).dt.date,
            "stock_id": df["stock_id"].astype(str),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["max"], errors="coerce"),
            "low": pd.to_numeric(df["min"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["Trading_Volume"], errors="coerce") / 1000,
            "avg_price": pd.to_numeric(df["Trading_money"], errors="coerce")
            / pd.to_numeric(df["Trading_Volume"], errors="coerce"),
        }
    )
    return daily.dropna(subset=["open", "high", "low", "close", "volume"])


def fetch_institutional(client: FinMindClient, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch institutional buy/sell and derive concentration fields.

    This is used as the automatic public-data proxy for "main-force" flow.
    Broker branch concentration is added later for final candidates.
    """

    rows = client.dataset(
        "TaiwanStockInstitutionalInvestorsBuySell",
        start_date=start_date,
        end_date=end_date,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FinMindError("FinMind institutional buy/sell returned no rows")
    buy = df["buy"].map(_shares_to_lots)
    sell = df["sell"].map(_shares_to_lots)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]).dt.date,
            "stock_id": df["stock_id"].astype(str),
            "net_lots": buy - sell,
        }
    )
    main = df.groupby(["date", "stock_id"], as_index=False)["net_lots"].sum()
    main = main.rename(columns={"net_lots": "main_buy_sell"})
    return main


def add_concentration(main: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5-day and 20-day chip concentration from net buy lots."""

    df = main.merge(daily[["date", "stock_id", "volume"]], on=["date", "stock_id"], how="left")
    df = df.sort_values(["stock_id", "date"]).copy()
    grouped = df.groupby("stock_id", group_keys=False)
    buy_5 = grouped["main_buy_sell"].rolling(5).sum().reset_index(level=0, drop=True)
    vol_5 = grouped["volume"].rolling(5).sum().reset_index(level=0, drop=True)
    buy_20 = grouped["main_buy_sell"].rolling(20).sum().reset_index(level=0, drop=True)
    vol_20 = grouped["volume"].rolling(20).sum().reset_index(level=0, drop=True)
    df["concentration_5d"] = buy_5 / vol_5
    df["concentration_20d"] = buy_20 / vol_20
    return df.drop(columns=["volume"])


def build_branch_row(token: str, stock_id: str, target_date: str) -> dict[str, Any]:
    """Fetch and summarize Top15 buyer branch data for one candidate."""

    client = FinMindClient(token=token, timeout=15)
    rows = client.branch_report(stock_id, target_date, target_date)
    if not rows:
        return {"date": pd.to_datetime(target_date).date(), "stock_id": stock_id}
    df = pd.DataFrame(rows)
    df["buy"] = df["buy"].map(_shares_to_lots)
    df["sell"] = df["sell"].map(_shares_to_lots)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    grouped = df.groupby(["securities_trader", "securities_trader_id"], as_index=False).agg(
        buy=("buy", "sum"),
        sell=("sell", "sum"),
        buy_amount=("price", lambda s: 0.0),
    )
    buy_amount = (
        df.assign(_amount=df["price"] * df["buy"])
        .groupby(["securities_trader", "securities_trader_id"], as_index=False)["_amount"]
        .sum()
    )
    grouped = grouped.merge(buy_amount, on=["securities_trader", "securities_trader_id"], how="left")
    grouped["net"] = grouped["buy"] - grouped["sell"]
    top = grouped[grouped["buy"] > 0].sort_values("buy", ascending=False).head(15)
    total_buy = top["buy"].sum()
    avg_price = top["_amount"].sum() / total_buy if total_buy else math.nan
    buyer_count = int((grouped["net"] > 0).sum())
    seller_count = int((grouped["net"] < 0).sum())
    names = ";".join(top["securities_trader"].astype(str).head(15).tolist())
    return {
        "date": pd.to_datetime(target_date).date(),
        "stock_id": stock_id,
        "top15_avg_price": avg_price,
        "top15_brokers": names,
        "buyer_count": buyer_count,
        "seller_count": seller_count,
        "count_diff": buyer_count - seller_count,
    }


def _candidate_ids(daily: pd.DataFrame, main: pd.DataFrame, target_date: str, limit: int) -> list[str]:
    """Build a cheap preliminary candidate list before broker calls.

    This is deliberately looser than the final strategy.  Its job is to decide
    which stocks deserve the slower branch API calls, so it keeps anything with
    usable liquidity plus at least one practical rebound clue.
    """

    from .screener import calculate_indicators

    target = pd.to_datetime(target_date).date()
    price = calculate_indicators(daily)
    merged = price.merge(main, on=["date", "stock_id"], how="left")
    if "main_buy_sell" not in merged.columns:
        merged["main_buy_sell"] = math.nan
    merged["main_volume_ratio"] = merged["main_buy_sell"].fillna(0) / merged["volume"]
    merged["count_diff"] = -1
    if "concentration_5d" not in merged.columns:
        merged["concentration_5d"] = math.nan
    if "concentration_20d" not in merged.columns:
        merged["concentration_20d"] = math.nan
    merged["top15_avg_price"] = math.nan
    merged["close_above_main_cost_pct"] = math.nan
    day = merged[merged["date"] == target].copy()
    if day.empty:
        return []
    config = ScreenConfig()
    day = day[day.apply(lambda row: not is_excluded(row, config), axis=1)].copy()
    day["_price_weak"] = day.apply(lambda row: is_price_weak(row, config), axis=1)
    day["_main_positive"] = day["main_buy_sell"].fillna(0) > 0
    day["_main_ratio_ok"] = day["main_volume_ratio"].fillna(0) >= 0.02
    day["_concentration_ok"] = day["concentration_5d"].fillna(-999) > day["concentration_20d"].fillna(-999)
    day = day[day["_price_weak"] | day["_main_positive"] | day["_concentration_ok"]].copy()
    day["_rank"] = (
        day["_price_weak"].astype(int) * 40
        + day["_main_positive"].astype(int) * 25
        + day["_main_ratio_ok"].astype(int) * 15
        + day["_concentration_ok"].astype(int) * 10
        + day["main_volume_ratio"].fillna(0).clip(lower=0, upper=0.2) * 100
        + day["main_buy_sell"].fillna(0).clip(lower=0, upper=10000) / 1000
    )
    return day.sort_values("_rank", ascending=False)["stock_id"].head(limit).astype(str).tolist()


def diagnostic_counts(
    daily: pd.DataFrame,
    main: pd.DataFrame,
    branch: pd.DataFrame,
    target_date: str,
) -> dict[str, int]:
    """Count how many stocks survive each major strategy gate."""

    config = ScreenConfig()
    target = pd.to_datetime(target_date).date()
    df = merge_inputs(daily, main, branch_chip=branch)
    day = df[df["date"] == target].copy()
    if day.empty:
        return {
            "target_total": 0,
            "not_excluded": 0,
            "price_weak": 0,
            "main_buy_1000": 0,
            "main_ratio_5": 0,
            "concentration_stronger": 0,
            "count_diff_negative": 0,
            "chip_strong": 0,
            "score_over_60": 0,
        }

    not_excluded = day[~day.apply(lambda row: is_excluded(row, config), axis=1)]
    price_weak = not_excluded[not_excluded.apply(lambda row: is_price_weak(row, config), axis=1)]
    main_buy_1000 = price_weak[price_weak["main_buy_sell"] >= config.min_main_buy]
    main_ratio_5 = main_buy_1000[main_buy_1000["main_volume_ratio"] >= config.min_main_volume_ratio]
    concentration = main_ratio_5[
        main_ratio_5.apply(lambda row: chip_strong_conditions(row, config)["concentration_5d_stronger"], axis=1)
    ]
    count_diff = concentration[concentration["count_diff"] < 0]
    chip_strong = count_diff[
        count_diff.apply(
            lambda row: all(chip_strong_conditions(row, config).values()),
            axis=1,
        )
    ]
    score_over_60 = chip_strong[
        chip_strong.apply(lambda row: score_stock(row, config)[0] > 60, axis=1)
    ]
    return {
        "target_total": int(len(day)),
        "not_excluded": int(len(not_excluded)),
        "price_weak": int(len(price_weak)),
        "main_buy_1000": int(len(main_buy_1000)),
        "main_ratio_5": int(len(main_ratio_5)),
        "concentration_stronger": int(len(concentration)),
        "count_diff_negative": int(len(count_diff)),
        "chip_strong": int(len(chip_strong)),
        "score_over_60": int(len(score_over_60)),
    }


def screen_with_public_data(
    token: str,
    target_date: str | None = None,
    days: int = 45,
    branch_limit: int = 80,
    top_n: int = 30,
    mode: str = "practical",
) -> FetchResult:
    """Fetch public data, enrich candidates with broker branches, and screen."""

    client = FinMindClient(token=token)
    start_date, end_date = _range_dates(target_date, days)
    info = fetch_stock_info(client)
    daily = fetch_prices(client, start_date, end_date)
    daily = daily.merge(info[["stock_id", "stock_name"]], on="stock_id", how="inner")
    if target_date is None:
        target = str(max(daily["date"]))
    else:
        target = target_date

    main = fetch_institutional(client, start_date, end_date)
    main = add_concentration(main, daily)

    candidates = _candidate_ids(daily, main, target, branch_limit)
    branch_rows = []
    branch_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(build_branch_row, client.token, stock_id, target): stock_id
            for stock_id in candidates
        }
        for future in as_completed(futures):
            stock_id = futures[future]
            try:
                branch_rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - keep scanning other candidates.
                branch_errors.append(f"{stock_id}: {exc}")
    branch = pd.DataFrame(branch_rows)
    diagnostics = diagnostic_counts(daily, main, branch, target)

    if mode == "strict":
        result = screen_stocks(daily, main, branch_chip=branch, target_date=target)
        rows = result.to_dict(orient="records")
    else:
        result = rank_stocks(
            daily,
            main,
            branch_chip=branch,
            target_date=target,
            top_n=top_n,
            candidate_stock_ids=candidates,
        )
        rows = result.to_dict(orient="records")

    meta = {
        "target_date": target,
        "start_date": start_date,
        "end_date": end_date,
        "mode": mode,
        "top_n": top_n,
        "price_rows": int(len(daily)),
        "chip_rows": int(len(main)),
        "branch_candidates": int(len(candidates)),
        "branch_rows": int(len(branch)),
        "branch_errors": branch_errors[:10],
        "diagnostics": diagnostics,
        "data_source": "FinMind TaiwanStockPrice, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockTradingDailyReport",
    }
    return FetchResult(rows=rows, meta=meta)
