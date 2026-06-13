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
from html import unescape
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
TWSE_PRICE_URL = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
TWSE_INST_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_PRICE_URL = "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"
TPEX_INST_URL = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
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
        text = str(value).strip().replace("/", "-")
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            raise FinMindError(f"日期格式錯誤：{value}。請使用 YYYY-MM-DD，例如 2026-06-12。")
        return parsed.date().isoformat()
    return date.today().isoformat()


def _range_dates(end_date: str | None, days: int) -> tuple[str, str]:
    end = datetime.strptime(_to_date_string(end_date), "%Y-%m-%d").date()
    start = end - timedelta(days=max(days, 25) * 2)
    return start.isoformat(), end.isoformat()


def _resolve_target_date(daily: pd.DataFrame, requested_date: str | None) -> str:
    """Use requested date when available, otherwise latest available trading day."""

    available = sorted(set(pd.to_datetime(daily["date"]).dt.date))
    if not available:
        raise FinMindError("價量資料沒有可用交易日。")
    if requested_date is None:
        return available[-1].isoformat()
    requested = datetime.strptime(_to_date_string(requested_date), "%Y-%m-%d").date()
    candidates = [day for day in available if day <= requested]
    if not candidates:
        raise FinMindError(f"{requested.isoformat()} 前沒有可用交易日資料。")
    return candidates[-1].isoformat()


def _iter_dates(start_date: str, end_date: str) -> list[date]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(total_days + 1)]


def _twse_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _tpex_date(value: date) -> str:
    return f"{value.year - 1911}/{value:%m/%d}"


def _to_number(value: Any) -> float:
    if value in (None, "", "-"):
        return math.nan
    text = re.sub(r"<[^>]+>", "", unescape(str(value))).strip()
    if text in ("", "-", "---", "--"):
        return math.nan
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return math.nan


def _shares_to_lots(value: Any) -> float:
    number = _to_number(value)
    if not math.isfinite(number):
        return math.nan
    # FinMind buy/sell and trading volume fields are published in shares.
    # Strategy thresholds are in lots, so normalize once at the data boundary.
    return number / 1000


def _is_permission_error(exc: FinMindError) -> bool:
    text = str(exc).lower()
    return "your level" in text or "update your user level" in text or "sponsor" in text


def _http_json(url: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": "tw-chip-rebound/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _table_with_fields(payload: dict[str, Any], required: set[str]) -> tuple[list[str], list[list[Any]]] | None:
    for table in payload.get("tables", []):
        fields = table.get("fields") or []
        if required.issubset(set(fields)):
            return fields, table.get("data") or []
    return None


def _fetch_twse_prices_one(day: date) -> pd.DataFrame:
    payload = _http_json(TWSE_PRICE_URL, {"response": "json", "date": _twse_date(day), "type": "ALLBUT0999"})
    if payload.get("stat") != "OK":
        return pd.DataFrame()
    table = _table_with_fields(payload, {"證券代號", "證券名稱", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"})
    if table is None:
        return pd.DataFrame()
    fields, rows = table
    idx = {name: fields.index(name) for name in fields}
    records = []
    for row in rows:
        stock_id = str(row[idx["證券代號"]]).strip()
        if not re.fullmatch(r"\d{4}", stock_id):
            continue
        volume_shares = _to_number(row[idx["成交股數"]])
        trading_money = _to_number(row[idx["成交金額"]])
        records.append(
            {
                "date": day,
                "stock_id": stock_id,
                "stock_name": str(row[idx["證券名稱"]]).strip(),
                "open": _to_number(row[idx["開盤價"]]),
                "high": _to_number(row[idx["最高價"]]),
                "low": _to_number(row[idx["最低價"]]),
                "close": _to_number(row[idx["收盤價"]]),
                "volume": volume_shares / 1000 if math.isfinite(volume_shares) else math.nan,
                "avg_price": trading_money / volume_shares if volume_shares and math.isfinite(volume_shares) else math.nan,
            }
        )
    return pd.DataFrame(records)


def _fetch_tpex_prices_one(day: date) -> pd.DataFrame:
    payload = _http_json(TPEX_PRICE_URL, {"l": "zh-tw", "d": _tpex_date(day), "o": "json"})
    if payload.get("stat") != "ok":
        return pd.DataFrame()
    table = _table_with_fields(payload, {"代號", "名稱", "收盤", "開盤", "最高", "最低", "均價", "成交股數"})
    if table is None:
        return pd.DataFrame()
    fields, rows = table
    idx = {name: fields.index(name) for name in fields}
    money_field = "成交金額(元)" if "成交金額(元)" in idx else "成交金額"
    records = []
    for row in rows:
        stock_id = str(row[idx["代號"]]).strip()
        if not re.fullmatch(r"\d{4}", stock_id):
            continue
        volume_shares = _to_number(row[idx["成交股數"]])
        trading_money = _to_number(row[idx[money_field]]) if money_field in idx else math.nan
        records.append(
            {
                "date": day,
                "stock_id": stock_id,
                "stock_name": str(row[idx["名稱"]]).strip(),
                "open": _to_number(row[idx["開盤"]]),
                "high": _to_number(row[idx["最高"]]),
                "low": _to_number(row[idx["最低"]]),
                "close": _to_number(row[idx["收盤"]]),
                "volume": volume_shares / 1000 if math.isfinite(volume_shares) else math.nan,
                "avg_price": _to_number(row[idx["均價"]]) if "均價" in idx else (trading_money / volume_shares if volume_shares else math.nan),
            }
        )
    return pd.DataFrame(records)


def fetch_prices_official(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily prices from TWSE and TPEx public endpoints."""

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for day in _iter_dates(start_date, end_date):
            if day.weekday() >= 5:
                continue
            futures.append(executor.submit(_fetch_twse_prices_one, day))
            futures.append(executor.submit(_fetch_tpex_prices_one, day))
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise FinMindError("台交所/櫃買公開價量資料抓取失敗")
    result = pd.concat(frames, ignore_index=True).dropna(subset=["open", "high", "low", "close", "volume"])
    result.attrs["source"] = "TWSE/TPEx public"
    return result


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

    try:
        rows = client.dataset("TaiwanStockPrice", start_date=start_date, end_date=end_date)
    except FinMindError as exc:
        if _is_permission_error(exc):
            return fetch_prices_official(start_date, end_date)
        raise
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
    result = daily.dropna(subset=["open", "high", "low", "close", "volume"])
    result.attrs["source"] = "FinMind TaiwanStockPrice"
    return result


def _fetch_twse_institutional_one(day: date) -> pd.DataFrame:
    payload = _http_json(TWSE_INST_URL, {"date": _twse_date(day), "selectType": "ALLBUT0999", "response": "json"})
    if payload.get("stat") != "OK":
        return pd.DataFrame()
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    if not fields or not rows:
        return pd.DataFrame()
    idx = {name: fields.index(name) for name in fields}
    foreign_field = "外陸資買賣超股數(不含外資自營商)"
    foreign_dealer_field = "外資自營商買賣超股數"
    trust_field = "投信買賣超股數"
    total_field = "三大法人買賣超股數"
    records = []
    for row in rows:
        stock_id = str(row[idx["證券代號"]]).strip()
        if not re.fullmatch(r"\d{4}", stock_id):
            continue
        foreign = _to_number(row[idx[foreign_field]]) if foreign_field in idx else 0
        foreign_dealer = _to_number(row[idx[foreign_dealer_field]]) if foreign_dealer_field in idx else 0
        records.append(
            {
                "date": day,
                "stock_id": stock_id,
                "main_buy_sell": _shares_to_lots(row[idx[total_field]]) if total_field in idx else math.nan,
                "foreign_buy_sell": (foreign + foreign_dealer) / 1000,
                "investment_trust_buy_sell": _shares_to_lots(row[idx[trust_field]]) if trust_field in idx else 0,
            }
        )
    return pd.DataFrame(records)


def _fetch_tpex_institutional_one(day: date) -> pd.DataFrame:
    payload = _http_json(
        TPEX_INST_URL,
        {"l": "zh-tw", "o": "json", "se": "EW", "t": "D", "d": _tpex_date(day)},
    )
    if payload.get("stat") != "ok":
        return pd.DataFrame()
    table = payload.get("tables", [{}])[0]
    rows = table.get("data") or []
    records = []
    for row in rows:
        stock_id = str(row[0]).strip()
        if not re.fullmatch(r"\d{4}", stock_id):
            continue
        # TPEx duplicate field names are position-based:
        # 8-10 foreign total, 11-13 investment trust, 23 total institutions.
        records.append(
            {
                "date": day,
                "stock_id": stock_id,
                "main_buy_sell": _shares_to_lots(row[23]) if len(row) > 23 else math.nan,
                "foreign_buy_sell": _shares_to_lots(row[10]) if len(row) > 10 else 0,
                "investment_trust_buy_sell": _shares_to_lots(row[13]) if len(row) > 13 else 0,
            }
        )
    return pd.DataFrame(records)


def fetch_institutional_official(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch institutional buy/sell from TWSE and TPEx public endpoints."""

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for day in _iter_dates(start_date, end_date):
            if day.weekday() >= 5:
                continue
            futures.append(executor.submit(_fetch_twse_institutional_one, day))
            futures.append(executor.submit(_fetch_tpex_institutional_one, day))
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise FinMindError("台交所/櫃買公開法人資料抓取失敗")
    result = pd.concat(frames, ignore_index=True)
    result.attrs["source"] = "TWSE/TPEx public institutional"
    return result


def fetch_institutional(client: FinMindClient, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch institutional buy/sell and derive concentration fields.

    This is used as the automatic public-data proxy for "main-force" flow.
    Broker branch concentration is added later for final candidates.
    """

    try:
        rows = client.dataset(
            "TaiwanStockInstitutionalInvestorsBuySell",
            start_date=start_date,
            end_date=end_date,
        )
    except FinMindError as exc:
        if _is_permission_error(exc):
            return fetch_institutional_official(start_date, end_date)
        raise
    df = pd.DataFrame(rows)
    if df.empty:
        raise FinMindError("FinMind institutional buy/sell returned no rows")
    buy = df["buy"].map(_shares_to_lots)
    sell = df["sell"].map(_shares_to_lots)
    investor_name = df.get("name")
    if investor_name is None:
        investor_name = df.get("institutional_investors", "")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]).dt.date,
            "stock_id": df["stock_id"].astype(str),
            "investor_name": investor_name.astype(str) if hasattr(investor_name, "astype") else "",
            "net_lots": buy - sell,
        }
    )
    main = df.groupby(["date", "stock_id"], as_index=False)["net_lots"].sum()
    main = main.rename(columns={"net_lots": "main_buy_sell"})
    foreign = (
        df[df["investor_name"].str.contains("外資", na=False)]
        .groupby(["date", "stock_id"], as_index=False)["net_lots"]
        .sum()
        .rename(columns={"net_lots": "foreign_buy_sell"})
    )
    trust = (
        df[df["investor_name"].str.contains("投信", na=False)]
        .groupby(["date", "stock_id"], as_index=False)["net_lots"]
        .sum()
        .rename(columns={"net_lots": "investment_trust_buy_sell"})
    )
    main = main.merge(foreign, on=["date", "stock_id"], how="left")
    main = main.merge(trust, on=["date", "stock_id"], how="left")
    return main


def add_concentration(main: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5-day and 20-day chip concentration from net buy lots."""

    df = main.merge(daily[["date", "stock_id", "volume"]], on=["date", "stock_id"], how="left")
    df = df.sort_values(["stock_id", "date"]).copy()
    grouped = df.groupby("stock_id", group_keys=False)
    for col in ("foreign_buy_sell", "investment_trust_buy_sell"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    buy_5 = grouped["main_buy_sell"].rolling(5).sum().reset_index(level=0, drop=True)
    vol_5 = grouped["volume"].rolling(5).sum().reset_index(level=0, drop=True)
    buy_20 = grouped["main_buy_sell"].rolling(20).sum().reset_index(level=0, drop=True)
    vol_20 = grouped["volume"].rolling(20).sum().reset_index(level=0, drop=True)
    df["concentration_5d"] = buy_5 / vol_5
    df["concentration_20d"] = buy_20 / vol_20
    df["foreign_buy_3d_count"] = (
        df.assign(_flag=(df["foreign_buy_sell"] > 0).astype(int))
        .groupby("stock_id")["_flag"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["investment_trust_buy_3d_count"] = (
        df.assign(_flag=(df["investment_trust_buy_sell"] > 0).astype(int))
        .groupby("stock_id")["_flag"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["investment_trust_buy_5d_count"] = (
        df.assign(_flag=(df["investment_trust_buy_sell"] > 0).astype(int))
        .groupby("stock_id")["_flag"]
        .rolling(5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    trust_prev_buy_days = (
        df.assign(_flag=(df["investment_trust_buy_sell"] > 0).astype(int))
        .groupby("stock_id")["_flag"]
        .rolling(20)
        .sum()
        .reset_index(level=0, drop=True)
        .groupby(df["stock_id"])
        .shift(1)
    )
    trust_ratio = df["investment_trust_buy_sell"] / df["volume"]
    df["investment_trust_new_buy_signal"] = (
        (df["investment_trust_buy_sell"] > 0)
        & (trust_ratio >= 0.02)
        & (trust_prev_buy_days.fillna(0) == 0)
    ).astype(int)
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
    sell_top = grouped[grouped["sell"] > 0].sort_values("sell", ascending=False).head(15)
    total_buy = top["buy"].sum()
    total_sell = sell_top["sell"].sum()
    net_top15 = total_buy - total_sell
    avg_price = top["_amount"].sum() / total_buy if total_buy else math.nan
    buyer_count = int((grouped["net"] > 0).sum())
    seller_count = int((grouped["net"] < 0).sum())
    names = ";".join(top["securities_trader"].astype(str).head(15).tolist())
    return {
        "date": pd.to_datetime(target_date).date(),
        "stock_id": stock_id,
        "top15_avg_price": avg_price,
        "top15_brokers": names,
        "top15_net_buy": net_top15,
        "top15_sell": total_sell,
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
    merged["top15_net_volume_ratio"] = math.nan
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
    day["_trust_new"] = day.get("investment_trust_new_buy_signal", pd.Series(0, index=day.index)).fillna(0) >= 1
    day["_trust_3d"] = day.get("investment_trust_buy_3d_count", pd.Series(0, index=day.index)).fillna(0) >= 3
    day["_foreign_3d"] = day.get("foreign_buy_3d_count", pd.Series(0, index=day.index)).fillna(0) >= 3
    day = day[day["_price_weak"] | day["_main_positive"] | day["_concentration_ok"]].copy()
    day["_rank"] = (
        day["_price_weak"].astype(int) * 40
        + day["_main_positive"].astype(int) * 25
        + day["_main_ratio_ok"].astype(int) * 15
        + day["_concentration_ok"].astype(int) * 10
        + day["_trust_new"].astype(int) * 18
        + day["_trust_3d"].astype(int) * 12
        + day["_foreign_3d"].astype(int) * 8
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
    if "count_diff" not in concentration.columns:
        concentration = concentration.assign(count_diff=math.nan)
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
    price_source = daily.attrs.get("source", "FinMind TaiwanStockPrice")
    daily = daily.merge(info[["stock_id", "stock_name"]], on="stock_id", how="inner", suffixes=("", "_info"))
    if "stock_name_info" in daily.columns:
        if "stock_name" in daily.columns:
            daily["stock_name"] = daily["stock_name"].fillna(daily["stock_name_info"])
        else:
            daily["stock_name"] = daily["stock_name_info"]
        daily = daily.drop(columns=["stock_name_info"])
    requested_target = _to_date_string(target_date) if target_date else None
    target = _resolve_target_date(daily, target_date)

    main = fetch_institutional(client, start_date, end_date)
    chip_source = main.attrs.get("source", "FinMind TaiwanStockInstitutionalInvestorsBuySell")
    main = add_concentration(main, daily)

    candidates = _candidate_ids(daily, main, target, branch_limit)
    branch_rows = []
    branch_errors: list[str] = []
    branch_skipped_reason = ""
    branch_enabled = price_source.startswith("FinMind") and chip_source.startswith("FinMind") and bool(client.token)
    if branch_enabled:
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
    else:
        branch_skipped_reason = "FinMind 全市場權限不足或未提供 token，已跳過分點 API 以加快掃描。"
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
        "requested_date": requested_target,
        "start_date": start_date,
        "end_date": end_date,
        "mode": mode,
        "top_n": top_n,
        "price_rows": int(len(daily)),
        "chip_rows": int(len(main)),
        "branch_candidates": int(len(candidates)),
        "branch_rows": int(len(branch)),
        "branch_errors": branch_errors[:10],
        "branch_skipped_reason": branch_skipped_reason,
        "diagnostics": diagnostics,
        "data_source": "FinMind TaiwanStockPrice, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockTradingDailyReport",
        "price_source": price_source,
        "chip_source": chip_source,
    }
    return FetchResult(rows=rows, meta=meta)
