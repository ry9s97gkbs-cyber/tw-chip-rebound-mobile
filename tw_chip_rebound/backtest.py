"""Backtest the mobile price-weak/chip-strong watchlist.

The backtest uses the same ranking logic as the mobile app, then measures
next-session and 2-to-5-session follow-through from the signal-day close.
It is an observation backtest, not a fill-accurate trading simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd

from .public_fetcher import (
    FinMindClient,
    FinMindError,
    _candidate_ids,
    _fetch_twse_institutional_one,
    _fetch_twse_prices_one,
    _iter_dates,
    add_concentration,
    fetch_institutional,
    fetch_prices,
    fetch_stock_info,
)
from .screener import rank_stocks


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters for one observation backtest."""

    start: str
    end: str
    token: str = ""
    top_n: int = 10
    candidate_limit: int = 120
    lookback_days: int = 90
    mode: str = "rank"
    public_market: str = "twse"


def _parse_date(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(str(value).strip().replace("/", "-"), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"日期格式錯誤：{value}，請使用 YYYY-MM-DD")
    return parsed.normalize()


def _fetch_twse_only(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch reliable historical TWSE-only price and institutional data."""

    def with_retry(fn, day):
        for attempt in range(3):
            try:
                return fn(day)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        return pd.DataFrame()

    price_frames: list[pd.DataFrame] = []
    chip_frames: list[pd.DataFrame] = []
    days = [day for day in _iter_dates(start_date, end_date) if day.weekday() < 5]
    with ThreadPoolExecutor(max_workers=4) as executor:
        price_futures = {executor.submit(with_retry, _fetch_twse_prices_one, day): day for day in days}
        chip_futures = {executor.submit(with_retry, _fetch_twse_institutional_one, day): day for day in days}
        for future in as_completed(price_futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                price_frames.append(frame)
        for future in as_completed(chip_futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                chip_frames.append(frame)
    if not price_frames or not chip_frames:
        raise FinMindError("TWSE-only 公開回測資料抓取失敗")
    daily = pd.concat(price_frames, ignore_index=True).dropna(subset=["open", "high", "low", "close", "volume"])
    main = pd.concat(chip_frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily["stock_id"] = daily["stock_id"].astype(str)
    main["date"] = pd.to_datetime(main["date"]).dt.date
    main["stock_id"] = main["stock_id"].astype(str)
    daily.attrs["source"] = "TWSE public historical"
    main.attrs["source"] = "TWSE public institutional historical"
    return daily, main


def _fetch_backtest_inputs(config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Fetch one continuous data window for the full backtest."""

    start = _parse_date(config.start)
    end = _parse_date(config.end)
    fetch_start = (start - timedelta(days=config.lookback_days)).date().isoformat()
    fetch_end = (end + timedelta(days=10)).date().isoformat()

    client = FinMindClient(token=config.token)
    if not client.token and config.public_market == "twse":
        daily, main = _fetch_twse_only(fetch_start, fetch_end)
    else:
        daily = fetch_prices(client, fetch_start, fetch_end)
        try:
            info = fetch_stock_info(client)
            daily = daily.merge(info[["stock_id", "stock_name"]], on="stock_id", how="left", suffixes=("", "_info"))
            if "stock_name_info" in daily.columns:
                daily["stock_name"] = daily["stock_name"].fillna(daily["stock_name_info"])
                daily = daily.drop(columns=["stock_name_info"])
        except Exception:
            if "stock_name" not in daily.columns:
                daily["stock_name"] = ""
        main = fetch_institutional(client, fetch_start, fetch_end)

    price_source = daily.attrs.get("source", "FinMind TaiwanStockPrice")
    chip_source = main.attrs.get("source", "FinMind TaiwanStockInstitutionalInvestorsBuySell")
    main = add_concentration(main, daily)
    meta = {
        "price_source": price_source,
        "chip_source": chip_source,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "public_market": config.public_market,
    }
    return daily, main, meta


def _future_metrics(price_history: pd.DataFrame, signal_date: pd.Timestamp, close: float) -> dict[str, float | None]:
    """Measure future return from signal close to later closes/highs/lows."""

    future = price_history[price_history["date_ts"] > signal_date].sort_values("date_ts").head(5)
    if future.empty or not close:
        return {
            "隔日報酬%": None,
            "2日報酬%": None,
            "5日報酬%": None,
            "5日最高報酬%": None,
            "5日最大回落%": None,
        }

    def ret_at(index: int) -> float | None:
        if len(future) <= index:
            return None
        return round((float(future.iloc[index]["close"]) / close - 1) * 100, 2)

    return {
        "隔日報酬%": ret_at(0),
        "2日報酬%": ret_at(1),
        "5日報酬%": ret_at(4),
        "5日最高報酬%": round((float(future["high"].max()) / close - 1) * 100, 2),
        "5日最大回落%": round((float(future["low"].min()) / close - 1) * 100, 2),
    }


def _summarize(rows: pd.DataFrame, group_cols: Iterable[str] = ()) -> pd.DataFrame:
    """Summarize return quality for all rows or grouped slices."""

    if rows.empty:
        return pd.DataFrame()

    def one(group: pd.DataFrame) -> pd.Series:
        next_ret = pd.to_numeric(group["隔日報酬%"], errors="coerce")
        ret_2d = pd.to_numeric(group["2日報酬%"], errors="coerce")
        ret_5d = pd.to_numeric(group["5日報酬%"], errors="coerce")
        high_5d = pd.to_numeric(group["5日最高報酬%"], errors="coerce")
        low_5d = pd.to_numeric(group["5日最大回落%"], errors="coerce")
        next_valid = next_ret.dropna()
        ret_2d_valid = ret_2d.dropna()
        ret_5d_valid = ret_5d.dropna()
        high_5d_valid = high_5d.dropna()
        low_5d_valid = low_5d.dropna()

        def win_rate(series: pd.Series) -> float | None:
            if series.empty:
                return None
            return round((series > 0).mean() * 100, 2)

        return pd.Series(
            {
                "樣本數": int(len(group)),
                "隔日有效樣本": int(len(next_valid)),
                "隔日勝率%": win_rate(next_valid),
                "隔日平均%": round(next_valid.mean(), 2) if not next_valid.empty else None,
                "2日有效樣本": int(len(ret_2d_valid)),
                "2日勝率%": win_rate(ret_2d_valid),
                "2日平均%": round(ret_2d_valid.mean(), 2) if not ret_2d_valid.empty else None,
                "5日有效樣本": int(len(ret_5d_valid)),
                "5日勝率%": win_rate(ret_5d_valid),
                "5日平均%": round(ret_5d_valid.mean(), 2) if not ret_5d_valid.empty else None,
                "5日中位%": round(ret_5d_valid.median(), 2) if not ret_5d_valid.empty else None,
                "5日曾漲逾3%機率%": round((high_5d_valid >= 3).mean() * 100, 2) if not high_5d_valid.empty else None,
                "5日平均最高%": round(high_5d_valid.mean(), 2) if not high_5d_valid.empty else None,
                "5日平均最大回落%": round(low_5d_valid.mean(), 2) if not low_5d_valid.empty else None,
            }
        )

    if group_cols:
        return rows.groupby(list(group_cols), dropna=False).apply(one, include_groups=False).reset_index()
    return one(rows).to_frame().T


def run_backtest(config: BacktestConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    """Run a full historical watchlist backtest."""

    daily, main, meta = _fetch_backtest_inputs(config)
    daily = daily.copy()
    daily["date_ts"] = pd.to_datetime(daily["date"])
    start = _parse_date(config.start)
    end = _parse_date(config.end)
    available_dates = sorted(d for d in daily["date_ts"].dropna().unique() if start <= pd.Timestamp(d) <= end)
    price_by_stock = {stock_id: frame.sort_values("date_ts") for stock_id, frame in daily.groupby("stock_id")}

    records: list[dict[str, object]] = []
    for day in available_dates:
        day_text = pd.Timestamp(day).date().isoformat()
        candidates = _candidate_ids(daily, main, day_text, config.candidate_limit)
        watchlist = rank_stocks(
            daily,
            main,
            branch_chip=pd.DataFrame(),
            target_date=day_text,
            top_n=config.top_n,
            candidate_stock_ids=candidates,
        )
        if watchlist.empty:
            continue
        for rank, row in enumerate(watchlist.to_dict(orient="records"), start=1):
            stock_id = str(row["股票代號"])
            close = float(row["收盤價"])
            future = _future_metrics(price_by_stock.get(stock_id, pd.DataFrame()), pd.Timestamp(day), close)
            records.append(
                {
                    "日期": day_text,
                    "排名": rank,
                    **row,
                    **future,
                    "分數區間": _score_bucket(row.get("分數")),
                    "排名區間": _rank_bucket(rank),
                }
            )

    trades = pd.DataFrame(records)
    summaries = {
        "overall": _summarize(trades),
        "by_score": _summarize(trades, ["分數區間"]),
        "by_rank": _summarize(trades, ["排名區間"]),
        "by_signal": _summarize(trades, ["訊號分類"]),
    }
    meta = {**meta, "start": config.start, "end": config.end, "top_n": str(config.top_n), "samples": str(len(trades))}
    return trades, summaries, meta


def _score_bucket(score: object) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "NA"
    if value >= 90:
        return "90+"
    if value >= 80:
        return "80-89"
    if value >= 70:
        return "70-79"
    if value >= 60:
        return "60-69"
    return "<60"


def _rank_bucket(rank: int) -> str:
    if rank <= 3:
        return "Top1-3"
    if rank <= 10:
        return "Top4-10"
    return "Top11+"


def write_backtest_outputs(trades: pd.DataFrame, summaries: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write trades and summary CSV files."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    for name, frame in summaries.items():
        frame.to_csv(path / f"backtest_{name}.csv", index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回測價弱籌碼強觀察名單")
    parser.add_argument("--start", required=True, help="回測開始日 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="回測結束日 YYYY-MM-DD")
    parser.add_argument("--token", default="", help="FinMind token；空白時使用公開資料備援")
    parser.add_argument("--top-n", type=int, default=10, help="每日觀察前 N 名")
    parser.add_argument("--candidate-limit", type=int, default=120, help="每日初選候選檔數")
    parser.add_argument(
        "--public-market",
        choices=("twse", "all"),
        default="twse",
        help="沒有 token 時的公開回測市場；twse 較適合歷史回測，all 僅供資料源驗證",
    )
    parser.add_argument("--output-dir", default="outputs/backtest", help="CSV 輸出資料夾")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BacktestConfig(
        start=args.start,
        end=args.end,
        token=args.token,
        top_n=args.top_n,
        candidate_limit=args.candidate_limit,
        public_market=args.public_market,
    )
    trades, summaries, meta = run_backtest(config)
    write_backtest_outputs(trades, summaries, args.output_dir)
    print(f"完成回測：{meta['start']} ~ {meta['end']}，樣本 {meta['samples']} 筆")
    print(f"資料源：價量={meta['price_source']}，籌碼={meta['chip_source']}")
    if not summaries["overall"].empty:
        print(summaries["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
