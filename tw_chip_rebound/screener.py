"""Price-weak/chip-strong stock selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

KNOWN_BRANCH_KEYWORDS = ("凱基台北", "富邦", "元大", "統一", "國票", "港商野村")

OUTPUT_COLUMNS = [
    "股票代號",
    "股票名稱",
    "收盤價",
    "今日漲跌幅",
    "最高價",
    "成交量",
    "主力買賣超",
    "主力買超佔成交量比例",
    "家數差",
    "5日集中度",
    "20日集中度",
    "買方Top15均價",
    "收盤價與主力均價差距",
    "分數",
    "訊號分類",
]


def _value(row: pd.Series, key: str, default: float = float("nan")) -> float:
    """Read a numeric row value without raising KeyError."""

    value = row.get(key, default)
    return value


@dataclass(frozen=True)
class ScreenConfig:
    """Thresholds for the screener.

    Values are intentionally centralized so daily tuning does not require
    changing condition code.
    """

    min_volume: float = 3000
    min_price: float = 20
    min_main_buy: float = 1000
    min_main_volume_ratio: float = 0.05
    max_gain_for_price_weak: float = 7.0
    max_gain_for_intraday_spike: float = 3.0
    intraday_spike_pct: float = 7.0
    long_upper_shadow_ratio: float = 0.04
    near_low_ratio: float = 0.02
    limit_down_pct: float = -9.5
    max_close_above_main_cost_pct: float = 0.08


def calculate_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages, return percentages and next-day confirmation.

    MA2/MA5 are used to detect short-term trend breaks.  The next-day close is
    only used when historical data already contains the next session, because
    the "可進場" classification requires standing back above today's midpoint.
    """

    df = daily.sort_values(["stock_id", "date"]).copy()
    grouped = df.groupby("stock_id", group_keys=False)

    df["prev_close"] = grouped["close"].shift(1)
    df["pct_change"] = (df["close"] / df["prev_close"] - 1) * 100
    df["intraday_high_pct"] = (df["high"] / df["prev_close"] - 1) * 100
    df["ma2"] = grouped["close"].rolling(2).mean().reset_index(level=0, drop=True)
    df["ma5"] = grouped["close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["ma20_volume"] = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    df["today_mid_price"] = (df["high"] + df["low"]) / 2
    df["next_close"] = grouped["close"].shift(-1)

    if "avg_price" not in df.columns:
        # 若資料源沒有今日均價，使用 OHLC 的簡化均價作為近似值。
        df["avg_price"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    return df


def merge_inputs(
    daily: pd.DataFrame,
    main_chip: pd.DataFrame,
    branch_chip: pd.DataFrame | None = None,
    custody: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge daily price, main-force, branch and optional custody data."""

    df = calculate_indicators(daily)
    df = df.merge(main_chip, on=["date", "stock_id"], how="left", suffixes=("", "_main"))

    if branch_chip is not None and not branch_chip.empty:
        df = df.merge(branch_chip, on=["date", "stock_id"], how="left", suffixes=("", "_branch"))

    if custody is not None and not custody.empty:
        custody = custody.sort_values(["stock_id", "date"]).copy()
        custody["prev_holder_count"] = custody.groupby("stock_id")["holder_count"].shift(1)
        df = df.merge(custody, on=["date", "stock_id"], how="left")

    for col in (
        "main_buy_sell",
        "count_diff",
        "concentration_5d",
        "concentration_20d",
        "top15_avg_price",
    ):
        if col not in df.columns:
            df[col] = float("nan")
    if "top15_brokers" not in df.columns:
        df["top15_brokers"] = ""

    df["main_volume_ratio"] = df["main_buy_sell"] / df["volume"]
    df["main_cost_gap"] = (df["top15_avg_price"] - df["close"]) / df["close"]
    df["close_above_main_cost_pct"] = (df["close"] - df["top15_avg_price"]) / df["top15_avg_price"]
    df["prev_concentration_5d"] = df.groupby("stock_id")["concentration_5d"].shift(1)
    df["main_buy_3d_count"] = (
        df.assign(_main_buy_flag=(df["main_buy_sell"] > 0).astype(int))
        .groupby("stock_id")["_main_buy_flag"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df


def price_weak_conditions(row: pd.Series, config: ScreenConfig = ScreenConfig()) -> dict[str, bool]:
    """Return price-weak condition flags.

    價弱不是單純下跌，而是「短線價格表現沒有跟上盤中強勢」或「收盤位置偏弱」。
    """

    return {
        # 今日收盤低於今日均價，代表尾盤沒有守住日內平均成本。
        "close_below_avg": bool(row["close"] < row["avg_price"]),
        # 收盤接近最低價，代表賣壓一路壓到收盤附近。
        "close_near_low": bool((row["close"] - row["low"]) / row["close"] <= config.near_low_ratio),
        # 收黑 K，代表收盤價低於開盤價。
        "black_k": bool(row["close"] < row["open"]),
        # 長上影線：盤中衝高後回落，且最高價與收盤價落差達 4%。
        "long_upper_shadow": bool((row["high"] - row["close"]) / row["close"] >= config.long_upper_shadow_ratio),
        # 盤中曾漲超過 7%，但收盤漲幅小於 3%，代表強勢未能延續到收盤。
        "intraday_spike_faded": bool(
            row["pct_change"] < config.max_gain_for_intraday_spike
            and row["intraday_high_pct"] >= config.intraday_spike_pct
        ),
        # 跌破 MA2 或 MA5，代表短均線支撐轉弱。
        "break_short_ma": bool(row["close"] < row["ma2"] or row["close"] < row["ma5"]),
    }


def chip_strong_conditions(row: pd.Series, config: ScreenConfig = ScreenConfig()) -> dict[str, bool]:
    """Return chip-strong condition flags.

    籌碼強的核心是主力逆勢買進，且買超量、佔比、集中度都不能只是零星買盤。
    """

    concentration_turns_strong = (
        pd.notna(row["concentration_5d"])
        and (
            (pd.notna(row["prev_concentration_5d"]) and row["prev_concentration_5d"] < 0 <= row["concentration_5d"])
            or (pd.notna(row["prev_concentration_5d"]) and row["concentration_5d"] > row["prev_concentration_5d"])
        )
    )

    return {
        # 主力買賣超大於 0，表示主力站在買方。
        "main_buy_positive": bool(row["main_buy_sell"] > 0),
        # 主力買超至少 1000 張，避免小量買超造成雜訊。
        "main_buy_over_1000": bool(row["main_buy_sell"] >= config.min_main_buy),
        # 主力買超佔成交量至少 5%，代表買盤對當日成交有影響力。
        "main_buy_ratio_over_5": bool(_value(row, "main_volume_ratio") >= config.min_main_volume_ratio),
        # 家數差小於 0，表示買方家數少於賣方家數，買盤較集中。
        "count_diff_negative": bool(row["count_diff"] < 0),
        # 5 日集中度由負轉正或高於前一天，代表短期籌碼集中度轉強。
        "concentration_5d_stronger": bool(concentration_turns_strong),
    }


def bonus_conditions(
    row: pd.Series,
    known_branch_keywords: Iterable[str] = KNOWN_BRANCH_KEYWORDS,
) -> dict[str, bool]:
    """Return bonus condition flags."""

    brokers = str(row.get("top15_brokers", ""))
    holder_count = row.get("holder_count")
    prev_holder_count = row.get("prev_holder_count")

    return {
        # 買方 Top15 均價高於收盤價，代表主力平均成本比現價高。
        "top15_cost_above_close": bool(pd.notna(row.get("top15_avg_price")) and row["top15_avg_price"] > row["close"]),
        # Top15 買方出現市場常追蹤的分點名稱，視為額外觀察訊號。
        "known_branch_in_top15": any(keyword in brokers for keyword in known_branch_keywords),
        # 今日成交量大於 20 日均量，代表有量能配合。
        "volume_above_ma20": bool(pd.notna(row["ma20_volume"]) and row["volume"] > row["ma20_volume"]),
        # 近 3 日主力連續買超，代表不是單日突發買盤。
        "main_buy_3d": bool(row["main_buy_3d_count"] >= 3),
        # 集保戶數下降，通常代表籌碼從散戶往少數人集中。
        "holder_count_down": bool(pd.notna(holder_count) and pd.notna(prev_holder_count) and holder_count < prev_holder_count),
    }


def is_price_weak(row: pd.Series, config: ScreenConfig = ScreenConfig()) -> bool:
    """A stock is price-weak when any listed price-weak condition is true."""

    return any(price_weak_conditions(row, config).values())


def is_chip_strong(row: pd.Series, config: ScreenConfig = ScreenConfig()) -> bool:
    """A stock is chip-strong only when all required chip conditions are true."""

    flags = chip_strong_conditions(row, config)
    required = (
        "main_buy_positive",
        "main_buy_over_1000",
        "main_buy_ratio_over_5",
        "count_diff_negative",
        "concentration_5d_stronger",
    )
    return all(flags[key] for key in required)


def is_excluded(row: pd.Series, config: ScreenConfig = ScreenConfig()) -> bool:
    """Apply hard exclusion rules before scoring."""

    cost_too_far = (
        pd.notna(row.get("top15_avg_price"))
        and row["top15_avg_price"] > 0
        and row["close_above_main_cost_pct"] > config.max_close_above_main_cost_pct
    )
    return bool(
        # 今日跌停附近排除，避免接刀。
        row["pct_change"] <= config.limit_down_pct
        # 成交量低於 3000 張排除，流動性不足。
        or row["volume"] < config.min_volume
        # 股價低於 20 元排除，避免低價股波動雜訊太大。
        or row["close"] < config.min_price
        # 今日漲幅超過 7% 不算價弱，不追已經強彈的標的。
        or row["pct_change"] > config.max_gain_for_price_weak
        # 收盤價高於主力均價太多，代表已離主力成本過遠，不追高。
        or cost_too_far
    )


def score_stock(row: pd.Series, config: ScreenConfig = ScreenConfig()) -> tuple[int, dict[str, bool]]:
    """Score one stock with the requested 100-point framework."""

    price = price_weak_conditions(row, config)
    chip = chip_strong_conditions(row, config)
    bonus = bonus_conditions(row)

    score = 0
    score += 15 if price["long_upper_shadow"] else 0
    score += 15 if price["close_near_low"] else 0
    score += 10 if price["black_k"] or price["intraday_spike_faded"] else 0
    score += 10 if price["break_short_ma"] else 0
    score += 15 if chip["main_buy_positive"] else 0
    score += 10 if chip["main_buy_ratio_over_5"] else 0
    score += 10 if chip["count_diff_negative"] else 0
    score += 10 if chip["concentration_5d_stronger"] else 0
    score += 15 if bonus["top15_cost_above_close"] else 0

    return min(score, 100), {**price, **chip, **bonus}


def classify_signal(row: pd.Series, score: int) -> str:
    """Classify the signal.

    可進場 requires score >= 80 and the next session closing back above today's
    midpoint.  If next-day data is unavailable, a high-score stock remains 強觀察.
    """

    next_day_reclaims_mid = pd.notna(row.get("next_close")) and row["next_close"] >= row["today_mid_price"]
    if score >= 80 and next_day_reclaims_mid:
        return "可進場"
    if score >= 70:
        return "強觀察"
    return "觀察"


def screen_stocks(
    daily: pd.DataFrame,
    main_chip: pd.DataFrame,
    branch_chip: pd.DataFrame | None = None,
    custody: pd.DataFrame | None = None,
    target_date: str | None = None,
    config: ScreenConfig = ScreenConfig(),
) -> pd.DataFrame:
    """Run the full screener and return formatted output rows."""

    df = merge_inputs(daily, main_chip, branch_chip, custody)
    if target_date:
        target = pd.to_datetime(target_date).date()
    else:
        target = df["date"].max()
    df = df[df["date"] == target].copy()

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if is_excluded(row, config) or not is_price_weak(row, config) or not is_chip_strong(row, config):
            continue

        score, _flags = score_stock(row, config)
        if score <= 60:
            continue

        rows.append(
            {
                "股票代號": row["stock_id"],
                "股票名稱": row.get("stock_name", ""),
                "收盤價": round(row["close"], 2),
                "今日漲跌幅": round(row["pct_change"], 2),
                "最高價": round(row["high"], 2),
                "成交量": int(row["volume"]),
                "主力買賣超": int(row["main_buy_sell"]),
                "主力買超佔成交量比例": round(_value(row, "main_volume_ratio"), 4),
                "家數差": int(row["count_diff"]),
                "5日集中度": round(row["concentration_5d"], 4),
                "20日集中度": round(row.get("concentration_20d", 0), 4),
                "買方Top15均價": round(row.get("top15_avg_price", float("nan")), 2),
                "收盤價與主力均價差距": round(row.get("main_cost_gap", float("nan")), 4),
                "分數": score,
                "訊號分類": classify_signal(row, score),
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not output.empty:
        output = output.sort_values(["分數", "主力買超佔成交量比例"], ascending=False)
    return output


def write_csv(result: pd.DataFrame, output_path: str | Path) -> None:
    """Write screener output as UTF-8 with BOM for Excel compatibility."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
