"""Command line interface for the Taiwan stock screener."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_branch_chip, load_custody, load_daily_k, load_main_chip
from .screener import screen_stocks, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台股價弱籌碼強選股程式")
    parser.add_argument("--daily", required=True, help="日 K CSV：date, stock_id, open, high, low, close, volume")
    parser.add_argument("--main", required=True, help="主力買賣超 CSV")
    parser.add_argument("--branch", help="分點買賣超 / Top15 均價 CSV")
    parser.add_argument("--custody", help="集保戶數 CSV，可省略")
    parser.add_argument("--date", help="選股日期，格式 YYYY-MM-DD；省略時使用資料中最新日期")
    parser.add_argument("--output", default="outputs/price_weak_chip_strong.csv", help="輸出 CSV 路徑")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    daily = load_daily_k(args.daily)
    main_chip = load_main_chip(args.main)
    branch_chip = load_branch_chip(args.branch) if args.branch else None
    custody = load_custody(args.custody) if args.custody else None

    result = screen_stocks(
        daily=daily,
        main_chip=main_chip,
        branch_chip=branch_chip,
        custody=custody,
        target_date=args.date,
    )
    write_csv(result, args.output)
    print(f"完成：{args.output}，共 {len(result)} 檔")


if __name__ == "__main__":
    main()
