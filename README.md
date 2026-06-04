# 台股「價弱籌碼強」選股程式

這是一個模組化 Python 選股器，用來找出短線股價偏弱、但籌碼面轉強的台股，適合收盤後觀察隔日或後續 2 到 5 日反彈機會。

## 安裝

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 輸入資料

支援 CSV，欄位可用英文或常見中文名稱。

日 K：

```text
date,stock_id,stock_name,open,high,low,close,volume,avg_price
```

主力籌碼：

```text
date,stock_id,main_buy_sell,buyer_count,seller_count,count_diff,concentration_5d,concentration_20d
```

分點 Top15：

```text
date,stock_id,top15_avg_price,top15_brokers
```

集保戶數，可省略：

```text
date,stock_id,holder_count
```

## 執行

```bash
python3 -m tw_chip_rebound.cli \
  --daily data/daily.csv \
  --main data/main_chip.csv \
  --branch data/branch_chip.csv \
  --custody data/custody.csv \
  --date 2026-06-04 \
  --output outputs/price_weak_chip_strong.csv
```

如果沒有集保資料，拿掉 `--custody` 即可。

## iOS 版本

iPhone 或 iPad 建議使用手機網頁版，會自動串接 FinMind 公開資料，不需要手動上傳 CSV。

啟動本機伺服器：

```bash
python3 -m tw_chip_rebound.mobile_server
```

電腦和手機在同一個 Wi-Fi 時，用 iPhone Safari 開啟電腦的區網網址，例如：

```text
http://你的電腦IP:8787
```

第一次使用時輸入 FinMind token，之後 token 會存在 iPhone 瀏覽器本機。按「更新掃描」後會自動抓：

- `TaiwanStockPrice`：全市場日 K 價量
- `TaiwanStockInstitutionalInvestorsBuySell`：三大法人買賣超，作為自動化主力買超來源
- `TaiwanStockTradingDailyReport`：候選股分點資料，用於 Top15 均價與家數差

原本的單檔上傳版仍保留在 `outputs/ios_stock_screener.html`，但主要版本是 `outputs/mobile_app.html`。

## 不靠電腦的雲端版本

如果電腦不會一直開著，請部署到 Render。部署後手機可直接開 Render 的 HTTPS 網址使用。

本專案已包含：

- `render.yaml`
- `/api/ping` 健康檢查
- Render 啟動指令：`python -B -m tw_chip_rebound.mobile_server --host 0.0.0.0 --port $PORT`

詳細看 `DEPLOY.md`。

## 訊號分類

- `觀察`：分數 61 到 69。
- `強觀察`：分數 70 以上。
- `可進場`：分數 80 以上，且資料中已有下一交易日收盤價站回今日中位價。
- `60 分以下`：不輸出。

注意：當天收盤後尚未有隔日資料時，即使分數超過 80，也會先列為 `強觀察`。隔日更新日 K 後重新執行，若站回前一日中位價，才會升級為 `可進場`。
