# 手機獨立使用部署說明

這個版本可以部署到 Render，部署後 iPhone 直接開 HTTPS 網址即可使用，不需要電腦一直開著。

## Render 部署方式

1. 把此專案推到 GitHub。
2. 到 Render 建立 `Blueprint` 或 `Web Service`。
3. 如果用 Blueprint，Render 會讀取根目錄的 `render.yaml`。
4. 部署完成後，打開 Render 提供的網址，例如：

```text
https://tw-chip-rebound.onrender.com
```

## FinMind Token

FinMind token 不寫在雲端伺服器，也不寫在 Git 裡。

建議部署後在 Render 的 Environment 設定：

```text
FINMIND_TOKEN=你的 FinMind token
```

這樣手機打開網頁就不用再輸入 token。若雲端沒有設定 `FINMIND_TOKEN`，頁面才會要求在 iPhone Safari 貼上 token；token 會存在該手機瀏覽器的 localStorage。

## 注意

Render Free 方案服務閒置後可能休眠，第一次打開會比較慢。若希望更穩定，可以改用付費方案。
