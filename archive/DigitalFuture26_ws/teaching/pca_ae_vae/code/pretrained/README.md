# pretrained/ — 安全網模型

學生現場訓不完時,把腳本頂端的 `USE_PRETRAINED = True`,就會載入這裡的模型、跳過訓練,
流程不卡,照樣走到「重建」與「生成」。

| 檔 | 內容 |
|---|---|
| `ae.pt` | 預訓練 AE 權重(含 c_in / latent) |
| `vae.pt` | 預訓練 Beta-VAE 權重 |
| `mean_std.npz` | 標準化用的 mean / std(反標準化、生成還原數字都要用) |

## ⚠️ 注意
目前這裡的模型是用 repo 內的**示範資料**(`v03/cache/cells.parquet`,N=5785, C=9)訓的,
只供「程式跑得起來」的展示。

**當上游交付真實接口資料後**,請老師用真實資料重新產生官方預訓練檔:
1. 確認腳本頂端 `USE_PRETRAINED = False`、`DATA` 指到真實資料。
2. 跑 `python ae.py` 與 `python vae.py`(模型會存到 `outputs/`)。
3. 把 `outputs/ae.pt`、`outputs/vae.pt`、`outputs/..` 對應的 `pretrained/mean_std.npz`
   複製到本資料夾覆蓋。
   (`mean_std.npz` 在每次跑 ae/vae 時就會自動寫到 `pretrained/`。)

學生之後把 `USE_PRETRAINED = True` 就會用到你這份新的官方模型。
