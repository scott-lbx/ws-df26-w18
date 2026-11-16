# code/ — 可執行程式

PCA / AE / VAE 三個模型,**每個都是獨立、自包含的檔案**(讀一個檔就懂全部,不用跳檔)。
每個模型有**兩個版本**,程式邏輯完全相同、只差註解密度:

```
simple/    pca.py  ae.py  vae.py     ← 精簡註解:快速瀏覽重要步驟(已會 py 的人)
detailed/  pca.py  ae.py  vae.py     ← 逐行詳解:給完全沒寫過 py 的同學
notebook/  PCA_AE_VAE_teaching.ipynb ← 教學用 Jupyter:一份涵蓋 PCA→AE→VAE,由上往下跑
pretrained/                          ← 安全網:訓好的模型 + 標準化參數
outputs/                             ← 跑出來的圖與結果(可重生,不進 git)
requirements.txt
```

> ⚠️ **資料注意**:目前的範例與 `pretrained/` 模型都是用 repo 內**示範資料**(`v03/cache/cells.parquet`)
> 訓練,**不是最完整/最正確的原始資料**。正式資料(見 `../06_INTERFACE_HANDOFF.md`)到位後,
> 把各檔頂端的 `DATA` 改成正式路徑、重新訓練,數字與模型才有效。

## 教學 Notebook
`notebook/PCA_AE_VAE_teaching.ipynb` 是課堂主用版:markdown 概念 + 可執行程式交錯,
從載入資料一路到 VAE 生成,含「🔧 練習」(改 BETA / LATENT)。開啟:
```bash
pip install jupyter
jupyter notebook notebook/PCA_AE_VAE_teaching.ipynb
```

## 安裝
```bash
pip install -r requirements.txt
```
- PCA 只需 numpy/pandas/pyarrow。AE/VAE 需要 PyTorch(CPU 版即可)。matplotlib 沒裝也能跑(只是不出圖)。

## 跑(依序)
```bash
cd simple        # 或 cd detailed,兩版結果一樣
python pca.py    # PCA,秒出
python ae.py     # AE,訓練幾分鐘
python vae.py    # Beta-VAE,會生成新格子
```

## 沒有命令列參數 —— 設定都在檔案最上面
為了讓不熟 py 的同學好懂,**不用命令列旗標**。要調整就直接改檔案頂端的設定:

| 變數 | 在哪個檔 | 作用 |
|---|---|---|
| `USE_PRETRAINED` | ae.py / vae.py | 改成 `True` → 載入 `pretrained/` 的現成模型、跳過訓練(安全網) |
| `BETA` | vae.py | 重建 vs 整齊的旋鈕;試 `0.1` 與 `5` 看差別 |
| `EPOCHS` | ae.py / vae.py | 訓練輪數 |
| `LATENT` | 三個檔 | 壓成幾維(預設 2,為了畫圖) |
| `DATA` | 三個檔 | 輸入表格路徑(預設 repo 內示範資料) |

## 安全網怎麼用
- **學生**:訓不完 → 把 `USE_PRETRAINED = True`,直接用訓好的模型走到「生成」。
- **老師**(真實資料到位後重產官方模型):跑一次 `python ae.py` / `python vae.py`(`USE_PRETRAINED=False`),
  它會把模型存到 `outputs/`,再把 `outputs/ae.pt`、`outputs/vae.pt` 複製到 `pretrained/` 覆蓋即可。

## 預期結果(repo 內示範資料 N=5785, C=9)
| 模型 | 驗證重建誤差 | 能生成 |
|---|---|---|
| PCA | ~0.55 | 勉強 |
| AE | ~0.08(最會壓) | ❌ |
| VAE | ~0.14(略遜 AE) | ✅ |

→ VAE 用一點點重建品質,換來「可從亂數生成」的整齊 latent。

## 輸入資料從哪來
見 `../06_INTERFACE_HANDOFF.md`(上游契約)。沒有真實資料時,預設讀 repo 內 `v03/cache/cells.parquet` 當示範。
