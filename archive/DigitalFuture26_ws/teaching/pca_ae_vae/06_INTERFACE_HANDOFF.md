# 06 · 介面交接文件 — 上游資料 → 我們的 ML 程式

> **這份是「契約」。** 上游(持有 60+GB 原始上海資料、負責建接口的老師)只要產出符合本文件的
> **一個表格檔**,我們的 PCA / AE / VAE 程式就能直接吃,雙方不必知道對方內部怎麼做。
>
> 一句話:**你把原始資料聚合成「每個 grid cell 一列、每列幾個 channel 數字」的表格,交給我們。**
> channel **要給哪些、給幾個由你決定**,我們的程式自動採用(見 §3)。

---

## 1. 你要交付什麼(就一個檔)

- **格式**:Parquet(`.parquet`,優先)或 CSV。
- **內容**:一張表,**一列 = 一個 grid cell**;每列含**任意數量**的數值 channel 欄(+ 身分欄)。
- **放哪**:預設 `v03/cache/cells.parquet`;或任意路徑,改腳本頂端的 `DATA` 指定。
- 我們**不需要**原始建築逐棟資料、幾何、stakeholder 標籤(那條線是另一位老師)。你只要給**聚合後的數字**。

---

## 2. grid 怎麼定義(你聚合時的網格規則)

| 項目 | 值 | 說明 |
|---|---|---|
| 一個窗 window | 1000 m × 1000 m | 一個基地 |
| 一格 cell | 50 m × 50 m | 聚合單位 |
| 每窗格數 | 20 × 20 = 400 | `row`,`col` ∈ 0..19 |
| 多個窗 | 各自 20×20,用 `window` 區分 | |

> 格子大小/數量可改,但**全部窗要一致**,且要在交付時告訴我們數字。

---

## 3. Channel 欄位:**數量不限,你決定要給什麼**

> 💡 **我們不限制你給哪些、給幾個 channel。** 我們的程式會**自動把表格裡「所有數值欄」當成 channel**
> (排除位置/名稱等保留欄,見 §3.2),模型維度 `C` 自動跟著你給的數量變。
> 你可以給 5 個、9 個、30 個都行——**給越多資訊,模型能學的越豐富**。

### 3.1 範例 channel(這只是「示範」,不是規定清單)
下面是目前示範資料用的 9 個。你可以照給、可以增減、可以換成別的——**只要是數值欄就會被自動採用**:

| 範例欄名 | 型別 | 單位/範圍 | 每格的聚合規則(範例) |
|---|---|---|---|
| `gfa` | float | m²,≥0 | 格內各棟 GFA 加總 |
| `coverage` | float | 0–1 | 格內 footprint 面積和 ÷ 格面積,上限 1 |
| `mean_h` | float | m,≥0 | 格內各棟高度的平均 |
| `sh_house` / `sh_slab` / `sh_tower` / `sh_large` / `sh_mid` | float | 0–1 | 各建築家族的 footprint 佔比 |
| `dom_sem_share` | float | 0–1 | 格內主導機能的佔比 |
| *(可自由新增)* `poi_gov`, `poi_office`, `age_mean`, `dist_road`… | float | 任意 | 你想加的任何數值資訊 |

> `coverage` 這欄**若有給**,我們會用它(<0.02)自動判定並丟掉空格;沒給也能跑(就不丟空格)。

### 3.2 保留欄(會被自動「跳過」,不當 channel)
這些是位置/名稱/幾何/舊標籤等非數值輸入,**就算在表裡也不會被當 channel**:
```
window, row, col, role, name, geometry, bid, bidx,
n_bld, token, token_v03, token_v04, judge, semantic, family
```
其中 `window` / `row` / `col`(身分欄)**強烈建議一起給**,讓我們能把結果**貼回地圖**。

> 想精準控制只用某幾欄?在我們的腳本頂端把 `CHANNELS` 從 `None` 改成你的欄名 list 即可,
> 那就只用你指定的那些(其餘忽略)。

---

## 4. 硬規則(這些才是真的「必須」)

1. **不可有 NaN / Inf。** 空格(格內沒建築)請把各 channel **全填 0**。
2. **給原始尺度的真實數值**(真的公尺、真的 m²)。**不要自己標準化/正規化**——z-score 由我們做(只用 train 算,避免洩漏)。
3. **每個 channel 是數值欄**(float/int);文字類別請先轉成數值(例如 one-hot 或比例)再給。
4. **一列一格,不要一列一棟。** 交付的是**聚合後**的格,不是建築明細。
5. **欄名穩定、不要改名。** 要新增 channel 直接加欄即可(數量不限);改既有欄的語意請升版(見 §7)。

---

## 5. 擴充 channel(未來要把資料變更豐富時)

**流程更簡單了——你不必通知我們改程式:**

1. 在表格**多加數值欄**(例如 `poi_gov`, `poi_office`, …, `age_mean`, `dist_road`)。
2. 就這樣。我們的程式**自動偵測**到新欄、`C` 自動變大,**不用改任何程式**。

> ⚠️ 唯一要注意:**channel 數一變,舊的預訓練模型就不能用了**(維度不合)。
> 程式會偵測到並提醒你「請重新訓練」。用新資料重訓一次即可(見 `code/pretrained/README.md`)。
> 提醒:`age` 多集中在 1985 後、`informal` 類無真實訊號——這些「資料界線」請隨表附註(我們不解讀,但要如實傳給分析線老師)。

---

## 6. 交付前自我驗證(30 秒)

把表存好後,跑這段檢查(不檢查「有沒有特定欄」,只檢查「格式對不對」):

```python
import pandas as pd, numpy as np
NON = {"window","row","col","role","name","geometry","bid","bidx",
       "n_bld","token","token_v03","token_v04","judge","semantic","family"}
df = pd.read_parquet("cells.parquet")
# 自動找出 channel:所有數值欄,排除保留欄
feats = [c for c in df.columns
         if c not in NON and pd.api.types.is_numeric_dtype(df[c])]
assert len(feats) >= 1, "至少要有一個數值 channel 欄"
assert df[feats].isna().sum().sum() == 0, "有 NaN(空格請填 0)"
assert np.isfinite(df[feats].to_numpy()).all(), "有 Inf"
print(f"OK:{len(df)} 列,偵測到 {len(feats)} 個 channel → {feats}")
```

全部通過 → 直接交給我們,跑任一支 `python pca.py` 會印出 `N / C / channels` 確認接上。

---

## 7. 版本與聯絡

- **Schema 版本**:`v1`(本文件)。channel 清單見 §3。
- 改動 channel 語意 → 升版號,並在交付時註明。
- 任何「該怎麼聚合」的灰色地帶 → 先問,不要各自猜(尤其家族佔比的分母、主導機能的定義)。

---

## 8. 一頁速記(貼在交付信裡)

> 交付一個 parquet:**一列一格(50m)**,含**任意數量的數值 channel 欄**(要給什麼、給幾個你決定)+ `window,row,col`;**原始尺度、不要標準化、不可有 NaN(空格填 0)、channel 用數值欄**。我們的程式自動採用所有數值欄、`C` 自動適應,接上後跑 PCA/AE/VAE。
