# 03 · 具體執行程式流程圖 + 注意事項

> 目的:助教的「操作手冊」。程式從頭到尾怎麼跑、每個檔案產出什麼、現場最常炸在哪、
> 怎麼當場 debug。學生卡住時,翻這份。

---

## 1. 整體流程圖(三條線共用前段)

```
                         ┌─────────────────── 共用前處理 ───────────────────┐
  cells.parquet ──▶ 取 9 channel ──▶ 丟空 cell ──▶ 切 train/val ──▶ 標準化 ──▶ X [N,9]
                         └──────────────── 存 mean/std(關鍵!)──────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            ▼                               ▼                               ▼
      ┌──────────┐                    ┌──────────┐                    ┌──────────┐
      │  PCA     │                    │   AE     │                    │   VAE    │
      │ 一步算   │                    │ 訓練迴圈 │                    │ 訓練迴圈 │
      └────┬─────┘                    └────┬─────┘                    └────┬─────┘
           │ W,Z                            │ checkpoint                    │ checkpoint
           ▼                               ▼                               ▼
      重建誤差/散點圖                 重建誤差/latent 圖              重建 + KL + 生成樣本
                                                                           │
                                                                           ▼
                                                                  從 N(0,I) 抽 z → decode → 新資料
```

---

## 2. 建議的檔案結構(每階一個可獨立跑的 script/notebook)

```
teaching/pca_ae_vae/code/
├── simple/            # 精簡註解版(已會 py 的人快速瀏覽)
│   ├── pca.py  ae.py  vae.py
├── detailed/          # 逐行詳解版(完全沒寫過 py 的同學)
│   ├── pca.py  ae.py  vae.py
├── pretrained/        # ★ 安全網:預訓練好的 ae.pt / vae.pt + mean_std.npz
└── outputs/           # 圖、重建誤差數字
```
- **每個檔自包含**(讀一個檔就懂全部,不跨檔 import)。simple / detailed 兩版**程式邏輯相同、只差註解密度**,結果一致。

**安全網原則**:`ae.py` / `vae.py` 檔案頂端有 `USE_PRETRAINED` 開關。學生現場訓不完 → 改成 `True` 即載入 `pretrained/` 權重、跳過訓練,後面照跑。**自己訓的和預訓練的存檔格式一致**(同一份程式),下游無痛接。

---

## 3. 每階「跑起來」的最小步驟

### 3.1 PCA(`pca.py`,numpy SVD)
```
load → standardize → SVD 求主軸 W → Z = X·W
     → X̂ = Z·Wᵀ → 印重建誤差 → 散點圖存檔
```
- 不用 GPU、不用訓練、秒出。**第一堂讓學生「成功一次」的最佳起點。**

### 3.2 AE(`ae.py`)
```
建模型 → 定 optimizer(Adam) → for epoch:
    for batch:  x̂ = model(x);  loss = MSE(x, x̂);  反向;  step
  → 存 ae.pt → 畫 encoder(X) 的 latent 散點
```

### 3.3 VAE(`vae.py`)
```
建模型 → for epoch:
    for batch:  x̂, μ, logσ² = model(x)            # 內含重參數化抽樣
                recon = Σ_channel(x−x̂)²(對 batch 平均)
                loss = recon + β·KL(μ,logσ²)        # recon 對 channel「加總」避免崩潰
                反向; step
  → 存 vae.pt → 畫 latent → 從 N(0,I) 抽 z → decode → 反標準化 → 存生成樣本
```
> ⚠️ 重建項務必對 channel **加總**(非平均),否則相對 KL 太弱會 posterior collapse(實測會卡在 recon≈變異)。

---

## 4. 注意事項 / 踩雷清單(★ = 現場最常發生)

### 資料層
- **★ 沒標準化**:loss 不降或 NaN;PCA 主軸全被 `gfa` 綁架。→ 一定先 z-score。
- **★ mean/std 用到驗證資料算**:資訊洩漏。→ 只用 train 算,存起來重用。
- **重建出來看不懂**:忘了**反標準化**。→ `x̂ * std + mean`。
- 空 cell 沒丟:latent 一大團擠在原點(全 0 樣本)。→ 先濾 `coverage≥0.02`。

### 神經網路層
- **★ loss 變 NaN**:learning rate 太大,或 VAE 的 `logσ²` 爆炸。→ lr 調小(1e-3→3e-4);對 `logσ²` 做 clamp(例如 [-6, 6])。
- **★ VAE 重建很糊、latent 縮成一點(posterior collapse)**:KL 權重 β 太大,模型偷懶把所有資料塞進 N(0,I) 中心、放棄重建。→ 把 β 調小,或用 **KL annealing**(β 從 0 慢慢升上來)。
- **★ 重參數化寫錯**:直接對 `z` 取樣而沒用 `μ + σ·ε`,結果無法反向傳播 / 不收斂。→ 一定寫成 `z = μ + exp(0.5*logvar) * torch.randn_like(μ)`。
- **σ 直接輸出(沒用 log)**:網路輸出負數時 σ 變負、開根號爆掉。→ 永遠輸出 `logσ²`,用時 `exp(0.5*logσ²)`。
- **忘了 `optimizer.zero_grad()`**:梯度累加,訓練亂掉。→ 每個 batch 開頭清零。
- **eval 沒切 `model.eval()` / `torch.no_grad()`**:畫 latent 時還在抽樣/算梯度,圖會抖、慢。

### 形狀層
- **★ shape 不合報錯**:最常見是忘了 batch 軸,或 latent 維設錯。→ 在每個 forward 後 `print(x.shape)` 對照 `02_DATAFLOW.md` 的表。
- **MSE 對到沒對齊的 tensor**:`x` 和 `x̂` 一個 `[B,9]` 一個 `[B,1,9]`。→ 對齊維度。

### 生成層
- **AE 拿去生成**:從 N(0,I) 抽 z 解出垃圾。→ **正常現象,不是 bug**;這正是要對比給學生看的「為什麼需要 VAE」。
- **VAE 生成也很糊**:資料太少 / latent 太小 / 訓練不足。→ 管理期待,或載入預訓練模型。

---

## 5. 快速 debug 對照表

| 症狀 | 最可能原因 | 一句話處置 |
|---|---|---|
| loss = NaN | lr 太大 / logσ² 爆 | lr↓、clamp logσ² |
| loss 不降 | 沒標準化 / 沒 zero_grad | 檢查前處理與訓練迴圈 |
| latent 全擠原點 | 沒丟空 cell / posterior collapse | 濾空 cell;β↓ 或 annealing |
| 重建看不懂 | 沒反標準化 | ×std+mean |
| 生成是垃圾 | 用了 AE 不是 VAE | 換 VAE;或這正是教學對比 |
| shape 報錯 | batch 軸/維度錯 | 逐層 print shape 對表 |

---

## 6. 現場節奏建議(給助教)

1. 先全班一起跑 **PCA**,確保「每個人都成功跑出一張圖」——建立信心。
2. AE:**先給能跑的完整程式**,讓學生**只填 encoder/decoder 的層**(TODO),按下訓練看 loss 下降。
3. VAE:重點口頭講 reparameterization 與 KL,**程式大部分給好**,讓學生改 `β` 觀察重建 vs 生成的拉鋸。
4. 任何一階卡住 → 把 `USE_PRETRAINED = True`,保證所有人都走到「生成」那一步。
