# 07 · Latent Urban —— NN 對都市資料的潛空間教學

用深度學習更深入分析 **05 Shanghai Power to Form** 的建築資料:把城市形態壓進
**連續、可採樣**的潛空間,讓「權力重新分配」能被量化、比較、甚至生成。

## 五本教學 notebook(依序跑)

| # | notebook | 資料層 | 學什麼 |
|---|---|---|---|
| 1 | [都市數據像量化](1_都市數據像量化.ipynb) | 建築 | 05 資料 → **兩種張量化**:表格向量 + 高度影像 |
| 2 | [統計方法壓縮-PCA_TSNE_ISOMAP](2_統計方法壓縮-PCA_TSNE_ISOMAP.ipynb) | 建築 | PCA/t-SNE/Isomap:解釋性 vs 視覺分群 |
| 3 | [深度學習壓縮-AE_VAE_CNN](3_深度學習壓縮-AE_VAE_CNN.ipynb) | 建築 | AE/VAE(表格)+ **Conv-VAE(CNN 看影像)**;**存模型** |
| 4 | [分析比較](4_分析比較.ipynb) | 體制 | 群聚 vs 手動 regimes(KMeans)、解釋性、連續性 |
| 5 | [用於生成與權力重新分配](5_用於生成與權力重新分配.ipynb) | 體制 | 重建 vs 生成、體制形態簽名(權力重分配) |

> **兩個分析層**:NB1-3 = 建築層(單城每棟樓,3263×16;含 CNN 影像);
> NB4-5 = 體制層(12 城 × 5 情形 = 60×9,接 regimes/權力)。
> 各用自己的 kernel 跑(兩層 `config` 不同)。

## 程式架構(notebook 只呼叫,細節藏這)

```
nn/                建築層 + CNN
  preprocess.py    parquet → 表格特徵矩陣
  models.py        PCA / AE / VAE(表格)
  raster.py        城市 footprint+高度 → 高度影像 → patch
  conv.py          Conv-VAE(CNN);pos_weight 防 posterior collapse
  io.py            save_model / load_into(訓練一次到處用)
  viz.py           潛空間散佈/重建表/latent grid
  _font.py         中文字型自動修好(import nn 即生效)
regime_latent/     體制層(城市×體制形態,借 05 引擎現建語料)
  corpus.py        12 城 × 5 情形 → out/features.csv
  encoders.py      Standardizer + PCA/AE/VAE(帶訓練history/β退火)
  analysis.py      PCA/t-SNE/Isomap 比較、AE/VAE loss、重建
  interpret.py     silhouette(體制 vs 家族)、PCA 載荷、體制簽名
  cluster.py       KMeans vs 手動 regimes(ARI)
  generate.py      採樣/生成能力(plausibility/validity/interp)
out/               features.csv · figs/ · models/ · 研究报告.md
_archive/          舊 01-06 notebook(歸檔)
_legacy_workflow/  舊多代理 workflow(未用)
```

## 依賴

`torch scikit-learn matplotlib pandas numpy geopandas rasterio shapely pyproj`
(不需 umap:第三種降維用 sklearn Isomap 替代)。資料源:`05 .../data/<slug>/buildings.parquet`。

## 核心結論(給同學的啟發)

| 目的 | 最佳方法 | 為什麼 |
|---|---|---|
| 壓縮/重建 | AE | 非線性瓶頸,重建誤差最低 |
| 採樣/生成 | **VAE** | 先驗 N(0,I),生成最真、插值最連續 |
| 可解釋軸 | PCA | 軸=特徵線性組合 |
| 空間紋理 | Conv-VAE | 卷積看街廓排列 |
| 探索可視化 | t-SNE / Isomap | 分群清楚但**不能生成**(無 decoder) |

詳見 [out/研究报告.md](out/研究报告.md)。
