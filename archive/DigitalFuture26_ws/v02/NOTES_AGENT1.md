# NOTES_AGENT1 — v0.2 Learn：POI 單一主判 ＋ 50 m 格網 token 文法

> 2026-06-12。依 JZ 的改向指示重做 STEP L：probe 的五源融合級聯（EULUC→AOI→OSM/Function→POI→形態）
> 太複雜、教不動。v0.2 改成 **一個主判＋補位**：POI 是唯一主源（人工貢獻的開放資料，
> 非 AI 生成——AI 衍生圖層如 EULUC、AI 解析層明確不作主判）。教學敘事一句話：
> **「POI 說了算，POI 沉默時才問別人」**。
>
> 重跑：`python3 run_learn_v02.py all`（或分塊 `extract <patch>` ×3 → `step1` → `step2`）。
> OSM/Function/EULUC 的逐棟接合直接重用 `../probe/cache/`（bid 鍵完全一致）；
> POI 以 30 m 重新抽取（`cache/<patch>_poi30.parquet`）。
> POI shapefile 幾何已驗證＝wgs84 欄位（偏移在 gcj 欄），幾何可信，直接投影使用。

---

## 1. 規則（全部，就這些）

**語義主判（一條可教規則）**：每棟建築取 **30 m 內權力相關 POI 類的多數決**
（比 probe 的 50 m 更緊，減少錯誤掛接）；**平手 → 取最近那顆 POI 的類**。
權力相關 = bigType 能映射到 8 個 canonical 類者（剔除 地名地址信息；映射不到的 bigType 不投票）：
政府机构及社会团体→public_admin、金融保险→office_finance、医疗保健→healthcare、
科教文化→education、购物/餐饮/生活/住宿/体育休闲→commercial_retail、公司企业→office_finance、
商务住宅依 midType 分流（住宅→residential_estate／楼宇→office_finance／产业园→industrial）。

**POI 沉默（30 m 內無票）才依嚴格順位問補位**：OSM 带type（含名稱正則弱補）→
带年份 Function（Age 一併保留，Agent 2 先驗要用）→ EULUC parcel（class2）→
形態補底（僅 floors≤3 且 fp<400 → residential_lowrise，其餘不硬猜）→ none。
每棟記 `judge ∈ {poi, osm, function, euluc, morph, none}`。

**住宅內分**（對所有 judge 一致）：floors≤3 且 fp<400 → residential_lowrise，否則 residential_estate。

**形態家族**（與 probe 完全相同的級聯與閾值）：large_footprint（fp≥2000）→
tower（h≥35 且長寬比<2）→ slab（長寬比≥2 且 4–12 層）→ house（floors≤3 且 fp<400，最後蓋章）→ midrise_other。

## 2. 覆蓋——誰判的（% of buildings）

| patch | **POI 主判** | osm | function | euluc | morph | none | 有標籤合計 |
|---|---|---|---|---|---|---|---|
| 陸家嘴 (n=322) | **74.5** | 2.8 | 2.8 | 9.0 | 0.6 | 10.2 | 89.8 |
| 老西門 (n=1962) | **70.3** | 1.8 | 9.2 | 16.3 | 0.8 | 1.6 | 98.4 |
| 曹楊 (n=523) | **51.4** | 9.2 | 14.7 | 19.9 | 1.7 | 3.1 | 96.9 |

曹楊 POI 最稀（小區內部建築 30 m 內常無票），EULUC/Function 補掉三成；
陸家嘴 none 最高（公園綠地/交通設施上的建築，連補位都沉默）。

## 3. 與舊五源融合的一致率（誠實數字）

bid 直接對齊 probe `fused_buildings.fgb`，限雙方都有標籤：

| patch | POI 主判 ∩ 舊標籤 一致率 | 全部 v0.2 標籤 ∩ 舊標籤 一致率 |
|---|---|---|
| 陸家嘴 | **47.5%** (n=223) | 55.2% (n=270) |
| 老西門 | **34.3%** (n=1364) | 50.1% (n=1905) |
| 曹楊 | **25.3%** (n=265) | 53.0% (n=498) |

不一致的主力（POI 主判 vs 舊融合，三 patch 合併）：
commercial_retail↔residential_estate **501 棟**、public_admin↔residential_estate **213 棟**、
commercial_retail↔residential_lowrise 82、public_admin↔residential_lowrise 62。
解讀：舊融合以 EULUC parcel 為先驗，整片住宅地塊塗成住宅；POI 主判把**街面有店的住宅樓**
判成 commercial_retail、把**小區裡掛著居委會/社團 POI 的樓**判成 public_admin
（高德 bigType 政府机构及社会团体 包含居委會）。這不是 bug，是兩種世界觀：
parcel 看土地、POI 看門牌。低一致率本身就是 v0.2 最有教學價值的一張表——
換一個主判，老西門從「住宅城」變成 36% 商業（708/1962）。

## 4. Token 文法（50 m 格，20×20×3 patch＝1200 格）

特徵 9 維：GFA、coverage、平均高度、5 個家族份額（footprint 面積佔比）、主導語義份額。
coverage<0.02 → 保留 token T0=EMPTY（407 格）；其餘 793 格 z-score 後 KMeans k=6，
依平均 GFA 升冪編號。**標籤只用量出來的數字**：

| token | 量測標籤 | 格數 | 池內建築 |
|---|---|---|---|
| T0 | EMPTY: cov<.02 | 407 | — |
| T1 | cov .24 h̄ 12m house .81 | 107 | 471 |
| T2 | cov .34 h̄ 20m slab .87 | 214 | 600 |
| T3 | cov .46 h̄ 16m mid .55 | 166 | 1011 |
| T4 | cov .26 h̄ 31m mid .91 | 206 | 496 |
| T5 | cov .33 h̄ 72m tower .85 | 85 | 187 |
| T6 | cov .99 h̄ 94m large .86 | 15 | 22 |

直觀對應（僅口頭輔助，不寫進標籤）：T1≈里弄/低層、T2≈板樓行列、T3≈高覆蓋混雜中層、
T4≈點式中高層、T5≈塔樓群、T6≈陸家嘴巨構。鄰接（4 鄰、合併、列正規化）：
對角優勢明顯（T0-T0 .61、T2-T2 .35、T3-T3 .35）；T5 最常鄰 T0（.41，塔樓配空地/廣場）；
T6 幾乎只鄰 T0（.59）與 T4（.26）——巨構被空格與中高層包著。
生成端從 token 池實例化：`token_pools.parquet`（bidx 指回 fused 表＋格內 dx/dy 偏移）、
`token_pools.fgb`（footprint 平移到 50 m 格局部座標）。

## 5. 誠實侷限

1. **單一主判全盤繼承 POI 的街面偏壓**：店鋪 POI 沿街密集、小區 POI「一區一點」，
   所以街面樓被判商業、小區內部靠 EULUC 補位。一致率 25–48% 是真實代價，不要美化。
2. **居委會問題**：政府机构及社会团体 含居委會/業委會，把住宅樓翻成 public_admin（275 棟）。
   可用 midType 細分修，但那就違反「一條規則」的教學約束——v0.3 再議。
3. **30 m 仍會錯掛**：老西門窄巷兩側互相吃票；tie→最近只解平手，不解錯掛。
4. **coverage 用質心歸格**：大底盤建築整棟算進質心所在格（T6 cov 飽和在 1.0 是上限截斷）；
   池內 footprint 會溢出 50 m 格（局部座標範圍 −75〜+148 m），實例化時需自行裁切或讓溢。
5. **token 文法只見過 3 個 patch**：793 個非空格、T6 只有 15 格／22 棟，鄰接列極薄；
   k=6 是選定不是推定。Age 覆蓋仍只 21.7–45.3%（陸家嘴最低），照舊留給 Agent 2 條件化。
6. morph 補底只敢判 house→lowrise，其餘留 none（陸家嘴 10.2%）——寧缺勿濫。

## 檔案清單

| 檔案 | 內容 |
|---|---|
| `run_learn_v02.py` | extract（POI 30 m）＋ step1（主判融合/FIG1）＋ step2（token/FIG2） |
| `fused_v02.fgb` / `fused_v02.parquet` | 逐棟：semantic, judge, family, height, fp_area, floors, age |
| `fig1_learn_poi.png` | 3 patch 語義地圖＋「誰判的」疊條＋與舊融合一致率 |
| `fig2_grammar.png` | 3 patch token 格網＋鄰接熱圖＋逐 token 量測剖面 |
| `tokens.json` / `adjacency.json` / `token_grid_*.npy` | token 詞彙、鄰接、逐 patch 格網 |
| `token_pools.parquet` / `token_pools.fgb` | 逐 token 實例化池（格局部座標） |
| `results_v02.json` | 覆蓋、一致率、top 不一致、token 統計 |
| `cache/*_poi30.parquet` | POI 30 m 多數決中間產物 |
