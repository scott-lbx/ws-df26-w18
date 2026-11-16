# NOTES_AGENT1 — v0.3 Learn at scale：24 個 1 km 窗口、TRAIN-only 文法、連續簽名雲

> 2026-06-12。業主最新指示（**覆寫 v0.2 的簡化路線**）：語義識別回到 probe 原版的
> **五源融合級聯**（EULUC parcel 先驗 → AOI 覆寫 → OSM-type/Function 覆寫 →
> POI 50 m 多數決補空（≥2 票）→ 形態補底）——POI 單一主判留作 v0.2 教學變體，不再是主線。
> 融合與形態家族規則**逐字重用** `../probe/run_learn.py`；v0.3 的工作是把它從 3 個 patch
> 擴到 24 個窗口、把文法學習嚴格限制在 TRAIN 子集、並產出連續簽名雲。
>
> 重跑（分步、皆有快取、seed 0）：
> `python3 run_v03.py select` → `extract A01 A02 ...`（4-6 個一批）→ `library` →
> `learn` → `signatures` → `fig1` / `fig2`。
> 向量一律 FlatGeobuf（經 /tmp 複製；本掛載點 **GPKG/sqlite 會壞，不用**）。

---

## 1. 選樣與分層（split.json）

候選＝`data_collection/atlas_500m/areas.csv` 的 50 個策展區（10 個 type group）。
規則：每 group ≥2、最大的 4 個 group（new_town／cbd_office／historic_lilong／
developer_resid）各 +1 → 24；**A01／A10／A13 強制入選並強制 TRAIN**（與 probe/v02 接續）；
與已選窗口重疊 >30% 者剔除（實際無人觸發，最近的 A01–A07 重疊 23%）；
切分 18 TRAIN／6 HELDOUT，依 group 分層（3 額度 group 各出 1 個 heldout，再由 seed 0
隨機抽 2 個 2 額度 group 各出 1 個）。**type group 名稱只是抽樣框 metadata**，
留在 split.json 供審計，圖上一律用地名。

| type group | TRAIN | HELDOUT |
|---|---|---|
| cbd_office | A01 陸家嘴、A04 徐家匯 | A02 南京東路/人民廣場 |
| historic_lilong | A10 老西門、A09 新天地 | A07 外灘 |
| worker_village | A13 曹楊、A17 天山新村 | — |
| developer_resid | A21 金橋住區、A19 聯洋花木 | A20 三林 |
| new_town | A26 嘉定新城、A27 臨港 | A29 奉賢南橋 |
| urban_village | A33 浦東城郊村、A32 閔行城郊村 | — |
| industrial | A36 張江 | A35 漕河涇 |
| heritage_watertown | A41 楓涇、A39 七寶 | — |
| midrise_resid | A45 北蔡 | A43 涼城 |
| waterfront_redev | A47 西岸、A49 楊浦濱江 | — |

## 2. 逐窗口處理（24×，cache/<id>_fused.parquet）

1 km 窗口從「上海市_建筑-AI影像解译-带高度.shp」切出（bbox ±0.0078° 讀取 →
EPSG:32651 → 精確 1 km 方格裁切 → fp_area>5 m² → height 下限夾 2 m →
floors=round(h/3.5)≥1），再做 probe 級聯融合（8 個 canonical 語義類）與形態家族
（house／slab／tower／large_footprint／midrise_other，閾值同 probe）。
合併庫 `library.parquet`/`library.fgb`：**14,527 棟**。

**融合覆蓋率（fused_labeled %，誰判的見 results_v03.json）**：

| 窗口 | labeled% | 窗口 | labeled% | 窗口 | labeled% | 窗口 | labeled% |
|---|---|---|---|---|---|---|---|
| A17 天山 | 99.8 | A29 南橋 | 99.6 | A32 閔行村 | 99.6 | A04 徐家匯 | 99.6 |
| A21 金橋 | 99.3 | A39 七寶 | 98.9 | A43 涼城 | 98.8 | A02 南京東路 | 98.5 |
| A09 新天地 | 97.7 | A36 張江 | 97.7 | A10 老西門 | 97.4 | A20 三林 | 97.2 |
| A13 曹楊 | 96.9 | A33 浦東村 | 96.3 | A35 漕河涇 | 95.7 | A19 聯洋 | 95.3 |
| A26 嘉定 | 94.0 | A49 楊浦濱江 | 91.3 | A01 陸家嘴 | 87.9 | A41 楓涇 | 85.0 |
| A45 北蔡 | 84.2 | A07 外灘 | 80.8 | A27 臨港 | 78.9 | **A47 西岸** | **31.0** |

A01/A10/A13 與 probe 完全一致（87.9/97.4/96.9）——管線複製驗證通過。
彙整（pooled judge 份額）：function+morpho 28.4%、aoi 23.6%、euluc+morpho 17.1%、
poi 補空只佔 7.8%、function 7.0%、osm 5.7%、euluc 5.5%、none 4.9%。

## 3. Token 文法 k=8（TRAIN-only；tokens.json / adjacency.json / token_pools.*）

50 m 格 20×20×24 窗。9 維特徵（GFA、coverage、平均高、5 家族份額、主導語義份額）
z-score 後 KMeans **k=8**，**只用 18 個 TRAIN 窗的 7200 格擬合**（非空 4254 格）；
coverage<0.02 → T0=EMPTY 保留不聚類。HELDOUT 格只做最近質心指派
（供骨架/簽名用），**不進詞彙、不進鄰接、不進實例化池**。標籤只寫量測數字：

| token | 量測標籤 | TRAIN 格 | HELDOUT 格 | 池內建築 |
|---|---|---|---|---|
| T0 | EMPTY: cov<.02 | 2946 | 869 | — |
| T1 | cov .17 h̄ 10m house .95 | 627 | 133 | 1841 |
| T2 | cov .23 h̄ 14m house .43（混雜低層） | 405 | 87 | 1330 |
| T3 | cov .31 h̄ 22m slab .92 | 1430 | 669 | 3014 |
| T4 | cov .21 h̄ 32m mid .92 | 827 | 292 | 1633 |
| T5 | cov .63 h̄ 22m mid .61（高覆蓋中層） | 477 | 157 | 2341 |
| T6 | cov .31 h̄ 60m tower .87 | 397 | 145 | 869 |
| T7 | cov .98 h̄ 29m large .94（大底盤） | 70 | 41 | 118 |
| T8 | cov .99 h̄ 94m large .83（巨構） | 21 | 7 | 29 |

對照 v0.2 的 k=6：k=8 把舊 T6「巨構」拆成 T7（中高大底盤＝商場/廠房裙樓）與
T8（超高巨構＝陸家嘴型），並把低層分出純里弄/村宅（T1）與混雜低層（T2）。
鄰接（4 鄰、TRAIN 18 窗、列正規化）：對角優勢照舊（T0-T0 .67、T3-T3 .46、T1-T1 .43）；
T8 仍幾乎只鄰 T0（.67）——巨構被空格包圍的 v0.2 觀察在 6 倍樣本下仍成立；
T7 最常鄰 T0（.42），大底盤同樣靠空地/停車場。

## 4. 簽名雲讀法（signatures.json／fig1 右）

每窗簽名＝9 維 token 組成 ＋ 10 指標（FAR、coverage、h p50/p90、median fp、
5 家族份額，家族份額以 footprint 面積計）→ TRAIN 統計 z-score → **PCA 只 fit TRAIN、
24 窗全投影**。PC1 29.8%＋PC2 27.0%＝57% 變異。

- **PC1 ≈ 強度/垂直度軸**（+：FAR .40、T6 塔樓格 .35、T4 .31、h_p50 .31、塔樓份額 .30；
  −：house 份額 −.29）。右端＝徐家匯 5.1、南京東路 4.4、陸家嘴 3.5；
  左端＝閔行城郊村 −5.2、西岸 −3.5、浦東城郊村 −3.4。
- **PC2 ≈ 空曠尖峰 vs 連續密實毯**（+：T0 空格 .37、h_p90 .32、T8 巨構 .32；
  −：coverage −.34、T5 高覆蓋中層 −.30）。上端＝臨港 4.9、陸家嘴 4.8、外灘 4.0
  （都是「大片空地＋少數高物」）；下端＝老西門 −4.1、楊浦濱江 −2.4、新天地 −2.1
  （連續中低層密毯）。
- 6 個 HELDOUT 星形落點合理：南京東路貼著徐家匯（CBD 區）、三林/涼城/漕河涇/南橋
  落在 T3 板樓毯子帶、外灘上浮到空曠尖峰帶——TRAIN 學的座標系對未見窗口可讀，
  雲是連續的，沒有離散類型的斷崖。

## 5. 誠實侷限

1. **融合覆蓋率在郊區窗口系統性下降，且原因各不相同**：臨港 78.9%（EULUC 在新填海區
   **整層缺失 0%**）；楓涇 85.0% 與南橋/兩個城郊村的 **Function/Age 層 0%**（带年份
   內測版不出市區）；西岸 31.0% 是全庫最低——EULUC 有 parcel（85.5%）但多為
   公園綠地/交通用地，映射不到 8 類，濱江廠房遺構大量 unlabeled。郊區窗口的語義
   標籤更依賴 AOI/POI 單源，可信度低於市區。
2. **POI 稀疏處正如預期**：閔行城郊村 50 m 內有 POI 的建築只 16.8%、浦東城郊村
   36.2%、楓涇 36.6%——但在融合級聯裡 POI 只是補空（pooled 7.8%），所以傷害可控；
   若沿用 v0.2 的 POI 主判，這幾窗會大面積失明。這是回到融合級聯的最強實證理由。
3. **AI 高度層在郊區有可疑值**：楓涇（水鄉）h_p90=75.8 m、臨港 h_p90=90.6 m，
   不排除影像解譯的高度噪聲把「水鄉」推到簽名雲的空曠尖峰帶；未做逐棟驗證。
4. **T8 仍然極薄**：21 個 TRAIN 格／29 棟池，鄰接列只有幾十次計數；k=8 是選定
   不是推定（沿 v0.2 慣例 +2 以容納拆分，未做 silhouette/gap 檢定）。
5. **floors=round(h/3.5) 是粗規則**（probe 的 patch 檔自帶樓層，v0.3 統一重算），
   對裙樓/廠房會高估層數 → GFA/FAR 偏壓；同窗內一致，跨研究比較需注意。
6. Age 覆蓋 0–64.2%（5 個郊區窗全 0）——Agent 2 的年代條件化先驗在郊區窗口
   只能落 no_age 低信心格。
7. 簽名雲的 z-score 與 PCA 對 18 個樣本的 std 仍敏感；PC1/PC2 合計只解釋 57% 變異，
   雲上距離是參考不是度量。重疊規則只防 >30%，A01 與 A07（重疊 23%）並非完全獨立樣本。

## 檔案清單

| 檔案 | 內容 |
|---|---|
| `run_v03.py` | select / extract / library / learn / signatures / fig1 / fig2（全部步驟） |
| `split.json` | 24 窗（id、名、lat/lon、type group、train/heldout）＋選樣規則 |
| `cache/<id>_fused.parquet` ×24 | 逐窗逐棟：semantic, judge, family, height, fp_area, floors, age, … |
| `library.parquet` / `library.fgb` | 合併庫 14,527 棟（bidx 索引供池引用） |
| `tokens.json` / `adjacency.json` | k=8 詞彙（TRAIN-only）＋ 4 鄰鄰接（TRAIN-only） |
| `token_grid_<id>.npy` ×24 | 逐窗 20×20 token 格網（heldout＝最近質心指派） |
| `token_pools.parquet` / `.fgb` | 實例化池（僅 TRAIN 格；格局部座標 dx/dy＋bidx） |
| `signatures.json` | 逐窗 token 組成＋10 指標＋PCA 座標；PCA 載荷與解釋變異 |
| `results_v03.json` | 覆蓋率表、judge 份額、token 統計、PCA 摘要 |
| `fig1_library.png` | 左：24 窗區位（train 圓/heldout 星）；右：簽名雲 PCA |
| `fig2_grammar.png` | 6 個代表 TRAIN 窗 token 格網＋鄰接熱圖＋逐 token 剖面 |
