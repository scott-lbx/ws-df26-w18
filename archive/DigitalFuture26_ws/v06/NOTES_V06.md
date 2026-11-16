# NOTES_V06 — 經典範式為誰造城?(工程筆記)

> 2026-06-12。業主任務:「先用經典的城市規劃理論得到原則和比例(多少住宅、多少辦公,
> 經典範式),生成一個城市,目標著力於 stakeholder 本身,最後看到城市形態長出來的樣子。」
> 交付:`../V06_REPORT.html`。本檔是工程側摘要。

## 設計

鏡像結構:正向(教義 → 配方 → 上海詞彙 → 生成)× 反向(肌理 → 權力讀回 → 「誰的城」),
**宣稱 vs 讀回的落差就是研究對象**。六個實驗:

| 實驗 | 一句話結果 |
|---|---|
| D1 教義蒸餾 | 5 範式 43 個帶出處數字(29 高信度)+ 30 處教義沉默;系統性發現:教義寫死象徵核心、留白經濟殘餘 |
| D2 翻譯層 | 5 份 400 格 token 預算全落教義靶內;18 處表達力失真(cottage 與十字塔說不出來) |
| D3 生成 | 10 畫布(5 單元+5 複合體);語義條件化拼貼 fallback 0% 但近單例供給(全上海只有 2 個 education T2 格元);gallery=業主要的圖 |
| D4 讀回落差 | **4/5 翻轉、全部翻向 developer_capital**;兩種解剖學(no_age 人造物 vs 真實翻譯損失);Perry 守住但站在最弱先驗上 |
| D5 簽名雲 | 投影閘 4.9e-5 通過;光輝城市落雲外、最近親戚=陸家嘴;**曹楊錨 MISS**(今日曹楊更像加密 microrayon——1951 在地化置換了住宅類型學) |
| D6 反轉 | GOV-first 不可達(候選 gov ≤.41,「政府之城」住在年代裡);DEV-first 最易達,贏家=宣稱 gov .95 的光輝城市(反諷) |

## 執行

- 7-agent workflow(D1 含 WebSearch 原典查證)+ 獨立 audit(10 checks:9 confirmed +
  1 圖面缺陷清單)+ 獨立圖面修復 agent + 報告 critic 輪。
- 凍結輸入:v0.3 詞彙/池/簽名/PCA、v0.4 typed voids、v05 先驗與讀回鏈(read-only)。
- 全程 seed 0;重跑:translate → generate_v06 → exp_readback → exp_cloud → exp_invert → audit → make_v06_report。

## 誠實侷限(報告 §10 詳表)

1. a priori 與讀回都是斷言;「翻轉」是兩個斷言體系的失配,不是歷史事實。
2. 翻譯層欠定(同教義可有多組合規預算),未做敏感性掃描——下一步首位。
3. 近單例池供給(education 2 格元、industrial 3 格元)。
4. D4 分解單路徑(b 後 a);L1 分量可符號相消(已並列展示)。
5. 單 seed;GOV-first 裁決差距小於代理誤差(已雙生成裁決)。
6. 教義數字部分僅二手驗證(microrayon 半徑中信度、GB 人均低信度單源)。

## 檔案清單

| 檔案 | 內容 |
|---|---|
| `doctrine_recipes.json` / `doctrine_sources.md` | 5 範式配方(逐數字出處/信度/沉默旗標) |
| `translate.py` / `translation.json` / `skeleton_*.npy` | 翻譯表 + 表達力審計 + 程序化骨架 ×5 |
| `generate_v06.py` / `cache_gen/{unit,city}_*.parquet` | 10 畫布(逐棟語義/家族/token/來源窗) |
| `exp_readback.py` / `results_d4.json` | 讀回 + 落差 + 三源分解 |
| `exp_cloud.py` / `results_d5.json` | 簽名雲投影(正確性閘)+ 曹楊錨 |
| `exp_invert.py` / `results_d6.json` | 15 候選 × 3 目標 + 3 贏家生成 |
| `audit_v06.py` | 獨立復算 |
| `agent_reports.json` | 7 agent 原文(for_report/choices/honest_notes) |
| `fig/` ×19 | 全部圖(含 fig_d3_gallery = 業主指定交付) |
| `report_static.py` + `make_v06_report.py` → `../V06_REPORT.html` | 報告 |
