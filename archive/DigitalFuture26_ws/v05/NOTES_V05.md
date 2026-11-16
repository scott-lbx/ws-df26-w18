# NOTES_V05 — 分類、權力與城市形態:從頭設計的研究輪

> 2026-06-12。業主指示:全新 version,吸收前面但以從頭開始的研究精神,輸出研究生等級的
> 研究報告 HTML(資料處理、實驗、結果、diagram、選擇與理由全披露;失敗也是結果)。
> 研究任務一句話:「我們有一些城市數據,怎麼去找到分類、權力與城市形態的關係的探索。」
>
> 交付:`../V05_REPORT.html`(自含,12 圖 12 節)。本檔是工程側筆記。

## 結構

問題操作化為「分類—權力—形態」三角,四個實驗各佔一條邊,全部站在凍結輸入上
(v0.3 融合表+簽名+token 詞彙、v0.4 遮罩+typed 格網、探針 v0.1 先驗,一律 read-only):

| 實驗 | 邊 | 一句話結果 |
|---|---|---|
| E1 辨識 | 形→類 | 留窗轉移下塌回基準附近(0.641 vs 0.601);權力類 recall ≤1.2%、healthcare=0;C 報告 72% 診斷為洩漏 |
| E2 權力鏈 | 類→權 | 24 窗跑通;橋尺度穩定(lowrise→house 0.999);警報:4 個零年代窗剖面=先驗回聲、外灘 gov 最低=先驗缺遺產通道 |
| E3 三角斜邊 | 形↔權 | 只有 dev 穩健可回收(LOO R² 0.46);訊號在 PC2 不在 PC1;TYPE 不是天花板(年代才是缺塊)——三角經由時間閉合 |
| E4 反事實 | 權→形 | 3/6 方向成立(垂直度、house 性傳得動);GOV 要 47.8% 公共語義渲染剩 7.5%(語義在 token 翻譯中剝離)|

## 執行方式

- Multi-agent workflow(7 agents):Dataset → {E1, E2, diagrams} 並行 → {E3, E4} 並行 → 獨立 audit。
- Audit agent 以獨立程式碼復算 6 項(A13 剖面 4 維 4 位小數吻合、E1 balanced acc ±0.1pp、
  橋格 1355/1357 一致、E3 dev R² ±0.007、E4 FAR 幾何重算一致、全 results 內部一致性)——
  **6/6 confirmed**。圖面缺陷 5 處由獨立修復 agent 處理。
- 報告:`report_static.py`(固定章節)+ `make_v05_report.py`(結果章節,數字直讀 results JSON)。

## 重跑

```
python3 build_dataset.py && python3 exp1_recognition.py && python3 exp2_power.py
python3 exp3_triangle.py && python3 exp4_counterfactual.py && python3 make_diagrams.py
python3 audit_v05.py && python3 make_v05_report.py
```
全部 seed 0;python3 = 3.9(geopandas/sklearn/scipy/pyarrow);不碰 GPKG。

## 誠實侷限(報告 §9 的工程版摘要)

1. 權力剖面是文獻先驗的斷言不是觀測;E3 測的是斷言的內部可回收性。
2. n=24 視窗;單一城市;Age 39.2% 覆蓋且 1984/85 截斷(曹楊工人新村僅 6% pre-1985 GFA)。
3. 標籤循環的邊界:E1 排除了標籤來源特徵,但 parcel 級標籤粗糙度仍在(A32 案例)。
4. E4:NNLS 欠定(T2/T4 永不用)、重縮放扭曲 FAR(GOV ×1.22)、單 seed。
5. 老天花板未動:池稀釋(v0.3)、度量盲區(v0.4)、latent 抽卡(parked)。

## 檔案清單

| 檔案 | 內容 |
|---|---|
| `build_dataset.py` / `dataset.parquet` / `windows.json` | 14,527 棟 × 24 窗 + 新脈絡特徵 |
| `exp1_recognition.py` / `results_e1.json` | E1 雙 CV × 雙特徵組 |
| `exp2_power.py` / `results_e2.json` / `power_priors_v05.json` | E2 剖面 + 橋 |
| `exp3_triangle.py` / `results_e3.json` | E3 篩查 + LOO + 敏感性 |
| `exp4_counterfactual.py` / `results_e4.json` / `cache_gen/cf_*.parquet` | E4 三配置畫布 |
| `make_diagrams.py` | fig_d1/d2/d3 方法圖 |
| `audit_v05.py` | 獨立復算(6/6 confirmed) |
| `agent_reports.json` | 七個 agent 的 for_report/choices/honest_notes 原文 |
| `fig/` ×12 | 全部圖 |
| `report_static.py` + `make_v05_report.py` → `../V05_REPORT.html` | 報告 |
