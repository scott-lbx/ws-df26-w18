# latent_research — 利益相關者 ↔ 城市形態 的潛在研究

> DF26 WS18｜循環城市 AI。把 `05` 的權力算子引擎，變成一條**自主研究管線**：
> 量化 6 種權力體制如何重寫城市形態，並檢驗這個映射能否**雙向反演**。

## 這是什麼

以 12 個上海街區為底，對每個街區施加 6 種權力體制（`05` 的確定性形態算子配方），
用 9 個形態指標量測，得到 **72 樣本 × 9 特徵**的語料，再問四個問題：

| # | 問題 | 結果（`out/latent_research/results.json`） |
|---|------|------|
| Q1 | 形態在低維潛在空間怎麼分佈？ | 權力效應空間 **2 維 ≈ 82%**（PCA）；AE 9→2 重建 MSE **0.0017**（效應流形近乎 2D） |
| Q2 | 每種權力把形態往哪推？ | developer_led→**slender +418%**、state_led→**concentration +121%**、resident_self_build/shared→**coverage ↓**（`regime_effects.csv`） |
| Q3 | 只看形態能反推權力嗎？ | 跨城 LOSO 分類 **97.2%**（chance 16.7%），置換檢定 **p=0.003** |
| Q4 | 映射可逆嗎？ | KMeans ARI **0.72**、最近質心反演 **94.4%** → 教學尺度上**近可逆** |

> ⚠ 誠實：6 體制是**確定性算子**，form 天生編碼 regime。Q3 高分量的是形態指紋**多可分離、能否跨城通用**，
> **非**真實世界因果推論。詳見 `REPORT.md` 限制節。

## 檔案

| 檔 | 作用 |
|----|------|
| `extract_features.py` | 掛 05 引擎 → `out/latent_research/features.csv`（72×9） |
| `analyze_latent.py` | PCA/AE/VAE + LOSO 分類 + Cohen's d + 置換檢定 → `results.json` / `regime_effects.csv` / `latent_coords.csv` / `figures/*.png` |
| `robustness.py` | 不確定性量化 → `robustness.json`:城市層 bootstrap 95% CI、多 seed、逐體制離散度、參數洩漏 |
| `workflow_latent_research.mjs` | 12 階段 Claude Workflow（ground→audit→interpret→adversarial→synthesize→critic） |
| `WORKFLOW_PLAN.md` | 規格 · 分工圖說 · agent 規模表 · 護欄 |
| `REPORT.md` | Workflow 綜合出的研究報告（含護欄、對抗判決 audit trail、穩健性補遺） |

## 快速跑

```bash
.venv\Scripts\activate
set PYTHONUTF8=1
python latent_research/extract_features.py     # 語料
python latent_research/analyze_latent.py       # 分析 + 圖
python latent_research/robustness.py           # 不確定性量化(CI / 多 seed / 洩漏)
# 推理層(Claude Code 內):Workflow({ scriptPath: "latent_research/workflow_latent_research.mjs" })
```
