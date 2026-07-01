# 潛在研究 Workflow — 規格與分工（DF26 WS18）

> **循環城市 AI｜利益相關者 ↔ 城市形態 的雙向映射**
> 台灣陽明交通大學 · DigitalFUTURES 2026 Workshop 18

把 `05` 的權力算子引擎，從「產生一張渲染圖」升級成一條**可自主執行的研究管線**：
量化 6 種權力體制如何重寫城市形態，並檢驗這個映射能否**雙向反演**（form → power）。

---

## 1. 研究問題

| # | 問題 | 方向 | 主要輸出 |
|---|------|------|---------|
| **Q1** | 72 種形態在低維潛在空間怎麼分佈？ | 表徵 | PCA / AE / VAE 三種壓縮、silhouette |
| **Q2** | 每種權力把形態往哪個方向推？ | **power → form** | `regime_effects.csv`、效應熱圖、Cohen's d |
| **Q3** | 只看形態，能不能反推是哪種權力造的？ | **form → power** | 跨城 LOSO 分類、混淆矩陣、置換檢定 |
| **Q4** | 這個映射雙向一致 / 近可逆嗎？ | 綜合 | KMeans ARI、最近質心反演準確率 |

---

## 2. 兩層架構

```
┌─────────────────────────────────────────────────────────────┐
│ 資料層(Python，硬數字，零 AI 推斷)                          │
│   extract_features.py  ── 掛 05 引擎 ──▶ out/.../features.csv  │
│        12 城 × 6 體制 = 72 樣本 × 9 形態特徵                    │
│   analyze_latent.py    ── PCA/AE/VAE + LOSO + 效應量 + 置換 ──▶ │
│        results.json · regime_effects.csv · latent_coords.csv  │
│        figures/*.png                                          │
├─────────────────────────────────────────────────────────────┤
│ 推理層(Workflow，Claude 子代理,把數字放回程式碼脈絡解讀)    │
│   workflow_latent_research.mjs — 12 階段                       │
│        ground → audit → interpret → adversarial → synthesize  │
│        → critic  ──▶ REPORT.md                                │
└─────────────────────────────────────────────────────────────┘
```

**設計原則**：硬數字由 Python 決定性算出、可重算；Claude 子代理**只做解讀、驗證、綜合**，不得憑空產生數據。每個主張都要能回溯到 `features.csv` 的列或引擎程式碼的行。

---

## 3. 兩個座標系(方法核心)

| 座標系 | 定義 | 用途 | 現象 |
|--------|------|------|------|
| **形態空間** | 9 特徵 log1p + 全域 z-score | 描述「72 種形態座落何處」 | 站點尺度(棟數/面積)主導 → 城市各自成群 |
| **權力效應空間** | 每特徵對「該站現狀」取 log 比值 | **主分析**：抵消站點尺度，只剩「權力做了什麼」 | 現狀=原點；每種權力=一個位移向量 |

> 若權力在**效應空間**跨城成群，就證明「權力留下**站點無關**的形態指紋」——這是 Q3/Q4 的關鍵。

---

## 4. Workflow 12 階段分工

| 階段 | 名稱 | agent 數 | 角色 | 產出 |
|------|------|:---:|------|------|
| 1 | **Ground** | 2(並行) | 讀 `operators.py` / `measure.py` / `regimes.yaml`，建立 6體制×9特徵權威定義 | `ENGINE_SPEC` |
| 2 | **Audit** | 1 | `features.csv` 與引擎定義對帳(72×9、平衡、跨體制 sanity) | `AUDIT` |
| 3 | **Interpret** | 4(並行) | Q1–Q4 各一 agent，引用實際輸出檔解讀 | 4× `INTERPRETATION` |
| 4 | **Adversarial** | 3(並行) | 敵意審稿：攻 **洩漏/循環論證**、**小樣本外推**、**效應穩健性** | 3× `VERDICT` |
| 5–6 | **Reconcile + Synthesize** | 1(high effort) | 把對抗判決折進主張，寫 `REPORT.md` | `REPORT` |
| 7 | **Critic** | 1 | 完備性評審：未跑方法 / 未驗主張 / overclaim | `CRITIQUE` |

**agent 規模**：單輪 ≈ **12 個子代理**（2+1+4+3+1+1）。並行上限自動夾在 `min(16, cores−2)`。
高成本階段只有 Synthesize（`effort: high`）；grounding 與 audit 走一般 effort。

**護欄折返**：Adversarial 判決 `refuted=true` 或 `severity=fatal` 的主張，Synthesize 階段**必須改寫或移除**——寧可少宣稱不誇大。

---

## 5. 交付物清單

| 檔案 | 內容 |
|------|------|
| `latent_research/extract_features.py` | 語料生成器（掛 05 引擎，12×6→72 樣本） |
| `latent_research/analyze_latent.py` | 潛在分析（PCA/AE/VAE + LOSO + 效應量 + 置換檢定 + 8 圖） |
| `latent_research/robustness.py` | 穩健性補遺（城市層 bootstrap CI + leave-one-recipe + 多 seed + 逐體制離散度 + 參數洩漏量化） |
| `latent_research/workflow_latent_research.mjs` | 12 階段 Workflow 腳本（`node` 語法驗證通過） |
| `latent_research/WORKFLOW_PLAN.md` | 本檔 |
| `latent_research/REPORT.md` | Workflow 綜合出的研究報告（含護欄、圖對照、對抗判決 audit trail、穩健性補遺） |
| `out/latent_research/robustness.json` | 不確定性量化（95% CI、多 seed、洩漏數字） |
| `out/latent_research/features.csv` | **72 樣本 × 9 特徵**研究語料 |
| `out/latent_research/results.json` | 全部量化指標 |
| `out/latent_research/regime_effects.csv` | power→form 效應表（log 比值 + Cohen's d） |
| `out/latent_research/latent_coords.csv` | 各樣本在 PCA/AE/VAE 的潛在座標 |
| `out/latent_research/figures/*.png` | 8 張圖 |

---

## 6. 護欄（Guardrails）

- **零 AI 推斷寫進資料**：`features.csv` 全由 05 決定性算子 + `measure.diagnose` 算出。
- **可重算**：`seed=20260702` 固定 numpy / torch；`concentration` 重心參照固定為該站現狀。
- **對抗預設可疑**：skeptic 預設主張「可疑直到證明」，證據不足即降級。
- **循環性誠實**：6 體制是**確定性算子配方**，form 天生編碼 regime。故 Q3 的高分量的是「形態指紋**有多可分離、能否跨城通用**」，**不是**真實世界因果推論。此點必須寫進 REPORT 的限制節。
- **樣本尺度誠實**：僅上海 12 城、72 樣本；LOSO 仍在同一城市文化母體內。

---

## 7. 怎麼跑

```bash
# 0) 環境(專案根)
.venv\Scripts\activate

# 1) 資料層 —— 產生語料 + 分析(約數十秒)
python latent_research/extract_features.py      # → out/latent_research/features.csv
python latent_research/analyze_latent.py        # → results.json / 圖 / 效應表 / 置換檢定
python latent_research/robustness.py            # → robustness.json / 圖（不確定性量化)

# 2) 推理層 —— 在 Claude Code 內執行 Workflow(需 ultracode / 明確要求編排)
#    Workflow({ scriptPath: "latent_research/workflow_latent_research.mjs" })
#    回傳 { report_markdown, ... } → 寫成 latent_research/REPORT.md
```

> Windows 主控台若打印中文報 `cp950`，兩支 py 已在開頭 `sys.stdout.reconfigure(encoding="utf-8")`；
> 或跑前設 `set PYTHONUTF8=1`。

---

## 8. 資料血緣

```
05/data/<slug>/buildings.parquet   (AI 實測高度 + EULUC 用途 + 帶年份 Function + 百度 AOI)
   └─ common.assign_all           一栋一個 stakeholder(級聯查表,非產權考證)
   └─ common.to_recs              → recs [{geom,h,sh,frozen}]
        └─ operators.apply_regime  regimes.yaml 的算子配方(power → form)
             └─ measure.diagnose    9 形態特徵
                  └─ features.csv    72 樣本
```

12 城：`caoyang · dapuqiao · kongjiang · laoximen · lujiazui · nanjingxi · pengpu · tianshan · waitan · xujiahui · yuyuan · 七宝镇`
6 體制：`current · developer_led · state_led · resident_self_build · developer_scale_graded · shared`
