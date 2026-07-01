# REPORT.md — 權力↔形態雙向映射:一次「設計文法」的內部效度稽核

> 工作坊主題:利益相關者(權力)↔城市形態的雙向映射。
> 語料:上海 12 站 × 6 體制 = 72 樣本,單一 seed 20260702(`out/latent_research/results.json:2-4,16`)。
> **本報告最重要的方法學前提**:6 個「權力體制」不是被觀測的真實政治經濟安排,而是 `regimes.yaml` 手寫、參數固定、城市無關的**確定性算子配方**,逐站套用於各站 `current` 後由 `measure.py` 量測而成。因此本報告一律以「文法自我分離度 / 內部建構效度」而非「經驗發現 / 因果可逆」的層級來陳述結果。凡涉及「真實城市可由形態反推真實權力」的斷言,均已依對抗性判決(三項 fatal / refuted=true)移除或降級。

---

## 1. 摘要:可辯護的頭條發現

以下每條均已扣除被 skeptic 駁倒的部分,只保留在「合成語料 + 確定性算子」前提下站得住的斷言。

1. **這 9 個形態特徵構成的量測詞彙,足以把 6 種「權力=算子配方」的風格化變換彼此乾淨地分離開。** 在權力效應空間(對各站 `current` 取 log 比值)中,PCA/AE/VAE 三法輪廓係數一致落在 0.60–0.76(去除退化參照 `current` 後升到 0.69–0.76;`results.json:41-53`)。這是量測器敏感度 + 設計語言表達力的展示,**不是**關於真實城市的經驗發現。

2. **每個體制的最強形態指紋都能逐條回溯到它被建構要產生的那個算子——即「配方被重述」。** developer_led→slender +418%(d=5.92)、state_led→concentration +121%(d=5.6)、resident_self_build→coverage −25.6%(d=−5.84)、shared→coverage −34.2%(d=−5.55)、developer_scale_graded→coverage +19.3%(d=1.81)(`results.json:56-81`)。這些巨大的 d 是**確定性算子的簽名**(配對 std→0 使 d 爆炸),不可當作「異常強的真實效應」引用。

3. **形態→體制的可分離性在跨站上近乎恆定,但這是 log-ratio 特徵工程 + 城市無關算子的解析結果,不是跨脈絡泛化。** LOSO 準確率 0.972(`results.json:84`)測的是「同一組固定常數被重跑 12 次」的可重現性;它**不能**宣稱真實權力可從真實城市形態反推。

4. **唯一誠實的「模糊帶」暴露了真實的 equifinality:developer_scale_graded 是唯一 recall<1(0.833)的體制。** 它效應最弱(coverage d=1.81)、貼近 `current`/原點,12 例中 2 例被誤判為 current(`results.json:91`、`figures/confusion_loso.png`)。它跨站變異明顯較大(scale_graded 的漸變衰減使效應真的依站點而變),反而是全語料中**最接近「有真實建模自由度」**的一條。

5. **可逆性讀數(僅在合成語料尺度內):有標籤引導 ≈0.94–0.97,純無監督 ≈0.72。** nearest_centroid=0.944(in-sample,偏樂觀)、KMeans ARI=0.721(`results.json:102-104`)。此讀數量的是「文法把自己的體制分得多開」,不是真實 power↔form 的因果可逆。

---

## 2. 方法:兩座標系與為何以效應空間為主分析

### 2.1 兩座標系

- **形態空間(絕對)**:直接用 9 個特徵(n, far, coverage, h_mean, h_max, h_cv, grain, slender, concentration;`results.json:5-15`)的原始量值。此空間被城市尺度主導——PCA 前 2 主成分僅解釋 68.2%(0.4015+0.2801;`results.json:17-19`),散佈由城市各自分開(`figures/pca_form_space.png`)。

- **權力效應空間(log 比值)**:對每個特徵取 `log(regime_feat / current_feat)`,抵消各站尺度基線。前 2 主成分解釋 82.3%(0.4959+0.3266),前 3 達 95.7%(`results.json:28-31`)。

![形態空間 PCA:城市依尺度分開](out/latent_research/figures/pca_form_space.png)

![權力效應空間 PCA:體制跨城成群](out/latent_research/figures/pca_power_effect.png)

![累積解釋變異](out/latent_research/figures/explained_variance.png)

### 2.2 為何以效應空間為主分析——以及它「保證了什麼」

取 log 比值後,`current` 在每座城市都變成**精確零向量**(`latent_coords.csv` 12 站 current 完全重合),體制施加的「相對轉變量」被獨立出來,讓同一體制跨城對齊成群。

**但必須誠實標記循環性**:既然每個 regime = 城市無關的確定性算子作用於 current,log 比值 ≈ 算子的固定乘性效應 + 只由二階幾何差異決定的小殘差。因此「site-invariance(跨站不變)」在此**是被 normalization 解析上強制的,不是數據湧現的**。`analyze_latent.py` 自陳「若權力在此空間跨城成群,就證明權力留下站點無關指紋」——這是循環論證:先抽掉站點尺度,再宣告發現尺度無關。**主分析採效應空間是正確的方法選擇,但它保證的是「文法內部自洽性」,不是「經驗不變性」。**

---

## 3. 四問逐節

### Q1 — 潛在空間:6 體制是否跨城成群?三種壓縮是否一致?

**可辯護的讀法**:在效應空間中,六體制明確成群,且三種壓縮法一致。輪廓係數(全體 / 去 current):PCA 0.597 / 0.689、AE 0.641 / 0.763、VAE 0.615 / 0.700(`results.json:41-53`)。AE 重建近乎完美(MSE 0.00172),VAE 因 KL 正則化重建較差(MSE 0.2274,約高 132 倍;`results.json:39-40`)卻仍給出一致的高輪廓——結論對壓縮方法穩健。

![PCA/AE/VAE 三面板同色點跨城成團](out/latent_research/figures/latent_methods_compare.png)

![VAE 權力效應空間](out/latent_research/figures/vae_power_effect.png)

**為何用效應空間**:形態空間前 3 主成分才 82.8%,且被城市尺度主導;效應空間前 2 已 82.3%,體制簽名被獨立出來(§2.2)。

**降級與限制(對抗判決折入)**:
- 「成群」不是經驗湧現,而是「對每站套同一函數 + log-ratio」兩段式寫死的結果。輪廓/ARI 量的是「文法把自己分得多開」。
- `current` 為退化參照點(零向量),含 current 的較低輪廓值應以 excluding_current 版本為準。
- developer_scale_graded 簽名最弱、與 current 幾乎重疊(見 Q3),是唯一召回率<1 的體制。
- 樣本 n=72,但**有效獨立單位其實是「配方數=6」**(每站真正獨立的觀測只有 current,其餘 5 個是其確定式變換);n=72 嚴重高估證據量。
- 信心:對「三法一致、效應空間可分離」為 high;對「這代表真實跨城不變性」為**不成立(refuted)**。

---

### Q2 — power→form:每個非 current 體制的最強指紋 + 算子解釋

下表每一條都能逐項回溯到 `regimes.yaml` 配方(`operators.py` 實作)。**關鍵誠實標記:這些是算子定義的恆等結果(tautology by construction),不是「權力如何塑造形態」的經驗發現。**

![regime × feature 平均 log 比值熱圖](out/latent_research/figures/regime_effects_heatmap.png)

| 體制 | 最強指紋 | Cohen's d | %變化 | 為何(算子機制) |
|---|---|---|---|---|
| developer_led | slender | +5.92 | +418.2% | slim(footprint×0.45, h÷0.45) + split_to_towers(area/3) + densify(h×1.8):分子 h 暴漲、分母 √footprint 下降,slender 機械地翻 ~5 倍。**YAML feature 欄已預先寫死「瘦長比明顯升高」。** |
| state_led | concentration | +5.6 | +120.6% | concentrate(center=state_centroid, reach_frac=0.18)把守恒 GFA 往 state 質心 λ 半徑內收攏,**而 concentration 指標量的正是同一個質心、同一個 0.18 內的 GFA 佔比(`measure.py`)——算子與指標共用參照,升高是最緊的恆等式。** |
| resident_self_build | coverage | −5.84 | −25.6% | infill(cell 120m², buffer −step×0.06 留巷 + 丟棄邊界碎格)使小單元無法鋪滿母 footprint。**注意:此符號與 regimes.yaml 自陳的「高覆蓋」相反,且與真實非正式聚落(通常高覆蓋)相反——它是 infill 網格化的數值人工產物。** |
| developer_scale_graded | coverage | +1.81 | +19.3% | 單一算子 scale_graded(developer footprint 近中心×2、向外衰減到×1)。**全語料最弱指紋,d 遠低於其餘 ~5–6,跨站變異明顯較大——是唯一有真實建模自由度的一條。** |
| shared | coverage | −5.55 | −34.2% | open_ground(footprint×0.6 釋放共享地, h÷0.6 補償) + level(h→median)。**signature 近乎平手:coverage d=−5.55 僅險勝 h_cv d=−5.48;regimes.yaml 標的頭條特徵其實是「h_cv 最低(均質)」,不宜只報 coverage。** |

**對抗判決折入(fatal)**:
- 「Cohen's d」在此為配對效應量 mean(Δ)/std(Δ) 跨 12 城,量的是「方向變化的跨城一致性」,非教科書兩組 d。d≈6 不是「異常強真實效應」,而是**確定性算子對每站施加近乎相同相對位移使 std→0** 的退化 DGP 警訊。
- 多個特徵在 12 站上 log 比值與 d **精確為 0.000000**(state_led 的 n/coverage/grain、developer_scale_graded 的 h_mean/h_max/h_cv、shared 的 n)——這是「該算子不觸碰該自由度」的位元級恆等,不是經驗規律。
- h_max 在多站被 cap 釘死(developer_led=480.0=densify cap_m;state_led=600.0=concentrate cap_m)——特徵報的是參數而非形態。
- 因此:developer_led→slender、state_led→concentration、shared→coverage↓、resident→grain↓/n↑ 皆為**引擎必然,不可當發現引用**。較可辯護者僅 developer_scale_graded→coverage(有真實站點依賴),以及各體制「共變模式/相對排序」作為設計語彙提案。

---

### Q3 — form→power:LOSO 反推準確率、誰最易被混淆

**讀數**:LOSO 準確率 0.972,相對 chance 0.167 約 5.8 倍;72 樣本僅 2 個錯分(`results.json:84-85`)。置換檢定 observed 0.9722、null_mean 0.1245、null_p95 0.2222、p=0.0033、n_perm=300(`results.json:94-100`)。旁證 nearest_centroid 0.944、KMeans ARI 0.721(`results.json:102-104`)。

**唯一系統性失誤**:developer_scale_graded 是唯一 recall<1(0.833)者;12 例中 2 例被誤判為 current,其餘五體制 recall 全為 1.0(`results.json:86-92`)。混淆單向不對稱——current 從未被錯分為 developer_scale_graded。機制:developer_scale_graded 是相對 current 最輕微的擾動(coverage d=1.81),且多個特徵(h_mean/h_max/h_cv)與 current 完全相同(`features.csv`)。

![LOSO 混淆矩陣](out/latent_research/figures/confusion_loso.png)

![置換檢定虛無分布](out/latent_research/figures/permutation_test.png)

**對抗判決折入(fatal,此問受創最重)**:
- **0.972 不是關於城市的經驗/因果發現。** 分類器只是在辨認「施加了哪個確定性算子」。current 的 recall=1.0 是**算術保證**(它在每城都是精確零向量,佔 1/6 頭條準確率)。
- LOSO「留一城卻從不留一機制」——同一組固定常數(ratio=0.45、far_gain=1.8、factor=2.0)施於全部 12 城,故 97% 近乎恆真。它測的是決定式函數的可重現性,不是跨脈絡遷移。
- 置換檢定 p=0.003 只排除「標籤隨機」,對確定性生成器的洩漏完全無感。
- **誠實的殘餘主張**:9 個形態特徵的分辨率足以把 6 種確定性配方在特徵空間分開,唯一相撞處(developer_scale_graded↔current,recall 0.833)恰恰暴露了真實的 equifinality。這是量測器敏感度的檢核,不支持「form→power 近可逆」的經驗斷言。

---

### Q4 — 可逆性:此教學尺度上「雙向一致」到什麼程度?

**綜合讀數(僅在合成語料內有效)**:正向(power→form)為確定性算子,效應向量巨大且各異;反向(form→power)有標籤引導 ≈0.94–0.97(nearest_centroid 0.944、LOSO 0.972),純無監督 ≈0.72(KMeans ARI 0.721)(`results.json:84,102-104`)。輪廓 0.60–0.76 證實各體制在 2D 效應空間成群。唯一穩定漏點是效應最弱、貼近原點的 developer_scale_graded。

**對抗判決折入(fatal,最大限制)**:
- **循環性/由建構而來**:6 體制是確定性算子配方,form 天生編碼 regime。Q4 量的是形態指紋的「可分離性與跨城通用性」,**不是**真實 power↔form 的因果可逆。任何超出合成語料的「真雙向一致」外推須降級移除。
- nearest_centroid=0.944 為 in-sample(質心含被分類點本身),偏樂觀;較誠實的樣本外數字應以 LOSO 0.972 為準——但後者同樣受循環性限制。
- 兩個指標僅在 2D 效應 PCA 空間計算(捨棄 ~18% 變異);KMeans ARI 僅單一 random_state,為點估計無區間。
- **可辯護的可逆性讀數**:這套「power→form」設計文法是**可注入/可讀的**——不同確定性配方在 9 維形態空間基本不相撞。定位為「文法自我分離度」,而非「權力被發現」。

---

## 4. 誠實與限制

1. **確定性算子 → 分數上限的循環性(最致命)**:每一列 = `diagnose(apply_regime(current, recipe))`,無噪聲、城市無關、純確定性。每條 recipe 的算子恰好直接寫入隨後被分類器讀回、並當作「該體制指紋」的那個特徵(最緊的例子:state_led 的 concentrate 與 concentration 指標共用同一 state 質心 + 同一 reach_frac=0.18)。所有高分(d≈6、LOSO 0.97、silhouette 0.7)是**設計語言被自己的量測器重述**,不是經驗規律被發現。

2. **僅上海 12 城、同一文化母體**:LOSO 仍在同一城市母體內,無跨城市/年代外部驗證。連 12 點都不是 12 次獨立體制實現——有效獨立單位是「配方數=6」。

3. **72 樣本、單一 seed 20260702、無置信區間**:VAE 於 2D latent 資料量偏少(MSE 0.2274),絕對重建品質需保留。

4. **log-ratio 保證了什麼**:它把 current 變成零向量、抽掉站點尺度,使「跨城成群」在解析上被強制。它保證的是內部自洽,不是經驗不變性。

5. **哪些純屬引擎機制必然(不可當發現引用)**:developer_led→slender↑、state_led→concentration↑、shared→coverage↓/h_cv↓、resident→grain↓/n↑,以及所有精確為 0 的特徵(算子不觸碰該自由度)、被 cap 釘死的 h_max。**唯一較有真實建模自由度**:developer_scale_graded→coverage(d=1.81,站點依賴)。

6. **符號可能是錯的**:resident_self_build 的 coverage↓ 與 regimes.yaml 自陳「高覆蓋」及真實非正式聚落相反,是 infill 侵蝕緩衝的數值人工產物。

7. **原始碼自我聲明**:`regimes.yaml` 明載這些 regime→形態是「教學假設 / 可爭論的語言,不是經驗斷言或真實規劃預測」。本報告與此聲明一致。

---

## 5. 對工作坊主題「利益相關者↔城市形態雙向映射」的意義與延伸

**正確定位**:這是一套**可重現的生成式設計文法(generative design grammar)**,而非關於「真實政治經濟→城市形態」的外部效度發現。它證明的是:

- **(a) 可注入性**:把「利益相關者權力配置」操作化為一組具名、確定性的形態算子,能在異質的真實街道初始形態上產生語意連貫的形態論述束(如 developer_led = far↑ + slender↑ + grain↓,一個可討論的「塔樓化高密度」語言)。
- **(b) 可讀性**:這些配方在 9 維形態空間分離得夠乾淨,量測詞彙有足夠分辨率把它們讀回來。
- **(c) 內部建構效度**:結果是**被「展示」而非被「發現」**;silhouette/LOSO/ARI 量的是文法自我分離度。

**對工作坊的價值**:這是一個讓學員操作「權力假設 ↔ 形態後果」的**設計探索與教學沙盤**——可注入不同權力敘事、即時讀出其形態簽名、討論其空間政治涵義。它的力量在教學與設計思辨,不在經驗預測。

**要把主張升級為經驗發現,至少需要**:
1. 體制標籤來自**獨立觀測**(如真實開發審批、地權、規劃制度資料),而非合成算子;
2. 跨越**非上海、非同一文化母體**的城市與年代;
3. 算子參數**不被人為固定為站點無關**(引入站點敏感的、可校準的映射);
4. 引入**噪聲與真實 equifinality**,並報告置信區間、多 seed;
5. 用真實 form→真實 power 的獨立標註做外部驗證,而非在自生成語料上做 LOSO。

在具備上述之前,所有 d 與分類準確率應標示為「文法自我分離度」,並附「這是被建構、非被觀測」的免責聲明。

---

## 6. 穩健性補遺 — 關閉完備性評審的 top gap(不確定性量化)

評審指出:主分析自承「有效獨立單位=6 配方」(非 72 樣本),卻把 LOSO 0.972、Cohen's d、silhouette、ARI、反演準確率全以**乾淨點估計**呈現,無任何區間。本節以誠實的重抽單位(城市 / 配方)補上(`out/latent_research/robustness.json`、`figures/robustness.png`;seed 固定可重現)。

![穩健性:城市層 bootstrap CI 與逐體制離散度](out/latent_research/figures/robustness.png)

**6.1 城市層 bootstrap 95% CI(B=600,重抽單位=城市而非 72 偽樣本)**

| 指標 | 95% CI | 中位 |
|---|---|---|
| silhouette(去 current) | [0.64, 0.79] | 0.71 |
| KMeans ARI | [0.61, 0.77] | 0.72 |
| 最近質心反演準確率 | [0.90, 0.99] | 0.94 |

→ 即使以「城市」為重抽單位、把有效樣本壓到最誠實的尺度,分離度仍**穩定遠離退化**(下界 silhouette 0.64、ARI 0.61);點估計不是單一幸運分割的產物。但這仍是**文法自我分離度**的區間,不改變 §3–§4 的循環性定性。

**6.2 LOSO 逐城 fold 分佈**:12 個留一城 fold 準確率 mean 0.972、min 0.833,僅 **2/12 城**出現誤分(皆為 developer_scale_graded→current)。分類近乎恆定 = 決定式函數可重現,非跨脈絡遷移(見 §3 fatal 判決)。

**6.3 leave-one-recipe 穩定度**:逐一移除 5 個非 current 體制後,剩餘體制的 silhouette 落在 [0.73, 0.79](Δ=0.06)→ **無單一配方主導**分離結構。

**6.4 多 seed 潛在穩健度**(5 seeds):AE silhouette **0.74±0.03**、VAE **0.69±0.01** → 潛在分離對初始化穩健,不是單次訓練假象。

**6.5 逐體制效應空間離散度**(修正正文 claim 4):
`current`=0.00(零向量,建構使然)、`developer_scale_graded`=**0.10**、`shared`=0.15、`developer_led`=0.20、`state_led`=0.30、`resident_self_build`=0.46。

> **⚠ 正文 claim 4 更正**:正文稱 developer_scale_graded「跨站變異明顯較大 / 最接近有真實建模自由度」——經此補遺實測,其離散度 0.10 為**次低**(僅高於零向量的 current)。故它與 current 相混的真正原因是**貼近 current(效應最弱)**,而非跨站變異大;且 `operators.py` 全管線零隨機(`infill` 的 `vary` 為固定索引雜湊),不存在「真實建模自由度」。**claim 4 的「變異較大 / 真實建模自由度」措辭據此撤回**,保留者僅「它是唯一 recall<1、效應最弱、暴露 equifinality 的模糊帶」。

**6.6 參數洩漏量化**(把 §4 的循環性從定性升為定量):非 current 的 log 比值格中,**97/540(18%)精確為 0**(該算子不觸碰該自由度);**10/72 列**的 `h_max` 釘死在算子 cap(densify 480 / concentrate 600 / open_ground 200)——這些格子的「特徵」實為在**讀算子參數**,而非量測形態。此為 §4「確定性洩漏」最直接的數字證據。

**補遺結論**:不確定性量化**不改變**本研究的定性(仍是設計文法之內部效度,非經驗發現),但把每一條頭條數字從點估計升級為**有區間、對重抽/多 seed/移除配方穩健**的陳述,並更正了 claim 4 的一處措辭。這關閉評審標記的 top gap。

---

## 附錄 A — 對抗性驗證判決(audit trail)
三位敵意審稿人各攻一個要害,判決已折入正文。摘要:

**A1. severity=fatal · refuted=True**
- 攻擊主張:只凭 9 个形态特徵,就能以 ~97% 跨城准确率反推出是哪种权力体制(form→power 近可逆),且这是一个关于城市的经验/因果发现。
- 最強反駁:致命的是资料生成与量测的循环:每个「权力」标籤不是被观测到的,而是 pipeline 自己贴的 —— 每一列 = diagnose(apply_regime(current, recipe))(operators.py + measure.py + regimes.yaml),无噪声、城市无关、纯确定性。而每条 recipe 的算子恰好直接写入分类器随后读回、并当作「该体制指纹」的那一个特徵:developer_led 的 slim(ratio=0.45) 直接抬 slender(d=5.92,+418%);state_led 的 concentrate(center=state_centroid, reach_frac=0.18)直接抬 concentration,而 measure.ref_center 正是用同一个 state 质心 + 同一个 reach_frac=0.18 去积分(d=5.6,+120.6%,同心同半径,最紧的循环);infill / open_ground(ratio=0.6) 直接砍 coverage(d=-5.84 / -5.55)。所以 form→power「反推」只是在反演一张手写、无噪声、城市无关的查表,不是发现经验规律。两个灌水放大器:(1) 分类跑在 power_effect_space,每列 = log(feat/current_feat),因此 current 在每座城市都是精确零向量 → 它 recall=1.0 是算术保证(占 1/6 头条准确率),而逐城取比值本身就人为制造了「跨城不变」——任何城市无关的确定性算子在此空间天生尺度不变;(2) 置换检定 p=.003 只排除「标籤随机」,对确定性生成器的洩漏完全无感;LOSO 留一城却从不留一机制(同一组固定常数 ratio=0.45、far_gain=1.8、factor=2.0 施于全部 12 城),故 97% 近乎恒真。它「不能」宣称真实、被观测到的权力安排可从真实城市形态反推,也不能宣称任何因果或经验意义上的可逆。它「只能」说这套确定性配方被自己的量测器重述出来了。
- 誠實殘餘:诚实版:这 9 个形态特徵组成的量测词彙,分辨率足以把 6 种「权力=算子配方」的风格化变换彼此区分开,且这些指纹内部一致、可用城市尺度无关的 log 比值表达——即这套「power→form」设计语言是可注入/可读的:不同的确定性配方在特徵空间基本不相撞(诚实的例外是 developer_scale_graded,recall 仅 0.833,与 developer_led/current 部分重叠,恰恰暴露了真实的 equifinality)。因此这项结果验证的是量测器的敏感度 + 设计语言的表达力/内部自洽性(对模拟的一致性检核),有教学价值。它不支持把 ~97% 当成真实城市中「form→power 近可逆」的经验或因果断言;正确读法是「配方被重述」而非「权力被发现」。

**A2. severity=fatal · refuted=True**
- 攻擊主張:「每種權力都留下站點無關(site-invariant)的形態指紋」——被當成從 12 城 × 6 體制 = 72 樣本中湧現的經驗發現(佐證:power-effect 空間 silhouette 0.60–0.76、LOSO acc 0.972、permutation p=0.003、KMeans ARI 0.72)。
- 最強反駁:最強反駁:「站點無關」不是數據湧現,而是被生成流程 + log-ratio 特徵工程「兩段式」寫死的恆等式,證據無法把「權力製造不變指紋」和「我對每個站點套了同一個函數」區分開。

(1) 6 個「權力體制」不是觀測,是合成算子輸出。engine/operators.py 的每個算子都是純函數、參數固定(slim ratio=0.45、split_to_towers k=3、densify far_gain=1.8、scale factor=2.0、concentrate reach_frac=0.18…),regimes.yaml 把這些固定配方原封不動地套到全部 12 站的同一個「current」上。連 infill 的「隨機」抖動都是決定式 v=((ci*7+11)%100)/100。所以每站真正獨立的觀測只有 1 個(current);另外 5 個是它的決定式變換。權力指紋的有效獨立單位是「配方數=6」,不是 72;n=72 嚴重高估證據量。

(2) power_effect_space 對每特徵取 log(regime/current)。既然 regime=決定式算子(current)且參數與站點無關,這個比值 ≈ 算子的固定乘性效應 + 只由二階幾何差異決定的小殘差。site-invariance 因此是解析上被強制的,analyze_latent.py 第 15 行自己寫「若權力在此空間跨城成群,就證明權力留下站點無關指紋」——這是循環論證:normalization 先抽掉站點尺度,再宣告發現了尺度無關。

(3) 冒煙的槍:regime_effects.csv 有多個特徵在 12 站上 mean_log_ratio 與 cohens_d 精確為 0.000000。state_led 的 n / coverage / grain 全 0(它只跑 weight_height+concentrate,只動高度、從不碰 footprint);developer_scale_graded 的 h_mean / h_max / h_cv 全 0(只跑 scale_graded,只碰 footprint);shared 的 n = 0。精確為零不是「跨城穩定的經驗規律」,而是「該算子可證明不觸碰該自由度」的位元級恆等。

(4) 我實測 developer_led 的招牌特徵 slender:12 站的 regime/current 比值全落在 4.54–5.80(含 n=9979 的七宝鎮≈5.55),因為 slim(ratio=0.45)+split_to_towers(k=3) 對每站施同一固定變換;殘差 ±15% 才是唯一「數據」成分,且只反映上海鄰里幾何彼此夠像。

(5) LOSO 0.972 的「跨城泛化」是假的:12 城全抽自同一生成配方,留一城預測第 12 城當然成;它測的是決定式函數的可重現性,不是跨脈絡遷移。permutation p=0.003 只證「標籤與決定式變換非隨機相關」——當然相關。這些指標無一能反駁 null:「同一函數套處處」。

(6) 對審稿人原本懷疑的兩點,我誠實裁定:外推/小樣本(全上海、同一文化母體、每體制 12 點)成立且更嚴重——連 12 點都不是 12 次獨立體制實現。但「silhouette/ARI 被七寶鎮 n=9979 離群站左右」不成立:log-ratio 以七寶自身 current 為基準把它的尺度歸一化,它落在群內而非拉開群集;真正製造群集的是 normalization 本身,不是離群站。此角度不 land,但反而加強主攻。
- 誠實殘餘:誠實版(扣掉被駁部分後仍站得住的):「當每種權力體制被定義為一組參數固定的形態算子配方(operators.py),並統一施加於各站點的現狀時,其站點歸一化(log-ratio)的相對形態簽名在 12 個上海站點間近乎恆定(招牌特徵的跨站比值變異僅約 ±15%,部分特徵因算子不觸碰該自由度而精確恆等)。」這是算子庫設計的可重現性/一致性性質——是被『展示』而非被『發現』的,殘餘變異只證明這 12 個上海鄰里的真實建築幾何彼此相似到讓固定算子的相對效應變化很小。它完全不支持『真實世界的政治經濟權力會在真實城市留下站點無關的形態指紋』——後者未被檢驗(regimes.yaml 第 19 行自陳這些是「教學假設/可爭論的語言,不是經驗斷言或真實規劃預測」)。要把主張升級為經驗發現,至少需要:體制標籤來自獨立觀測而非合成算子、跨越非上海/非同一文化母體的城市、且算子參數不被人為固定為站點無關。

**A3. severity=fatal · refuted=True**
- 攻擊主張:「developer_led→slender、state_led→concentration 等 power→form 規律是穩健的(經驗)發現」——即這些跨 12 站點、Cohen's d≈6、LOSO acc=0.97 的規律,構成關於「權力如何塑造城市形態」的穩健發現。
- 最強反駁:這些不是發現,是同義反覆(tautology by construction)。整條管線是:(1) 在 regimes.yaml 手工指定每個體制的算子配方;(2) measure.py 量測;(3) analyze_latent.py「發現」每個體制帶有它被建構出來要產生的指紋。逐條對照即可證明:developer_led 的配方(regimes.yaml L34)就是 slim(ratio=0.45)+densify,而 slim 的定義(operators.py L140)是 footprint×0.45、height÷0.45,數學上使 slender=h/√a 機械地 ×1/0.45^1.5≈3.3(再乘 densify 的 ×1.8)——YAML 的 feature 欄甚至先寫死「瘦長比明顯升高」。state_led 的 concentrate(center=state_centroid, reach_frac=0.18)把 GFA 抬到某重心 λ=0.18·span 內,而 concentration 指標(measure.py L41)量的正是「同一個重心、同一個 0.18」內的 GFA 佔比——算子與指標共用參照,升高是恆等式。所以 d≈6 不是「異常強的真實效應」,而是「確定性算子」的指紋:配對 std→0(固定的乘法變換對每站施加近乎相同的相對位移),d=mean/std 才會爆炸。作者把「跨站低變異」讀成「穩健」,但低變異是決定論保證的、毫無資訊量——真正的穩健需要在獨立生成的樣本間存活,這裡 12 站只是同一條規則被重跑 12 次(唯一經驗輸入是各站 current 初始形態)。n=12 完全足以估計 d≈6(4 對就夠),問題方向相反:d≈6 本身就是決定論/退化 DGP 的警訊,在真實社會過程中幾乎不可能出現。LOSO acc=0.97、p=0.003 同理:分類器只是在辨認「施加了哪個確定性算子」,一旦算子產生互異且一致的位移,高準確率是保證的,置換檢定只證明標籤非隨機(=算子身分,恆真)。至於 resident_self_build 的 coverage↓,審稿人的假設經數值坐實為 infill 網格化人工產物:buffer(-step*0.06) 對 step≈10.95m 的網格向內侵蝕 ~0.66m,滿格 120→(10.95−1.31)²=92.9m²=−22.6%,加上丟棄 <21.6m² 的碎片→≈−25~30%;實測 caoyang −27.4%、dapuqiao −24.0%、kongjiang −29.9%、pengpu −27.1%——在 1.2~19.9 km² 差異極大的站點間卻是近乎固定的比例,這正是「固定侵蝕緩衝」而非「站點敏感的權力語意」的鐵證,且其符號(自建→低覆蓋)還與真實非正式聚落(通常高覆蓋)相反。附帶:h_max 在多站被 cap 釘死(developer_led=480.0=densify cap_m;state_led=600.0=concentrate cap_m),特徵報的是參數而非形態。最後,作者自己在 regimes.yaml L19 白紙黑字寫「這些 regime→形態 是教學假設/可爭論的語言,不是經驗斷言或真實規劃預測」——「穩健發現」的敘事直接違背原始碼的自我聲明。
- 誠實殘餘:扣掉「經驗發現」的框架後仍站得住的誠實版本:這是一套「可重現的生成式設計文法(generative design grammar)」。每個權力體制被操作化為一組具名、確定性的形態算子配方;在 12 個異質的上海街道初始形態上施加後,能產生「內部一致、彼此可區分」的形態簽名。可辯護的斷言只到「編碼的表達力與內部建構效度(construct validity)」層級——即:(a) 這些配方是可重現的(seed 固定、零 AI 推斷、可由 features.csv 重算);(b) 不同配方確實在 9 維形態空間裡分離得很乾淨(silhouette、LOSO、ARI 量的是「文法把自己的體制分得多開」);(c) 每個體制的形態變化構成一個語意連貫的形態論述束(如 developer_led 同時 far↑、slender↑、grain↓ = 一個可討論的「塔樓化高密度」語言)。哪些是引擎必然(不可當發現引用):developer_led→slender、state_led→concentration、shared→coverage↓、resident→grain↓/n↑ 皆為算子定義的恆等結果;resident→coverage↓ 更同時是 infill 侵蝕的數值人工產物(符號可能是錯的)。哪些較可辯護(有真實建模自由度、非純恆等):developer_scale_graded→coverage(d=1.81,跨站變異明顯較大,因 scale_graded 的漸變衰減使效應真的依站點而變);以及各體制「共變模式/相對排序」作為設計語彙的提案。定位應是教學與設計探索工具,而非關於「真實政治經濟→城市形態」的外部效度發現;所有 d 與分類準確率應標示為「文法自我分離度」,並附上「這是被建構、非被觀測」的免責聲明(如作者原意)。

## 附錄 B — 完備性評審缺口(critic)
- **[unrun_method]** 零不確定性量化:研究自承『有效獨立單位=6配方』(claim 5),卻未把它折進任何統計。LOSO 0.972、Cohen's d 高達5.92、silhouette 0.60–0.76、ARI 0.721、反演0.944 全是單抽點估計,無 bootstrap/CI、無 leave-one-recipe 穩定度、AE/VAE 只有單一 seed+單一架構(16 hidden,4000 epoch)無跨 seed 變異。以6個確定性單位報這些精確數值卻不給穩定區間,是最該補的一項。
- **[overclaim]** Claim 4『developer_scale_graded 是全語料中最接近有真實建模自由度的一條』被 operators.py 直接反證:全管線零隨機性(infill 的 vary 是固定索引雜湊 ((ci*7+11)%100)/100,scale_graded 是純 exp 衰減幾何),沒有任何『建模自由度』。它模糊的真正機制是——只施加單一弱算子、且 n/h_mean/h_max/h_cv 四項與 current 完全相同(features.csv 可逐格核對,9特徵有5項不變)→ 貼近原點。把『弱確定性重疊』說成『真實建模自由度/equifinality』是自審漏掉的實質 overclaim。
- **[unverified_claim]** Claim 4『developer_scale_graded 跨站變異明顯較大』無任何計算支撐:results.json 未報任何 per-regime 在效應空間的離散度統計。由 features.csv 原始值看 coverage 變化約 +1.5%(pengpu)到 +37%(xujiahui/waitan),此主張很可能為真,但屬未量化、未落盤的斷言。
- **[unread_source]** claim 之出處無稽核軌跡:workflow_latent_research.mjs(:122-126)設計為由 Synthesize 階段產出 report_markdown + headline_claims,並經 adversarial 判決,但 REPORT.md 與 VERDICT/CRITIQUE 產物在磁碟上皆不存在。支撐全部5條頭條主張的『對抗判決:fatal』與護欄折返敘事無法被核對——無從確認主張是走過該對抗管線產生,還是事後手寫。
- **[unread_source]** 未被承認的輸入自由度:所有算子(freeze/scale/weight_height/concentrate)作用的 state/developer/resident 標籤,是 common.assign_all 透過 shanghai_lookup.yaml 級聯查表指派(common.py:136-141,『非产权考证』、改 yaml 即變)。故語料不是『算子作用於 current』,而是『算子作用於一個啟發式 stakeholder 標記的 current』。5條主張都未surface此點——這是第二個作者可調旋鈕,同樣餵進每個效應,進一步壓低外部效度。
- **[overclaim]** Claim 1 把 PCA/AE/VAE 當三個互證方法(『三法一致』),但三者非獨立:power-effect PCA 前2主成分already 82%、前3達96%,AE recon MSE=0.0017 證明流形本質是線性2D;VAE recon MSE=0.227 與 AE 差兩個數量級卻報出相近 silhouette。『三法一致』只是重複回收同一低維線性結構,誇大了方法獨立性/穩健性。
- **[unrun_method]** 顯著性檢定的精度與虛無結構偏弱(注:prompt 誤把 permutation test 列為未跑——它其實已跑,results.json:94-100)。但 n_perm=300 使最小可達 p=(0+1)/301=0.0033,報出的 p=0.0033 代表0次超過觀測值,實為解析度地板『p<0.0033』而非精確值;且置換自由洗牌全72標籤,無視研究自承的 site-block/有效N=6 結構。誠實的虛無應是 block 或 leave-one-recipe 置換,未跑。
- **[unrun_method]** 缺外部錨點/held-out(prompt 所指『以真實城市 held-out 驗證』確實缺,且以現設計不可跑——真實城市無 ground-truth regime 標籤)。9特徵詞彙從未在任何非由這6配方生成的資料上測試:無真實城市配獨立治理/權力標籤,亦無 held-out 的獨立第三方算子配方。所有分離/準確率/效應皆算在算子自身確定性輸出上→完全循環。claim 3/5 已誠實承認此為 fatal,故非隱藏 overclaim,但這是封頂整個研究的方法缺口,連 claim 1 誠實改寫後的『量測器敏感度』也無任何獨立校準點。
- **[overclaim]** 用詞誇大地理獨立性:『12 上海城/跨城 LOSO 泛化』其實是上海『12 街道(subdistrict)』,共享同一法規與建物母體(caoyang/waitan/lujiazui 等皆為市內街道)。稱 LOSO 為『跨城泛化』高估了脈絡獨立性,並強化空間自相關與有效N過小之疑慮。WORKFLOW_PLAN 護欄已承認『同一城市文化母體內』,但頭條 claim 3 仍鬆散用『12 城』。
- **[unrun_method]** 未檢查特徵是否只是在讀算子參數(參數洩漏):h_max 頻繁飽和在算子 cap——developer_led 有5站 h_max=480.0 恰為 cap_m:480,state_led yuyuan/xujiahui h_max=600.0 恰為 cap_m:600;grain/slender 亦直接來自 slim/split 的 ratio。無任何分析區分『湧現形態』與『直接編碼算子超參』,這會人為抬高 claim 3 的可分離度與 claim 2 的效應量。
- **[unrun_method]** 未驗特徵的必要性/冗餘:claim 1 稱『9個形態特徵構成的量測詞彙』,但無 ablation/feature-importance/共線性(VIF)證明9項都必要;power-effect PCA 顯示有效維僅約2–3,且部分特徵對某些 regime 與 current 完全相同(developer_scale_graded 的 n/h_mean/h_max/h_cv)。『9特徵詞彙』輕微膨脹了真正在做分離的有效維度。

**最該補(top gap)**:零不確定性量化。研究自己承認有效獨立單位只有6個配方(非72樣本),卻從未把這個承認折進統計:LOSO 0.972、Cohen's d 5.92、silhouette 0.60–0.76、ARI 0.721、反演0.944 全部以乾淨點估計呈現,沒有任何 bootstrap/信賴區間、沒有 leave-one-recipe 穩定度、AE/VAE 只有單一 seed 單一架構單次訓練無跨 seed 變異,連 permutation p 都被 n_perm=300 釘在 1/301 地板。用6個確定性單位報這些精確數字卻不給穩定範圍,是可跑、影響每一條頭條數值、且未被任何現有 hedge 涵蓋的最該補項。
