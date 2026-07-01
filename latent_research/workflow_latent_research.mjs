// workflow_latent_research.mjs
// =====================================================================
// DF26 WS18 — 循環城市 AI:利益相關者 ↔ 城市形態 雙向映射的「潛在研究」workflow
// ---------------------------------------------------------------------
// 這是一支 Claude Code Workflow 腳本(agent 編排層),跑在 Python 研究管線之上:
//   Python 已產出硬數字 → out/latent_research/{features.csv, results.json,
//   regime_effects.csv, latent_coords.csv, figures/*.png}
//   本 workflow 負責「研究推理層」:把數字放回 05 引擎的程式碼脈絡裡解讀、
//   對三個頭條主張做對抗性反駁、綜合成有誠實護欄的研究報告、再由完備性評審找漏。
//
// 深度設計(比 10 階段版更進一步):12 階段,含
//   ‧ grounding(先讀 operators/measure/regimes,解讀不得脫離程式碼)
//   ‧ 語料稽核(features.csv 與引擎定義對帳,防幻覺 / 防串資料)
//   ‧ pipeline 式四問解讀(Q1 潛在空間 / Q2 power→form / Q3 form→power / Q4 可逆性)
//   ‧ 對抗性驗證面板(3 個 skeptic 各攻一個要害:洩漏 / 循環論證 / 小樣本外推)
//   ‧ 誠實化 reconcile(把 skeptic 判決折進主張)
//   ‧ 完備性評審(loop-until-clean 一輪補洞)
//
// 執行:Workflow({ scriptPath: "latent_research/workflow_latent_research.mjs" })
//   回傳 { report_markdown, claims, audit, critique } — 主執行緒據此寫 REPORT.md。
// 護欄:零 AI 推斷寫進資料;所有主張須可由 features.csv / 引擎程式碼重算;
//       skeptic 預設「可疑」,證據不足即降級主張。
// =====================================================================

export const meta = {
  name: 'latent-research-power-to-form',
  description: 'DF26 WS18 潛在研究:把 12城×6體制×9特徵 的形態語料解讀成「利益相關者↔城市形態」雙向映射報告,含對抗性驗證',
  whenToUse: '已用 extract_features.py + analyze_latent.py 產出 out/latent_research/* 後,要把數字綜合成有護欄的研究報告時',
  phases: [
    { title: 'Ground', detail: '讀 operators/measure/regimes:6 體制做什麼、9 特徵是什麼' },
    { title: 'Audit', detail: 'features.csv 與引擎定義對帳(72×9、平衡、量級合理)' },
    { title: 'Interpret', detail: 'Q1 潛在空間 · Q2 power→form · Q3 form→power · Q4 可逆性' },
    { title: 'Adversarial', detail: '3 skeptic 反駁頭條主張:洩漏/循環論證/小樣本外推' },
    { title: 'Reconcile', detail: '把 skeptic 判決折進誠實主張集' },
    { title: 'Synthesize', detail: '綜合 REPORT.md(含護欄與 figure 對照)' },
    { title: 'Critic', detail: '完備性評審:缺什麼(未跑的方法/未驗的主張/未讀的來源)' },
  ],
}

// --------------------------------------------------------------------- 路徑常數
const ROOT = 'latent_research'
const OUT = 'out/latent_research'
const ENGINE = '05_Shanghai Power to Form Students/engine'
const REGIMES_YAML = '05_Shanghai Power to Form Students/regimes.yaml'

// --------------------------------------------------------------------- Schemas
const ENGINE_SPEC = {
  type: 'object',
  required: ['regimes', 'features', 'notes'],
  properties: {
    regimes: {
      type: 'array',
      description: '6 體制各自的算子配方與預期形態效應',
      items: {
        type: 'object',
        required: ['name', 'operators', 'expected_effect'],
        properties: {
          name: { type: 'string' },
          operators: { type: 'array', items: { type: 'string' } },
          expected_effect: { type: 'string', description: '這串算子預期把形態往哪推(引 regimes.yaml 的 feature 欄)' },
        },
      },
    },
    features: {
      type: 'array',
      description: '9 個 measure.diagnose 特徵的精確定義',
      items: {
        type: 'object',
        required: ['name', 'definition'],
        properties: { name: { type: 'string' }, definition: { type: 'string' } },
      },
    },
    notes: { type: 'string', description: '守恆/不守恆、frozen、concentration 固定參照等關鍵語意' },
  },
}

const AUDIT = {
  type: 'object',
  required: ['n_rows', 'n_sites', 'n_regimes', 'balanced', 'sanity_checks', 'verdict'],
  properties: {
    n_rows: { type: 'integer' },
    n_sites: { type: 'integer' },
    n_regimes: { type: 'integer' },
    balanced: { type: 'boolean', description: '每體制樣本數是否相等' },
    sanity_checks: {
      type: 'array',
      description: '把幾條 row 對回引擎預期(如 developer_led slender 應遠高於 current)',
      items: {
        type: 'object',
        required: ['claim', 'holds'],
        properties: { claim: { type: 'string' }, holds: { type: 'boolean' }, evidence: { type: 'string' } },
      },
    },
    verdict: { type: 'string', enum: ['clean', 'suspect', 'broken'] },
  },
}

const INTERPRETATION = {
  type: 'object',
  required: ['question', 'reading', 'evidence', 'confidence'],
  properties: {
    question: { type: 'string' },
    reading: { type: 'string', description: '這一問的結論(2-4 句),必須引用 results.json / csv 的具體數字' },
    evidence: { type: 'array', items: { type: 'string' }, description: '引用的檔案:數字 對' },
    caveats: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const VERDICT = {
  type: 'object',
  required: ['target_claim', 'attack', 'refuted', 'residual_claim'],
  properties: {
    target_claim: { type: 'string' },
    attack: { type: 'string', description: '最強的反駁論證' },
    refuted: { type: 'boolean', description: '證據是否足以推翻原主張' },
    severity: { type: 'string', enum: ['fatal', 'weakens', 'cosmetic'] },
    residual_claim: { type: 'string', description: '扣掉被駁部分後,還站得住的誠實版本' },
  },
}

const REPORT = {
  type: 'object',
  required: ['report_markdown', 'headline_claims'],
  properties: {
    report_markdown: { type: 'string', description: '完整 REPORT.md 內文(繁中,markdown,含 figure 相對路徑引用)' },
    headline_claims: { type: 'array', items: { type: 'string' }, description: '3-5 條可辯護的頭條發現' },
  },
}

const CRITIQUE = {
  type: 'object',
  required: ['gaps', 'top_gap'],
  properties: {
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        required: ['gap', 'kind'],
        properties: {
          gap: { type: 'string' },
          kind: { type: 'string', enum: ['unrun_method', 'unverified_claim', 'unread_source', 'overclaim'] },
        },
      },
    },
    top_gap: { type: 'string', description: '最該補的一項' },
  },
}

// --------------------------------------------------------------------- helper
const read = (f) => `讀 ${f}`

// ===================================================================== 主流程
// ---- Phase 1: Ground —— 三個讀者並行,把解讀錨在程式碼(barrier:後續全靠這份 spec)
phase('Ground')
log('grounding:讀 operators.py / measure.py / regimes.yaml,建立 6體制×9特徵 的權威定義')
const groundParts = await parallel([
  () => agent(
    `${read(ENGINE + '/operators.py')} 與 ${read(REGIMES_YAML)}。` +
    `列出 6 個權力體制(current + regimes.yaml 的 5 個)各自套用哪些原子算子、順序、以及 regimes.yaml 標註的預期 feature。` +
    `current = 現狀基線(不套算子)。只回結構化事實,不要評論。`,
    { label: 'ground:regimes', phase: 'Ground', schema: ENGINE_SPEC },
  ),
  () => agent(
    `${read(ENGINE + '/measure.py')}。精確定義 diagnose() 回傳的 9 個特徵:` +
    `n, far, coverage, h_mean, h_max, h_cv, grain, slender, concentration —— 各自的數學定義與單位,` +
    `特別說明 concentration 的「固定參照現狀重心 + λ」語意。回 ENGINE_SPEC(regimes 欄可留最小佔位)。`,
    { label: 'ground:features', phase: 'Ground', schema: ENGINE_SPEC },
  ),
])
const engineSpec = groundParts.filter(Boolean).reduce((acc, p) => ({
  regimes: (p.regimes && p.regimes.length > acc.regimes.length) ? p.regimes : acc.regimes,
  features: (p.features && p.features.length > acc.features.length) ? p.features : acc.features,
  notes: [acc.notes, p.notes].filter(Boolean).join(' | '),
}), { regimes: [], features: [], notes: '' })

// ---- Phase 2: Audit —— 語料與引擎對帳(防幻覺、防串資料)
phase('Audit')
const audit = await agent(
  `${read(OUT + '/features.csv')}。這是 extract_features.py 的產物,號稱 12城×6體制=72樣本×9特徵。\n` +
  `根據下列引擎定義做稽核:\n${JSON.stringify(engineSpec).slice(0, 3500)}\n` +
  `檢查:(1) 是否 72 列、12 站、6 體制、每體制 12 樣本;` +
  `(2) 至少 4 條跨體制 sanity:developer_led 的 slender 應遠高於同站 current;` +
  `state_led 的 concentration 應高於 current;resident_self_build 的 n 應遠高於 current;` +
  `shared 的 h_cv 應為各體制最低。逐條給 holds + evidence(引具體站點數值)。給 verdict。`,
  { label: 'audit:corpus', phase: 'Audit', schema: AUDIT },
)

// ---- Phase 3: Interpret —— 四個研究問題並行解讀(各自引用實際輸出檔)
phase('Interpret')
const QUESTIONS = [
  {
    key: 'Q1-latent',
    prompt:
      `${read(OUT + '/results.json')} 與 ${read(OUT + '/latent_coords.csv')}。看 power_effect_pca_evr、silhouette(pca/ae/vae 的 all 與 ex-current)、ae/vae recon MSE。` +
      `圖:${OUT}/figures/pca_power_effect.png、latent_methods_compare.png、explained_variance.png、pca_form_space.png。` +
      `解讀:在「權力效應空間」(對站點現狀取 log 比值,抵消城市尺度)裡,6 體制是否跨城成群?三種壓縮(PCA/AE/VAE)是否一致?` +
      `對比「形態空間」(絕對值,城市各自成群)說明為何要用效應空間。`,
  },
  {
    key: 'Q2-power-to-form',
    prompt:
      `${read(OUT + '/regime_effects.csv')} 與 results.json 的 regime_signatures。圖:${OUT}/figures/regime_effects_heatmap.png。` +
      `對每個非 current 體制,指出其最強形態指紋(feature + Cohen's d + %變化),並用 Phase1 的算子配方解釋「為什麼這串算子會產生這個形態效應」。` +
      `這是 power→form 的方向。`,
  },
  {
    key: 'Q3-form-to-power',
    prompt:
      `${read(OUT + '/results.json')} 的 form_to_power(loso_accuracy、per_regime_recall)。圖:${OUT}/figures/confusion_loso.png。` +
      `解讀:用「留一城交叉驗證」只憑 9 特徵反推權力體制,準確率 vs chance 0.167。哪個體制最容易被混淆、和誰混?這代表 form→power 反向映射多可靠?`,
  },
  {
    key: 'Q4-invertibility',
    prompt:
      `${read(OUT + '/results.json')} 的 invertibility(kmeans_ari_vs_regime、nearest_centroid_accuracy)與 silhouette。` +
      `綜合 Q1-Q3 判斷:利益相關者→形態 這個映射在此教學尺度上「雙向一致 / 近可逆」到什麼程度?給一個總結性的可逆性讀數。`,
  },
]
const interpretations = await parallel(
  QUESTIONS.map((q) => () => agent(
    `研究問題 ${q.key}。${q.prompt}\n每個數字都要標明來源檔。若某主張證據不足,confidence 降級並寫進 caveats。`,
    { label: `interpret:${q.key}`, phase: 'Interpret', schema: INTERPRETATION },
  )),
)
const interp = interpretations.filter(Boolean)

// ---- Phase 4: Adversarial —— 三個 skeptic 各攻一個要害(預設可疑)
phase('Adversarial')
log('對抗性驗證:三個 skeptic 分別攻擊 洩漏 / 循環論證 / 小樣本外推')
const ATTACKS = [
  {
    claim: '只憑 9 個形態特徵就能以 ~97% 跨城反推出是哪種權力(form→power 近可逆)',
    lens:
      '資料洩漏 / 特徵循環論證:9 特徵與體制標籤是否透過「同一組確定性算子」而人為相關?' +
      'concentration 的固定參照、log 比值是否讓 current 天生落原點而灌水分離度?' +
      '確定性配方下高分是否只是「重述配方」而非發現?請界定這高分「能」與「不能」宣稱什麼。',
  },
  {
    claim: '每種權力都留下站點無關(site-invariant)的形態指紋',
    lens:
      '外推 / 小樣本:僅 12 城、皆上海、72 樣本、每體制 12 點;LOSO 仍在同一城市文化母體內。' +
      '「站點無關」是否被 log-ratio 特徵工程直接保證(而非數據湧現)?silhouette/ARI 是否受單一離群站(如 七宝镇 n=9979)左右?',
  },
  {
    claim: 'developer_led→slender、state_led→concentration 等 power→form 規律是穩健發現',
    lens:
      '效應量與方法穩健性:Cohen\'s d 用配對站點 std,n=12 是否足以支撐 d≈6?' +
      'resident_self_build 的 coverage 下降是否為 infill 的 buffer(-step*0.06)人工產物而非「權力語意」?' +
      '哪些是引擎機制的必然、哪些才是可辯護的形態論述?',
  },
]
const verdicts = await parallel(
  ATTACKS.map((a) => () => agent(
    `你是敵意審稿人,預設這個主張「可疑直到證明」。主張:「${a.claim}」。\n` +
    `攻擊角度:${a.lens}\n` +
    `可讀 ${OUT}/results.json、${OUT}/features.csv、${OUT}/regime_effects.csv、${ENGINE}/operators.py、${ENGINE}/measure.py、${ROOT}/analyze_latent.py 求證。\n` +
    `給最強反駁 attack、是否 refuted、severity、以及扣掉被駁部分後仍站得住的 residual_claim(誠實版)。`,
    { label: 'skeptic', phase: 'Adversarial', schema: VERDICT },
  )),
).then((v) => v.filter(Boolean))

// ---- Phase 5+6: Reconcile + Synthesize —— 把判決折進報告
phase('Synthesize')
const report = await agent(
  `你是首席研究者,綜合下列材料寫成 REPORT.md(繁體中文,markdown)。\n\n` +
  `# 引擎定義\n${JSON.stringify(engineSpec).slice(0, 3000)}\n\n` +
  `# 語料稽核\n${JSON.stringify(audit)}\n\n` +
  `# 四問解讀\n${JSON.stringify(interp)}\n\n` +
  `# 對抗性判決(必須折進主張,不得無視)\n${JSON.stringify(verdicts)}\n\n` +
  `結構要求:\n` +
  `1. 摘要(3-5 條頭條發現,每條已扣掉被 skeptic 駁倒的部分)\n` +
  `2. 方法(兩座標系:形態空間 vs 權力效應空間;為何後者是主分析)\n` +
  `3. Q1-Q4 逐節,每個關鍵數字標來源檔,並嵌入對應 figure 的相對路徑(如 ![](out/latent_research/figures/xxx.png))\n` +
  `4. 誠實與限制(明列:確定性算子→分數上限的循環性、僅上海12城、72樣本、log-ratio 保證了什麼、哪些是引擎機制必然)\n` +
  `5. 這對工作坊主題「利益相關者↔城市形態雙向映射」的意義,與可延伸方向\n` +
  `寫作紀律:凡對抗判決標 refuted=true 或 severity=fatal 者,該主張必須改寫或移除;` +
  `寧可少宣稱也不誇大。headline_claims 回可辯護的 3-5 條。`,
  { label: 'synthesize:report', phase: 'Synthesize', schema: REPORT, effort: 'high' },
)

// ---- Phase 7: Critic —— 完備性評審(找漏,一輪)
phase('Critic')
const critique = await agent(
  `完備性評審。研究已產出下列頭條主張:\n${JSON.stringify(report.headline_claims)}\n` +
  `已跑方法:PCA/AE/VAE、LOSO 多項式邏輯迴歸、KMeans ARI、最近質心反演、Cohen's d 效應量。\n` +
  `語料:12 上海城 × 6 體制 × 9 特徵。可讀 ${ROOT}/analyze_latent.py 看實作。\n` +
  `問:還缺什麼?未跑的方法(如 permutation test 檢定 97% 是否顯著高於洗牌標籤、跨體制混淆的降維可視化、` +
  `以真實城市 held-out 驗證)、未驗的主張、未讀的來源、或任何 overclaim。分類每個 gap,指出 top_gap。`,
  { label: 'critic:completeness', phase: 'Critic', schema: CRITIQUE },
)

log(`完成:頭條 ${report.headline_claims.length} 條,對抗判決 ${verdicts.length} 份,gap ${critique.gaps.length} 項`)

return {
  report_markdown: report.report_markdown,
  headline_claims: report.headline_claims,
  audit,
  interpretations: interp,
  adversarial: verdicts,
  critique,
}
