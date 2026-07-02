// build_slides.js —— 教學簡報(深色房風,對齊 99_教學檔案 既有簡報)
// 裸面板(out/figs/slides/)+ 原生文字 + 原生流程圖。node build_slides.js
const pptx = require("pptxgenjs");
const P = new pptx();
P.layout = "LAYOUT_WIDE"; P.author = "07 Latent Urban";
P.title = "用神經網路延伸權力→形態分析";

// ---- 深色房風調色盤(取自 99_教學檔案 既有簡報) ----
const BG = "12253C", DARK = "0C1A2E", CARD = "1C3252", CARD2 = "22375A";
const TX = "F2F5FA", MUT = "8B949E", LINE = "33404F";
const BLUE = "5AB0FF", GOLD = "F2C14E", TEAL = "4EC9B0", CORAL = "F96167", PUR = "C586C0";
const FF = "Microsoft JhengHei";
const W = 13.33, H = 7.5;

// 裸面板長寬比(exact,避免壓縮變形)
const AR = {
  "p_proj_pca": 1.075, "p_proj_tsne": 1.075, "p_proj_isomap": 1.075,
  "p_loss_ae": 1.368, "p_loss_vae": 1.260, "p_loss_collapse": 1.260,
  "p_recon_bar": 1.380, "p_recon_feat": 1.550, "p_loadings": 1.760, "p_silhouette": 1.446,
  "p_signature": 1.903, "p_regime_arrows": 1.146,
  "p_gen_plaus": 1.149, "p_gen_valid": 1.138, "p_gen_interp": 1.124,
  "p_cloud_pca": 1.062, "p_cloud_ae": 1.062, "p_cloud_vae": 1.062,
  "p_ari": 1.237, "p_sil_cluster": 1.231, "p_confusion": 1.260,
  "p_bldg_pca": 1.053, "p_bldg_tsne": 1.053, "p_bldg_isomap": 1.053,
};
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 90, opacity: 0.28 });

// 概念圖(教科書式,深底同色)長寬比
const CAR = { m_pca: 1.389, m_tsne: 1.959, m_isomap: 2.373, m_ae: 1.888, m_vae: 2.083, m_cnn: 2.166, m_point: 2.171, m_latentgrid: 0.991, m_cae_recon: 3.184 };
const CITY = { regime_compare: 0.799, city_3d: 0.989, skyline: 0.805, regime_3d_pair: 1.286 };
// 任意路徑+長寬比置中放圖(白卡承載,05 城市圖為白底)
function pfit(s, path, ar, bx, by, bw, bh, cardBg) {
  let w = bw, h = w / ar; if (h > bh) { h = bh; w = h * ar; }
  const x = bx + (bw - w) / 2, y = by + (bh - h) / 2;
  if (cardBg) s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: x - 0.1, y: y - 0.1, w: w + 0.2, h: h + 0.2, fill: { color: cardBg }, rectRadius: 0.05, shadow: shadow() });
  s.addImage({ path, x, y, w, h });
}
// 依長寬比置中放圖,不變形
function fit(s, key, bx, by, bw, bh) {
  const ar = AR[key]; let w = bw, h = w / ar;
  if (h > bh) { h = bh; w = h * ar; }
  s.addImage({ path: `out/figs/slides/${key}.png`, x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h });
}
function cfit(s, key, bx, by, bw, bh) {
  const ar = CAR[key]; let w = bw, h = w / ar;
  if (h > bh) { h = bh; w = h * ar; }
  s.addImage({ path: `out/figs/concept_ml/${key}.png`, x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h });
}
// 額外圖(論點/GNN/案例)長寬比 + 放圖
const EAR = { m_thesis: 2.649, m_graph: 2.332, m_gnn_mp: 2.431, cs_embeddings: 3.237, cs_vae_street: 3.237, cs_gnn_building: 3.237, cs_generative: 3.237 };
function efit(s, key, bx, by, bw, bh) {
  const ar = EAR[key]; let w = bw, h = w / ar;
  if (h > bh) { h = bh; w = h * ar; }
  s.addImage({ path: `out/figs/extra/${key}.png`, x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h });
}
// 案例研究:滿版大圖 + 小字出處
function caseSlide(imgKey, ttl, cite) {
  const s = P.addSlide(); pg++; s.background = { color: DARK };
  s.addImage({ path: `out/case/${imgKey}.jpg`, x: 0, y: 0, w: W, h: H, sizing: { type: "cover", w: W, h: H } });
  s.addShape(P.shapes.RECTANGLE, { x: 0, y: 5.35, w: W, h: 2.15, fill: { color: DARK, transparency: 14 } });
  tbox(s, "案例研究 · 深度學習 × URBAN", { x: 0.7, y: 5.55, w: 11, h: 0.35, fontSize: 13, bold: true, color: GOLD, charSpacing: 2 });
  tbox(s, ttl, { x: 0.7, y: 5.95, w: 12, h: 0.9, fontSize: 25, bold: true, color: TX });
  tbox(s, cite, { x: 0.7, y: 6.98, w: 12, h: 0.4, fontSize: 10.5, color: "AEB6C2" });
  pageNum(s, pg);
  return s;
}
function tbox(s, txt, o) { s.addText(txt, Object.assign({ fontFace: FF, color: TX, margin: 0 }, o)); }
function kicker(s, t, c = TEAL) { tbox(s, t, { x: 0.7, y: 0.42, w: 11, h: 0.35, fontSize: 13, bold: true, color: c, charSpacing: 2 }); }
function title(s, t) { tbox(s, t, { x: 0.7, y: 0.74, w: 12.2, h: 0.72, fontSize: 29, bold: true, color: TX }); }
function pageNum(s, n) { tbox(s, String(n), { x: W - 0.85, y: H - 0.48, w: 0.5, h: 0.3, fontSize: 10, color: MUT, align: "right" }); }
function card(s, x, y, w, h, fill = CARD) { s.addShape(P.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.07, shadow: shadow() }); }
function contentSlide() { const s = P.addSlide(); s.background = { color: BG }; return s; }
function chip(s, x, y, txt, c) { s.addShape(P.shapes.OVAL, { x, y, w: 0.34, h: 0.34, fill: { color: c } }); tbox(s, txt, { x, y, w: 0.34, h: 0.34, fontSize: 13, bold: true, color: DARK, align: "center", valign: "middle" }); }

let pg = 0;
// ========== 封面 ==========
{
  const s = P.addSlide(); s.background = { color: DARK };
  s.addShape(P.shapes.OVAL, { x: 10.4, y: -1.7, w: 5.0, h: 5.0, fill: { color: TEAL, transparency: 82 } });
  s.addShape(P.shapes.OVAL, { x: 11.4, y: 4.4, w: 4.0, h: 4.0, fill: { color: BLUE, transparency: 85 } });
  tbox(s, "城市規劃 × 神經網路", { x: 0.9, y: 1.55, w: 11, h: 0.5, fontSize: 16, bold: true, color: BLUE, charSpacing: 3 });
  tbox(s, "用神經網路延伸「權力 → 形態」分析", { x: 0.9, y: 2.2, w: 11.6, h: 1.3, fontSize: 40, bold: true, color: TX });
  tbox(s, "把城市形態壓進可壓縮、可比較、可生成的潛空間", { x: 0.9, y: 3.7, w: 11, h: 0.6, fontSize: 19, color: MUT });
  card(s, 0.9, 5.4, 11.5, 1.0, CARD);
  s.addText([
    { text: "對象  ", options: { bold: true, color: GOLD } }, { text: "已學過 05 Shanghai、Python 還不熟的規劃學生      ", options: { color: TX } },
    { text: "目的  ", options: { bold: true, color: GOLD } }, { text: "重觀念、重直覺、重圖解(非實作)", options: { color: TX } },
  ], { x: 1.15, y: 5.55, w: 11, h: 0.7, fontFace: FF, fontSize: 14, valign: "middle" });
  s.addNotes("概念與啟發課,不帶敲程式。理解怎麼把 NN 接進已會的都市分析流程。");
}
// ========== §0 動機 ==========
{
  const s = contentSlide(); pg++; kicker(s, "0 · 開場"); title(s, "規劃師為什麼要碰神經網路?");
  // 左:凌亂 9 格(native)
  card(s, 0.7, 1.75, 4.5, 4.6, CARD);
  tbox(s, "人腦:一次看 9 張分佈圖", { x: 0.7, y: 1.95, w: 4.5, h: 0.4, fontSize: 14, bold: true, color: MUT, align: "center" });
  for (let i = 0; i < 9; i++) {
    const cx = 1.05 + (i % 3) * 1.35, cy = 2.55 + Math.floor(i / 3) * 1.15;
    s.addShape(P.shapes.RECTANGLE, { x: cx, y: cy, w: 1.05, h: 0.85, fill: { color: CARD2 }, line: { color: LINE, width: 0.5 } });
    for (let b = 0; b < 4; b++) s.addShape(P.shapes.RECTANGLE, { x: cx + 0.12 + b * 0.2, y: cy + 0.85 - (0.2 + 0.14 * ((i + b) % 4)), w: 0.16, h: 0.2 + 0.14 * ((i + b) % 4), fill: { color: MUT } });
  }
  tbox(s, "眼花、看不出全貌", { x: 0.7, y: 6.0, w: 4.5, h: 0.3, fontSize: 12, color: CORAL, align: "center", italic: true });
  // 箭頭
  s.addShape(P.shapes.RIGHT_ARROW, { x: 5.5, y: 3.7, w: 1.3, h: 0.7, fill: { color: TEAL } });
  tbox(s, "NN 壓縮", { x: 5.4, y: 3.15, w: 1.5, h: 0.4, fontSize: 13, bold: true, color: TEAL, align: "center" });
  // 右:成簇 dots(native ovals)
  card(s, 7.1, 1.75, 5.5, 4.6, CARD);
  tbox(s, "壓成一張 2D 潛空間圖", { x: 7.1, y: 1.95, w: 5.5, h: 0.4, fontSize: 14, bold: true, color: TX, align: "center" });
  const clus = [[BLUE, 8.4, 3.2], [GOLD, 10.9, 3.0], [TEAL, 9.7, 5.2]];
  clus.forEach(([c, cx, cy]) => { for (let k = 0; k < 12; k++) { const a = k * 1.9, r = 0.15 + (k % 4) * 0.12; s.addShape(P.shapes.OVAL, { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r * 0.8, w: 0.17, h: 0.17, fill: { color: c } }); } });
  tbox(s, "一眼看出族群", { x: 7.1, y: 6.0, w: 5.5, h: 0.3, fontSize: 12, color: TEAL, align: "center", italic: true });
  pageNum(s, pg); s.addNotes("NN 是放大鏡不是水晶球:給地圖,規劃師讀權力邏輯。");
}
// ========== 論點框架(thesis) ==========
{
  const s = contentSlide(); pg++; kicker(s, "本簡報的論點", GOLD);
  title(s, "重點不是蓋房子,是城市背後的「資訊層」");
  efit(s, "m_thesis", 0.6, 1.9, 12.1, 2.9);
  card(s, 0.7, 5.2, 11.9, 1.5);
  s.addText([
    { text: "城市不只是物理形態——背後有一層可被機器讀寫的資訊。", options: { color: TX, breakLine: true } },
    { text: "本簡報用深度學習做三件事:① 把城市編碼成潛空間(資訊層) · ② 在潛空間裡看見權力如何區分形態 · ③ 從潛空間重新生成城市。", options: { color: TX } },
  ], { x: 0.95, y: 5.35, w: 11.4, h: 1.3, fontFace: FF, fontSize: 14, valign: "top", paraSpaceAfter: 4 });
  pageNum(s, pg); s.addNotes("全簡報主線:編碼 → 權力區分 → 重新生成。");
}
// ========== §1 workflow(native) ==========
{
  const s = contentSlide(); pg++; kicker(s, "1 · 從 05 流程出發"); title(s, "NN 在哪裡介入你的工作流程?");
  const steps = ["多源\n拼接", "角色\n映射", "高度\n情景", "權力\n四體制", "3D\n量體"];
  steps.forEach((t, i) => {
    const x = 0.75 + i * 1.72;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x, y: 2.5, w: 1.5, h: 1.25, fill: { color: CARD2 }, rectRadius: 0.06, line: { color: BLUE, width: 1 } });
    tbox(s, t, { x, y: 2.5, w: 1.5, h: 1.25, fontSize: 13, bold: true, color: TX, align: "center", valign: "middle" });
    if (i < 4) s.addShape(P.shapes.RIGHT_ARROW, { x: x + 1.5, y: 2.95, w: 0.24, h: 0.34, fill: { color: MUT } });
  });
  tbox(s, "05 現有流程(不動一行)", { x: 0.75, y: 1.9, w: 8.5, h: 0.4, fontSize: 14, bold: true, color: BLUE });
  s.addShape(P.shapes.RIGHT_ARROW, { x: 9.35, y: 2.9, w: 0.7, h: 0.45, fill: { color: TEAL } });
  const nn = [["壓縮", "潛空間", BLUE], ["分析", "群聚結構", GOLD], ["生成", "新方案", TEAL]];
  nn.forEach((t, i) => {
    const y = 1.85 + i * 1.28;
    card(s, 10.2, y, 2.5, 1.1, CARD);
    tbox(s, t[0], { x: 10.4, y: y + 0.14, w: 2.1, h: 0.4, fontSize: 15, bold: true, color: t[2] });
    tbox(s, t[1], { x: 10.4, y: y + 0.58, w: 2.1, h: 0.4, fontSize: 12, color: MUT });
  });
  tbox(s, "NN 從輸出接手", { x: 10.2, y: 1.5, w: 2.5, h: 0.3, fontSize: 12, bold: true, color: TEAL, align: "center" });
  card(s, 0.75, 4.5, 8.35, 1.7, CARD);
  s.addText([
    { text: "不改資料來源。", options: { bold: true, color: GOLD, breakLine: true } },
    { text: "NN 讀的就是 05 拼接好的 buildings.parquet 與 9 個形態指標。", options: { color: TX, breakLine: true } },
    { text: "三種資料粒度:每棟樓→向量 · 整城→高度影像 · 每城×體制→9 維形態", options: { color: MUT } },
  ], { x: 1.0, y: 4.7, w: 7.9, h: 1.4, fontFace: FF, fontSize: 14, valign: "top", paraSpaceAfter: 6 });
  pageNum(s, pg); s.addNotes("NN 接在 05 輸出後,不動 05 一行。");
}
// ========== §2 三代方法(native cards) ==========
{
  const s = contentSlide(); pg++; kicker(s, "2 · 方法演進"); title(s, "三代方法:查表 → 非監督 ML → 神經網路");
  const gens = [
    ["①", "規則 Lookup", "人寫死規則(如分區規範)", "透明可控,但不會自己發現結構", BLUE, "lookup table"],
    ["②", "非監督 ML", "數學找資料主結構", "找族群,但多半不能生成", GOLD, "PCA · t-SNE · Isomap"],
    ["③", "神經網路", "網路學壓縮與重建", "能壓、能重建、能生成", TEAL, "AE · VAE · CNN"],
  ];
  gens.forEach((g, i) => {
    const x = 0.75 + i * 4.05;
    card(s, x, 1.85, 3.75, 3.9);
    chip(s, x + 0.25, 2.1, g[0], g[4]);
    tbox(s, g[1], { x: x + 0.75, y: 2.12, w: 2.9, h: 0.4, fontSize: 17, bold: true, color: g[4] });
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: 2.72, w: 3.25, h: 0.42, fill: { color: CARD2 }, rectRadius: 0.05 });
    tbox(s, g[5], { x: x + 0.25, y: 2.72, w: 3.25, h: 0.42, fontSize: 13, bold: true, color: g[4], align: "center", valign: "middle" });
    tbox(s, g[2], { x: x + 0.25, y: 3.25, w: 3.25, h: 0.8, fontSize: 13.5, color: TX });
    s.addShape(P.shapes.LINE, { x: x + 0.25, y: 4.05, w: 3.25, h: 0, line: { color: LINE, width: 1 } });
    tbox(s, g[3], { x: x + 0.25, y: 4.2, w: 3.25, h: 1.4, fontSize: 13, color: MUT });
  });
  card(s, 0.75, 5.85, 11.8, 0.85, "1A2E22");
  tbox(s, "關鍵差異:神經網路會「生成」——造一個沒看過但合理的城市;前兩代不會。",
    { x: 0.75, y: 5.85, w: 11.8, h: 0.85, fontSize: 15, bold: true, italic: true, color: TEAL, align: "center", valign: "middle" });
  pageNum(s, pg); s.addNotes("AE/VAE 嚴格是自監督;對學生只要記:它們會生成,前兩代不會。");
}
// ========== Section divider ==========
function divider(no, zh, en) {
  const s = P.addSlide(); s.background = { color: DARK };
  s.addShape(P.shapes.OVAL, { x: 0.9, y: 2.75, w: 1.5, h: 1.5, fill: { color: TEAL } });
  tbox(s, no, { x: 0.9, y: 2.75, w: 1.5, h: 1.5, fontSize: 44, bold: true, color: DARK, align: "center", valign: "middle" });
  tbox(s, zh, { x: 2.8, y: 2.9, w: 9.8, h: 0.9, fontSize: 38, bold: true, color: TX });
  tbox(s, en, { x: 2.85, y: 3.85, w: 9.8, h: 0.5, fontSize: 15, color: BLUE, charSpacing: 2 });
  return s;
}
divider("3", "非監督 ML 的原理", "PCA · t-SNE · ISOMAP  —  把高維壓到 2D,差別在保留什麼");
// 城市 = 一個點(向量化 why/how)
{
  const s = contentSlide(); pg++; kicker(s, "3.0 · 前提"); title(s, "先把每一棟建築變成「一個點」");
  cfit(s, "m_point", 0.5, 1.85, 12.4, 3.5);
  const cc = [["為什麼要這樣做?", "電腦不懂「建築」,只懂數字。把每棟樓的高度/面積/瘦高/屋齡/地價/用途排成一串,建築就成了可計算的向量。", GOLD],
  ["這個點代表什麼?", "座標 = 這棟建築的形態身分。兩點靠近 = 形態相似;整群點的排列 = 建築之間的關係。", TEAL],
  ["接下來要做什麼?", "一條街幾千棟建築 = 幾千個點,散在十幾維空間;人眼看不到 → 用降維壓到 2D 才看得見。", BLUE]];
  cc.forEach((k, i) => { const x = 0.6 + i * 4.12; card(s, x, 5.5, 3.9, 1.55); tbox(s, k[0], { x: x + 0.2, y: 5.62, w: 3.6, h: 0.35, fontSize: 13.5, bold: true, color: k[2] }); tbox(s, k[1], { x: x + 0.2, y: 6.0, w: 3.55, h: 1.0, fontSize: 11.5, color: TX }); });
  pageNum(s, pg); s.addNotes("向量化=把城市翻譯成電腦能算的座標。降維(下幾頁)就是把這些高維點壓到 2D 看關係。");
}
// PCA
{
  const s = contentSlide(); pg++; kicker(s, "3.1 · PCA"); title(s, "PCA:找城市變化的「主軸」");
  cfit(s, "m_pca", 0.4, 1.7, 7.2, 4.7);
  const c = [["直覺", "問「這團建築點雲最長的方向是哪條」,那就是 PC1", TEAL],
  ["可解讀", "PC1 = 強度/資本軸(負=高聳密集,正=瘦長細碎)", BLUE],
  ["規劃意涵", "近半城市差異能用單一強度軸解釋", GOLD]];
  c.forEach((k, i) => { const y = 1.95 + i * 1.5; card(s, 7.0, y, 5.6, 1.32); tbox(s, k[0], { x: 7.25, y: y + 0.13, w: 5.1, h: 0.35, fontSize: 14, bold: true, color: k[2] }); tbox(s, k[1], { x: 7.25, y: y + 0.55, w: 5.2, h: 0.7, fontSize: 13, color: TX }); });
  pageNum(s, pg); s.addNotes("PCA 是唯一有可解讀軸、又能逆回原空間的統計法。");
}
// t-SNE
{
  const s = contentSlide(); pg++; kicker(s, "3.2 · t-SNE"); title(s, "t-SNE:只在乎「誰跟誰是鄰居」");
  cfit(s, "m_tsne", 0.5, 2.0, 7.0, 4.0);
  card(s, 7.2, 2.1, 5.4, 3.4);
  s.addText([
    { text: "像把同學按「常一起玩」分小圈圈：", options: { color: TX, breakLine: true } },
    { text: "圈內對,但圈跟圈之間的距離沒意義。", options: { color: CORAL, bold: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "適合看聚類;小心:別讀團間距離、參數(perplexity)敏感、不穩定。", options: { color: MUT } },
  ], { x: 7.45, y: 2.35, w: 4.9, h: 3.0, fontFace: FF, fontSize: 14, valign: "top" });
  pageNum(s, pg); s.addNotes("perplexity 換一下圖就變。只看聚類、不解讀距離。");
}
// Isomap
{
  const s = contentSlide(); pg++; kicker(s, "3.3 · Isomap"); title(s, "Isomap:把摺起來的紙攤平");
  cfit(s, "m_isomap", 0.4, 2.1, 7.2, 3.6);
  card(s, 7.2, 2.1, 5.4, 3.4);
  s.addText([
    { text: "資料像捲起來的紙,直線距離會「穿牆」。", options: { color: TX, breakLine: true } },
    { text: "Isomap 沿紙面(測地線)量距離再攤平,", options: { color: TX, breakLine: true } },
    { text: "保全局結構。", options: { color: TEAL, bold: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "和 t-SNE 一樣:沒有可解讀軸、不能逆回城市。", options: { color: MUT } },
  ], { x: 7.45, y: 2.35, w: 4.9, h: 3.0, fontFace: FF, fontSize: 14, valign: "top" });
  pageNum(s, pg); s.addNotes("保流形全局結構。");
}
// 三方法同台
{
  const s = contentSlide(); pg++; kicker(s, "3 · 小結"); title(s, "同一份建築語料(lujiazui 3263 棟),三種投影");
  const labs = [["p_bldg_pca", "PCA", BLUE], ["p_bldg_tsne", "t-SNE", GOLD], ["p_bldg_isomap", "Isomap", TEAL]];
  labs.forEach((l, i) => { const x = 0.7 + i * 3.05; fit(s, l[0], x, 1.9, 2.95, 3.0); tbox(s, l[1], { x, y: 1.65, w: 2.95, h: 0.3, fontSize: 15, bold: true, color: l[2], align: "center" }); });
  // 圖例(native):顏色=建築高度
  card(s, 10.0, 1.9, 2.7, 1.6);
  tbox(s, "顏色 = 建築高度", { x: 10.2, y: 2.05, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: MUT });
  for (let i = 0; i < 24; i++) { const t = i / 23; s.addShape(P.shapes.RECTANGLE, { x: 10.25 + i * 0.09, y: 2.5, w: 0.1, h: 0.35, fill: { color: mix(["440154", "21908C", "FDE725"], t) }, line: { type: "none" } }); }
  tbox(s, "低", { x: 10.2, y: 2.9, w: 0.5, h: 0.3, fontSize: 11, color: TX }); tbox(s, "高", { x: 12.1, y: 2.9, w: 0.5, h: 0.3, fontSize: 11, color: TX, align: "right" });
  // 對照表(native table)
  s.addTable([
    [{ text: "", options: { fill: { color: CARD2 } } }, { text: "全局", options: { fill: { color: CARD2 }, color: TX, bold: true, align: "center" } }, { text: "可解讀軸", options: { fill: { color: CARD2 }, color: TX, bold: true, align: "center" } }, { text: "能生成", options: { fill: { color: CARD2 }, color: TX, bold: true, align: "center" } }],
    [{ text: "PCA", options: { color: TX } }, { text: "✔", options: { color: TEAL, align: "center" } }, { text: "✔", options: { color: TEAL, align: "center" } }, { text: "✔", options: { color: TEAL, align: "center" } }],
    [{ text: "t-SNE", options: { color: TX } }, { text: "✘", options: { color: CORAL, align: "center" } }, { text: "✘", options: { color: CORAL, align: "center" } }, { text: "✘", options: { color: CORAL, align: "center" } }],
    [{ text: "Isomap", options: { color: TX } }, { text: "✔", options: { color: TEAL, align: "center" } }, { text: "✘", options: { color: CORAL, align: "center" } }, { text: "✘", options: { color: CORAL, align: "center" } }],
  ], { x: 10.0, y: 3.8, w: 2.7, colW: [0.95, 0.55, 0.65, 0.55], rowH: 0.42, fontFace: FF, fontSize: 12, valign: "middle", border: { pt: 0.5, color: LINE }, fill: { color: BG } });
  tbox(s, "PCA 能說「PC1 是什麼」;t-SNE/Isomap 只給座標、回不去。", { x: 0.7, y: 6.4, w: 9, h: 0.4, fontSize: 13, color: MUT, italic: true });
  pageNum(s, pg); s.addNotes("顏色=家族,形狀=體制。");
}
// 案例:神經嵌入
caseSlide("embeddings",
  "城市大數據的「神經嵌入」自動浮現空間結構",
  "研究:Neural embeddings of urban big data reveal spatial structures in cities, Nature HSSC, 2024。　示意圖:城市衛星影像,CC BY-SA · Wikimedia Commons。");
// ========== Section:NN ==========
divider("4", "神經網路的原理", "AE · VAE · CNN · GNN  —  以 AE、VAE 為主");
// AE hourglass(native)
{
  const s = contentSlide(); pg++; kicker(s, "4.1 · AE 自編碼器"); title(s, "AE:學徒「臨摹 → 只憑幾個數字默寫」");
  cfit(s, "m_ae", 0.4, 1.85, 8.6, 4.5);
  card(s, 9.3, 2.2, 3.4, 3.2);
  s.addText([
    { text: "瓶頸 = 潛碼", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "壓到最窄還能畫回來,代表抓住了城市形態的精華。", options: { color: TX, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "非線性壓縮 → 同 2 維,重建遠比 PCA 準。", options: { color: TEAL, bold: true } },
  ], { x: 9.55, y: 2.45, w: 2.9, h: 2.7, fontFace: FF, fontSize: 14, valign: "top" });
  pageNum(s, pg); s.addNotes("編碼器→潛碼→解碼器。潛碼=城市形態DNA。");
}
// VAE continuity(native)
{
  const s = contentSlide(); pg++; kicker(s, "4.2 · VAE 變分自編碼器"); title(s, "VAE:學會「風格」才能創作");
  cfit(s, "m_vae", 0.4, 1.85, 9.0, 4.3);
  card(s, 9.5, 1.85, 3.15, 4.3);
  s.addText([
    { text: "把潛空間規範成連續高斯雲 N(0,I)", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "→ 隨手點一下,就解碼出沒看過但合理的城市。這就是生成。", options: { color: TX, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "代價:重建略遜 AE(多了規範化)。", options: { color: MUT } },
  ], { x: 9.7, y: 2.1, w: 2.75, h: 3.8, fontFace: FF, fontSize: 13, valign: "top" });
  pageNum(s, pg); s.addNotes("AE 潛空間有洞;VAE 填滿、處處可採樣。");
}
// VAE 潛空間影像網格(concrete image-VAE)
{
  const s = contentSlide(); pg++; kicker(s, "4.2 · VAE 潛空間(概念示意)"); title(s, "在潛空間網格上移動,生成的形態平滑變化");
  cfit(s, "m_latentgrid", 0.5, 1.75, 5.6, 5.3);
  tbox(s, "示意圖(概念)", { x: 0.5, y: 1.75, w: 2.2, h: 0.35, fontSize: 12, bold: true, color: GOLD });
  card(s, 6.6, 2.0, 6.1, 4.6);
  s.addText([
    { text: "影像 VAE 的潛空間長這樣", options: { bold: true, color: GOLD, breakLine: true } },
    { text: "(MNIST 手寫數字、人臉 VAE 的經典結果;此為概念示意圖)。", options: { color: MUT, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "每一格 = 潛空間一個座標解碼出的形態。左→右密度增、下→上高度增,", options: { color: TX, breakLine: true } },
    { text: "相鄰格子平滑過渡 = 潛空間連續 → 任一點採樣都合理。", options: { color: TEAL, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "誠實邊界:本專案只有 12 城、高度圖多為空白,資料量不足以訓練出這麼清晰的「城市影像 VAE」——故此頁為概念示意,實際可用的壓縮/生成用表格 AE/VAE(前幾頁)。", options: { color: CORAL, italic: true } },
  ], { x: 6.85, y: 2.25, w: 5.6, h: 4.2, fontFace: FF, fontSize: 13.5, valign: "top", paraSpaceAfter: 3 });
  pageNum(s, pg); s.addNotes("概念示意:影像 VAE 潛空間網格。誠實說明本資料量不足以訓練出清晰城市影像 VAE。");
}
// 案例:VAE 生成街道網路
caseSlide("vae_street",
  "VAE 把街道網路壓進潛空間,並生成新網路",
  "研究:Kempinska & Murcio (2019), Modelling urban networks using Variational Autoencoders, Applied Network Science。　示意圖:城市街道網路圖,CC BY-SA · Wikimedia Commons。");
// CNN(native flow)
{
  const s = contentSlide(); pg++; kicker(s, "4.3 · CNN(影像視角)"); title(s, "CNN 讀「紋理」:概念很好,但這份資料還不夠");
  // ① 概念
  tbox(s, "① 概念:把城市當影像,卷積讀街廓紋理", { x: 0.5, y: 1.6, w: 7.2, h: 0.35, fontSize: 13, bold: true, color: TEAL });
  cfit(s, "m_cnn", 0.4, 1.95, 6.9, 2.3);
  card(s, 7.5, 1.95, 5.2, 2.3);
  s.addText([
    { text: "前面把每棟樓當一列數字,丟了空間排列。", options: { color: TX, breakLine: true } },
    { text: "把整座城燒成高度影像,卷積能讀「街廓排列、疏密」——urban form 最自然的 NN 表示。", options: { color: TX } },
  ], { x: 7.75, y: 2.2, w: 4.75, h: 1.9, fontFace: FF, fontSize: 13.5, valign: "top" });
  // ② 實際證據
  tbox(s, "② 實際:此資料上 CNN 重建 = 模糊", { x: 0.5, y: 4.4, w: 8.4, h: 0.35, fontSize: 13, bold: true, color: CORAL });
  cfit(s, "m_cae_recon", 0.4, 4.75, 8.6, 2.15);
  card(s, 9.3, 4.55, 3.4, 2.35);
  s.addText([
    { text: "為什麼糊?", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "只有 12 城、高度圖約 86% 空白、建築邊界又細又高頻 → 小模型只能學到「密度分布」,學不到清晰建築。", options: { color: TX, breakLine: true } },
    { text: "要清晰需數千張以上街廓影像。故實作壓縮/生成仍用表格 AE/VAE。", options: { color: MUT } },
  ], { x: 9.55, y: 4.75, w: 2.95, h: 2.05, fontFace: FF, fontSize: 12, valign: "top", paraSpaceAfter: 3 });
  pageNum(s, pg); s.addNotes("上=真實高度輸入、下=CNN重建=模糊。誠實展示資料量限制,不假裝 CNN 在此可用。");
}
// GNN 概念
{
  const s = contentSlide(); pg++; kicker(s, "4.4 · GNN(圖神經網路)", TEAL);
  title(s, "城市是一張「圖」:建築是節點,鄰接是邊");
  efit(s, "m_graph", 0.5, 1.9, 12.2, 3.1);
  card(s, 0.7, 5.3, 11.9, 1.45);
  s.addText([
    { text: "表格把每棟樓當獨立一列,丟了「誰挨著誰」。", options: { color: TX, breakLine: true } },
    { text: "圖(Graph)= 節點(建築/地塊)+ 邊(鄰接/街道連通)。GNN 讀的是「關係」——urban 尺度最自然的結構。", options: { color: TEAL } },
  ], { x: 0.95, y: 5.45, w: 11.4, h: 1.25, fontFace: FF, fontSize: 14, valign: "top", paraSpaceAfter: 4 });
  pageNum(s, pg); s.addNotes("GNN:建築=節點,鄰接=邊。讀關係而非孤立特徵。");
}
// GNN 訊息傳遞
{
  const s = contentSlide(); pg++; kicker(s, "4.4 · GNN 怎麼運作", TEAL);
  title(s, "訊息傳遞:每棟建築聚合街坊的資訊");
  efit(s, "m_gnn_mp", 0.5, 1.9, 12.2, 3.1);
  card(s, 0.7, 5.3, 11.9, 1.45);
  s.addText([
    { text: "① 每個節點聚合鄰居特徵 → ② 多層堆疊 → 更新成「帶鄰域脈絡」的嵌入。", options: { color: TX, breakLine: true } },
    { text: "一棟建築的向量 = 自己屬性 + 街坊關係。應用:功能分區辨識、全城建築屬性預測、街道網路分類。", options: { color: MUT } },
  ], { x: 0.95, y: 5.45, w: 11.4, h: 1.25, fontFace: FF, fontSize: 14, valign: "top", paraSpaceAfter: 4 });
  pageNum(s, pg);
}
// 案例:GNN 建築屬性
caseSlide("gnn",
  "GNN 從街坊脈絡預測全城建築屬性",
  "研究:Predicting building characteristics at urban scale using GNNs and street-level context, Computers Environment and Urban Systems, 2024。　示意圖:住宅街廓空拍,CC BY-SA · Wikimedia Commons。");
// ========== Section:結果 ==========
{
  const s = divider("5", "訓練結果", "換分析單位:每座城 × 體制 = 一個點(60 樣本 × 9 形態特徵)");
  tbox(s, "前面 = 建築層;這裡 = 城市×體制層,接權力分析", { x: 2.85, y: 4.5, w: 5.0, h: 0.4, fontSize: 12.5, color: GOLD });
  pfit(s, "out/city/city_3d.png", CITY.city_3d, 8.0, 1.4, 4.6, 4.6, "F4F7FB");
  tbox(s, "真實城市 3D 量體(曹楊)", { x: 8.0, y: 6.35, w: 4.8, h: 0.3, fontSize: 12, color: "CADCFC", align: "center" });
}
// 重建
{
  const s = contentSlide(); pg++; kicker(s, "5.1 · 重建"); title(s, "AE 完勝:2 維就贏 PCA 4 維");
  fit(s, "p_recon_bar", 0.7, 1.9, 5.6, 3.2);
  tbox(s, "整體重建 MSE(越低越好)", { x: 0.7, y: 5.2, w: 5.6, h: 0.3, fontSize: 12, color: MUT, align: "center" });
  fit(s, "p_recon_feat", 6.5, 3.5, 6.3, 3.0);
  tbox(s, "逐特徵重建 MSE", { x: 6.5, y: 6.5, w: 6.3, h: 0.3, fontSize: 12, color: MUT, align: "center" });
  const st = [["0.061", "AE 壓 2 維", CORAL], ["0.104", "PCA 壓 4 維", MUT], ["0.033", "AE 壓 3 維", TEAL]];
  st.forEach((k, i) => { const x = 6.6 + i * 2.05; card(s, x, 1.85, 1.9, 1.4); tbox(s, k[0], { x: x + 0.1, y: x ? 1.95 : 1.95, w: 1.7, h: 0.7, fontSize: 26, bold: true, color: k[2], align: "center" }); tbox(s, k[1], { x: x + 0.1, y: 2.7, w: 1.7, h: 0.5, fontSize: 12, color: TX, align: "center" }); });
  pageNum(s, pg); s.addNotes("AE 用 2 維贏過 PCA 用 4 維。");
}
// 訓練
{
  const s = contentSlide(); pg++; kicker(s, "5.2 · 訓練"); title(s, "AE/VAE 怎麼訓練出來的");
  const L3 = [["p_loss_ae", "(a) 過擬合", "裸 AE 背答案(驗證變差)", BLUE], ["p_loss_vae", "(b) 損失拆解", "VAE 重建與 KL 此消彼長", TEAL], ["p_loss_collapse", "(c) 塌縮", "β 太大壓垮 KL;退火避免", CORAL]];
  L3.forEach((l, i) => { const x = 0.7 + i * 4.05; tbox(s, l[1], { x, y: 1.75, w: 3.9, h: 0.35, fontSize: 15, bold: true, color: l[3] }); fit(s, l[0], x, 2.2, 3.9, 3.1); tbox(s, l[2], { x, y: 5.5, w: 3.9, h: 0.7, fontSize: 12.5, color: MUT }); });
  pageNum(s, pg); s.addNotes("訓練不是按一下,是調正則、調 β。");
}
// 群聚
{
  const s = contentSlide(); pg++; kicker(s, "5.3 · 群聚(最關鍵)"); title(s, "資料自己會分出「權力體制」嗎?");
  tbox(s, "KMeans 復原 regime 的吻合度 ARI", { x: 0.7, y: 1.7, w: 5, h: 0.3, fontSize: 12, color: MUT });
  fit(s, "p_ari", 0.7, 2.0, 4.4, 3.0);
  tbox(s, "最佳空間:群 × regime 混淆", { x: 5.3, y: 1.7, w: 5, h: 0.3, fontSize: 12, color: MUT });
  fit(s, "p_confusion", 5.3, 2.0, 4.4, 3.0);
  card(s, 10.0, 2.0, 2.7, 3.6);
  s.addText([
    { text: "ARI 0.70", options: { bold: true, fontSize: 24, color: GOLD, breakLine: true } },
    { text: "原始 9 維", options: { color: MUT, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "不給標籤,KMeans 只看形態就大致復原 05 手訂的 5 種體制。", options: { color: TX, breakLine: true } },
    { text: "→ regime 有形態資料支撐。", options: { color: TEAL, bold: true } },
  ], { x: 10.2, y: 2.2, w: 2.3, h: 3.2, fontFace: FF, fontSize: 13, valign: "top" });
  pageNum(s, pg); s.addNotes("資料驅動有別於手動 lookup:客觀檢驗規則。");
}
// 成簇
{
  const s = contentSlide(); pg++; kicker(s, "5.4 · 解釋性"); title(s, "誰在成簇:體制,不是城市家族");
  tbox(s, "PCA 載荷:每根主軸=哪些特徵", { x: 0.7, y: 1.75, w: 6, h: 0.3, fontSize: 12, color: MUT });
  fit(s, "p_loadings", 0.7, 2.1, 6.2, 2.3);
  tbox(s, "silhouette:按家族 vs 按體制", { x: 0.7, y: 4.6, w: 6, h: 0.3, fontSize: 12, color: MUT });
  fit(s, "p_silhouette", 0.7, 4.9, 5.2, 2.3);
  card(s, 7.3, 2.1, 5.3, 4.2);
  s.addText([
    { text: "silhouette 按體制 ≫ 按家族", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "(0.24–0.42 vs ≈0)", options: { color: MUT, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "同一套權力算子把不同城推向一致方向;城市原本的類型反而被打散。", options: { color: TX, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "權力 > 地方性。", options: { bold: true, color: BLUE, fontSize: 18 } },
  ], { x: 7.55, y: 2.35, w: 4.85, h: 3.7, fontFace: FF, fontSize: 14, valign: "top" });
  pageNum(s, pg); s.addNotes("PC1 載荷=強度/資本 vs 細粒自建。");
}
// 生成
{
  const s = contentSlide(); pg++; kicker(s, "5.5 · 生成"); title(s, "能從潛空間採樣生成新樣本的,只有 PCA / AE / VAE");
  // capability(native table)
  s.addTable([
    [{ text: "方法", options: { fill: { color: CARD2 }, color: TX, bold: true } }, { text: "能生成", options: { fill: { color: CARD2 }, color: TX, bold: true, align: "center" } }, { text: "plaus↓", options: { fill: { color: CARD2 }, color: TX, bold: true, align: "center" } }, { text: "valid↑", options: { fill: { color: CARD2 }, color: TX, bold: true, align: "center" } }],
    [{ text: "VAE", options: { color: TEAL, bold: true } }, { text: "✔ 先驗", options: { color: TEAL, align: "center" } }, { text: "1.03", options: { color: TX, align: "center" } }, { text: "100%", options: { color: TX, align: "center" } }],
    [{ text: "PCA", options: { color: TX } }, { text: "~", options: { color: GOLD, align: "center" } }, { text: "1.40", options: { color: TX, align: "center" } }, { text: "97%", options: { color: TX, align: "center" } }],
    [{ text: "AE", options: { color: TX } }, { text: "~ 無先驗", options: { color: GOLD, align: "center" } }, { text: "2.11", options: { color: TX, align: "center" } }, { text: "93%", options: { color: TX, align: "center" } }],
    [{ text: "t-SNE/Isomap", options: { color: MUT } }, { text: "✘", options: { color: CORAL, align: "center" } }, { text: "—", options: { color: MUT, align: "center" } }, { text: "—", options: { color: MUT, align: "center" } }],
  ], { x: 0.7, y: 1.95, w: 5.6, colW: [2.0, 1.5, 1.05, 1.05], rowH: 0.5, fontFace: FF, fontSize: 13, valign: "middle", border: { pt: 0.5, color: LINE }, fill: { color: BG } });
  tbox(s, "能生成 = 有沒有 decoder。t-SNE/Isomap 只會投影、回不去。", { x: 0.7, y: 5.1, w: 5.6, h: 0.8, fontSize: 13, color: TX });
  const cl = [["p_cloud_pca", "PCA"], ["p_cloud_ae", "AE"], ["p_cloud_vae", "VAE"]];
  cl.forEach((c, i) => { const x = 6.7 + i * 2.05; fit(s, c[0], x, 2.2, 1.95, 2.4); tbox(s, c[1], { x, y: 1.95, w: 1.95, h: 0.3, fontSize: 13, bold: true, color: BLUE, align: "center" }); });
  tbox(s, "黑=真實 · 彩=生成:生成雲應蓋住真實,不外溢不塌成一點。", { x: 6.7, y: 4.8, w: 6, h: 0.6, fontSize: 12, color: MUT });
  fit(s, "p_gen_plaus", 6.7, 5.3, 2.9, 1.9);
  tbox(s, "plausibility(越低越像真實)", { x: 9.6, y: 5.6, w: 3, h: 1.2, fontSize: 12, color: MUT });
  pageNum(s, pg); s.addNotes("VAE 生成最真;AE 無先驗採樣落洞最差。");
}
// 案例:生成式都市設計
caseSlide("generative",
  "生成式都市設計:從草圖到即時真實渲染",
  "研究:Deep Image of the City (MIT Media Lab);StreetGAN / DeepStreet 街道網路生成。　示意圖:都市天際線,CC BY-SA · Wikimedia Commons。");
// 權力
{
  const s = contentSlide(); pg++; kicker(s, "5.6 · 權力重新分配"); title(s, "每種體制把城市推向哪裡(從指標到真實形態)");
  // 左:真實 3D 量體對照(現狀 vs 開發商主導)
  pfit(s, "out/city/regime_3d_pair.png", CITY.regime_3d_pair, 0.5, 2.0, 4.1, 4.4, "F4F7FB");
  tbox(s, "現狀", { x: 0.6, y: 2.05, w: 1.5, h: 0.3, fontSize: 12, bold: true, color: "1E3A5F" });
  tbox(s, "開發商主導", { x: 0.6, y: 4.25, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: CORAL });
  tbox(s, "同一城市 3D 量體:開發商主導 → 拔高、加密(曹楊)", { x: 0.4, y: 6.55, w: 4.4, h: 0.35, fontSize: 11, color: MUT, align: "center" });
  // 中:形態簽名熱圖(量化)
  tbox(s, "形態簽名:baseline→體制 的 Δ 特徵", { x: 4.9, y: 1.7, w: 7.8, h: 0.3, fontSize: 12, color: MUT });
  fit(s, "p_signature", 4.9, 2.0, 7.9, 2.5);
  // 右下:四體制卡(2×2)
  const sig = [["開發商", "拔高塔化 h_mean↑", CORAL], ["政府", "向公共重心收攏 concentration↑", BLUE], ["居民自建", "細粒化 grain↓、棟數↑", TEAL], ["共享", "釋放開放地面 coverage↓", GOLD]];
  sig.forEach((g, i) => { const x = 4.9 + (i % 2) * 4.05, y = 4.7 + Math.floor(i / 2) * 1.2; card(s, x, y, 3.85, 1.05); tbox(s, g[0], { x: x + 0.2, y: y + 0.12, w: 3.5, h: 0.35, fontSize: 14, bold: true, color: g[2] }); tbox(s, g[1], { x: x + 0.2, y: y + 0.5, w: 3.5, h: 0.45, fontSize: 12, color: TX }); });
  pageNum(s, pg); s.addNotes("左=真實城市形態、中=量化簽名。權力重寫城市=可量化的位移。");
}
// 光譜(native)
{
  const s = contentSlide(); pg++; kicker(s, "6 · 綜合比較"); title(s, "方法光譜:哪個工具解哪種規劃問題");
  const N = 60;
  for (let i = 0; i < N; i++) { const t = i / (N - 1); const col = mix(["4EC9B0", "F2C14E", "F96167"], t); s.addShape(P.shapes.RECTANGLE, { x: 0.9 + i * (11.5 / N), y: 3.3, w: 11.5 / N + 0.02, h: 0.7, fill: { color: col }, line: { type: "none" } }); }
  tbox(s, "可控 · 透明", { x: 0.9, y: 2.85, w: 3, h: 0.35, fontSize: 14, bold: true, color: TEAL });
  tbox(s, "強大 · 能生成", { x: 9.4, y: 2.85, w: 3, h: 0.35, fontSize: 14, bold: true, color: CORAL, align: "right" });
  const meth = [["Lookup", 1.3], ["PCA", 3.3], ["t-SNE/Isomap", 5.5], ["AE", 7.8], ["VAE", 9.9], ["Conv-VAE", 11.7]];
  meth.forEach(m => { s.addShape(P.shapes.OVAL, { x: m[1] - 0.1, y: 3.55, w: 0.2, h: 0.2, fill: { color: DARK }, line: { color: TX, width: 1 } }); tbox(s, m[0], { x: m[1] - 0.9, y: 4.15, w: 1.8, h: 0.5, fontSize: 12, bold: true, color: TX, align: "center" }); });
  card(s, 0.9, 5.3, 11.5, 1.2);
  tbox(s, "沒有最好的方法,只有對應規劃問題的方法", { x: 0.9, y: 5.4, w: 11.5, h: 0.4, fontSize: 16, bold: true, color: TX, align: "center" });
  tbox(s, "要可控→規則 Lookup   ·   要看結構→PCA/Isomap   ·   要生成/做方案→VAE   ·   要空間紋理→Conv-VAE", { x: 0.9, y: 5.9, w: 11.5, h: 0.5, fontSize: 13, color: MUT, align: "center" });
  pageNum(s, pg); s.addNotes("越左越可控,越右越能生成。");
}
// 結尾
{
  const s = P.addSlide(); s.background = { color: DARK }; pg++;
  tbox(s, "啟發 & 誠實邊界", { x: 0.9, y: 0.65, w: 11, h: 0.8, fontSize: 32, bold: true, color: TX });
  const insp = [["1", "把「權力→形態」從描述升級成可量化、可比較、可生成的空間"], ["2", "資料驅動群聚能檢驗人訂規則(ARI 0.70),甚至提出沒想到的族群"], ["3", "VAE 連續潛空間 = 情景生成器雛形:規劃師出概念,NN 出候選"]];
  insp.forEach((it, i) => { const y = 1.7 + i * 1.05; chip(s, 0.9, y, it[0], TEAL); tbox(s, it[1], { x: 1.55, y: y - 0.05, w: 7.1, h: 0.7, fontSize: 15, color: TX, valign: "middle" }); });
  card(s, 9.0, 1.7, 3.6, 3.15, CARD);
  tbox(s, "誠實邊界", { x: 9.25, y: 1.85, w: 3.2, h: 0.4, fontSize: 15, bold: true, color: CORAL });
  s.addText([
    { text: "樣本小 → AE/VAE/CNN 皆教學演示,非嚴謹預測", options: { color: TX, breakLine: true } },
    { text: "體制簽名是算子設定的必然結果(教學語言)", options: { color: TX, breakLine: true } },
    { text: "NN 是放大鏡不是水晶球:價值判斷仍是規劃師的責任", options: { color: TX } },
  ], { x: 9.25, y: 2.3, w: 3.2, h: 2.4, fontFace: FF, fontSize: 12.5, valign: "top", paraSpaceAfter: 9 });
  tbox(s, "NN 讓 05 資料能做更深的分析 —— 而規劃專業,始終是主角。", { x: 0.9, y: 5.5, w: 11.6, h: 0.7, fontSize: 18, bold: true, italic: true, color: TX, align: "center" });
  pageNum(s, pg); s.addNotes("務必說誠實邊界。");
}

// 顏色漸層混合(3 stop)
function mix(stops, t) {
  const seg = t < 0.5 ? 0 : 1; const lt = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const a = stops[seg], b = stops[seg + 1];
  const h2 = (s) => [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
  const ca = h2(a), cb = h2(b);
  const r = Math.round(ca[0] + (cb[0] - ca[0]) * lt), g = Math.round(ca[1] + (cb[1] - ca[1]) * lt), bl = Math.round(ca[2] + (cb[2] - ca[2]) * lt);
  return [r, g, bl].map(v => v.toString(16).padStart(2, "0")).join("");
}

P.writeFile({ fileName: "潛空間教學簡報.pptx" }).then(f => console.log("WROTE", f));
