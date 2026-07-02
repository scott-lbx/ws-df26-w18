// build_slides.js —— 教學簡報:用 NN 延伸「權力→形態」分析
// 執行: node build_slides.js   (在 07_Latent_Urban/ 下)
const pptx = require("pptxgenjs");
const P = new pptx();
P.layout = "LAYOUT_WIDE";              // 13.3 x 7.5 吋
P.author = "07 Latent Urban";
P.title = "用神經網路延伸權力→形態分析";

// ---- 調色盤(城市/權力主題:深海軍藍 + 青 + 珊瑚紅重點) ----
const NAVYBG = "12233A", NAVY = "1E3A5F", TEAL = "1C7293", CORAL = "F96167";
const ICE = "CADCFC", INK = "1E293B", MUTE = "64748B", WHITE = "FFFFFF", LIGHT = "EEF3F9";
const FF = "Microsoft JhengHei";
const W = 13.33, H = 7.5;

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.14 });

function img(s, path, x, y, w, h) {
  s.addImage({ path, x, y, w, h, sizing: { type: "contain", w, h } });
}
// 白卡承載圖(避免透明背景在深色頁看不清)
function figCard(s, path, x, y, w, h, pad = 0.12) {
  s.addShape(P.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: WHITE }, rectRadius: 0.06, shadow: shadow() });
  img(s, path, x + pad, y + pad, w - 2 * pad, h - 2 * pad);
}
function pageNum(s, n) {
  s.addText(String(n), { x: W - 0.8, y: H - 0.5, w: 0.5, h: 0.3, fontFace: FF, fontSize: 10, color: MUTE, align: "right" });
}
function kicker(s, txt, color = TEAL) {
  s.addText(txt, { x: 0.7, y: 0.42, w: 10, h: 0.35, fontFace: FF, fontSize: 13, bold: true, color, charSpacing: 2 });
}
function title(s, txt, color = NAVY) {
  s.addText(txt, { x: 0.7, y: 0.75, w: 12, h: 0.75, fontFace: FF, fontSize: 30, bold: true, color, margin: 0 });
}

let pg = 0;
// ============ 封面 ============
{
  const s = P.addSlide(); s.background = { color: NAVYBG };
  s.addShape(P.shapes.OVAL, { x: 10.6, y: -1.6, w: 4.6, h: 4.6, fill: { color: TEAL, transparency: 78 } });
  s.addShape(P.shapes.OVAL, { x: 11.6, y: 4.6, w: 3.6, h: 3.6, fill: { color: CORAL, transparency: 82 } });
  s.addText("城市規劃 × 神經網路", { x: 0.9, y: 1.5, w: 11, h: 0.5, fontFace: FF, fontSize: 16, bold: true, color: ICE, charSpacing: 3 });
  s.addText("用神經網路延伸「權力 → 形態」分析", { x: 0.9, y: 2.15, w: 11.5, h: 1.4, fontFace: FF, fontSize: 42, bold: true, color: WHITE });
  s.addText("把城市形態壓進可壓縮、可比較、可生成的潛空間", { x: 0.9, y: 3.7, w: 11, h: 0.6, fontFace: FF, fontSize: 19, color: ICE });
  s.addText([
    { text: "對象:", options: { bold: true, color: CORAL } },
    { text: "已學過 05 Shanghai Power to Form、Python 還不熟的規劃學生    ", options: { color: ICE } },
    { text: "目的:", options: { bold: true, color: CORAL } },
    { text: "重觀念、重直覺、重圖解", options: { color: ICE } },
  ], { x: 0.9, y: 5.7, w: 11.5, h: 0.6, fontFace: FF, fontSize: 14 });
  s.addNotes("這是概念與啟發課,不帶大家敲程式。目標:理解怎麼把 NN 接進你們已經會的都市分析流程。");
}
// ============ §0 動機 ============
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "0 · 開場"); title(s, "規劃師為什麼要碰神經網路?");
  figCard(s, "out/figs/concept/c0_motivation.png", 0.7, 1.7, 8.6, 4.9);
  const cards = [
    ["問題", "9 指標 × 幾千棟樓 × 十幾街區 × 5 體制,人腦一次看不完", CORAL],
    ["NN 的角色", "把高維形態壓成一張看得懂的圖,不是取代規劃判斷", TEAL],
    ["主角仍是你", "NN 給地圖,規劃師讀「這代表什麼權力邏輯」", NAVY],
  ];
  cards.forEach((c, i) => {
    const y = 1.8 + i * 1.6;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: 9.6, y, w: 3.1, h: 1.4, fill: { color: LIGHT }, rectRadius: 0.06, shadow: shadow() });
    s.addText(c[0], { x: 9.8, y: y + 0.12, w: 2.8, h: 0.35, fontFace: FF, fontSize: 14, bold: true, color: c[2], margin: 0 });
    s.addText(c[1], { x: 9.8, y: y + 0.5, w: 2.75, h: 0.85, fontFace: FF, fontSize: 11.5, color: INK, margin: 0 });
  });
  pageNum(s, pg);
  s.addNotes("NN 是放大鏡不是水晶球。左圖手動看9張分佈眼花,右圖壓成一張2D一眼看出族群。");
}
// ============ §1 NN 介入點 ============
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "1 · 從 05 流程出發"); title(s, "NN 在哪裡介入你的工作流程?");
  figCard(s, "out/figs/concept/c1_workflow.png", 0.7, 1.75, 11.9, 3.6);
  s.addText([
    { text: "不改資料來源", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "NN 讀的就是 05 拼接好的 buildings.parquet 與 9 個形態指標(measure.diagnose)。", options: { color: INK } },
  ], { x: 0.9, y: 5.6, w: 6.0, h: 1.2, fontFace: FF, fontSize: 14, valign: "top" });
  s.addText([
    { text: "三種資料粒度、三種切入", options: { bold: true, color: TEAL, breakLine: true } },
    { text: "每棟樓→向量 · 整城→高度影像 · 每城×體制→9 維形態", options: { color: INK } },
  ], { x: 7.0, y: 5.6, w: 5.6, h: 1.2, fontFace: FF, fontSize: 14, valign: "top" });
  pageNum(s, pg);
  s.addNotes("NN 接在 05 輸出後面,不動 05 一行。");
}
// ============ §2 三代方法 ============
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "2 · 方法演進"); title(s, "三代方法:查表 → 非監督 ML → 神經網路");
  figCard(s, "out/figs/concept/c2_three_generations.png", 0.7, 1.7, 8.5, 4.5);
  const rows = [
    ["規則 Lookup", "人寫死規則(如分區規範)", "透明可控,但不會自己發現結構", NAVY],
    ["非監督 ML", "數學找資料主結構", "找族群,但多半不能生成", TEAL],
    ["神經網路", "網路學壓縮與重建", "能壓、能重建、能生成", CORAL],
  ];
  rows.forEach((r, i) => {
    const y = 1.8 + i * 1.55;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: 9.5, y, w: 3.25, h: 1.4, fill: { color: LIGHT }, rectRadius: 0.06, shadow: shadow() });
    s.addText(r[0], { x: 9.7, y: y + 0.1, w: 2.9, h: 0.35, fontFace: FF, fontSize: 14, bold: true, color: r[3], margin: 0 });
    s.addText([{ text: r[1], options: { breakLine: true, color: INK } }, { text: r[2], options: { color: MUTE } }],
      { x: 9.7, y: y + 0.48, w: 2.9, h: 0.9, fontFace: FF, fontSize: 10.5, margin: 0 });
  });
  s.addText("關鍵差異:神經網路會「生成」——造一個沒看過但合理的城市;前兩代不會。",
    { x: 0.7, y: 6.35, w: 11.9, h: 0.5, fontFace: FF, fontSize: 14, bold: true, italic: true, color: CORAL, align: "center" });
  pageNum(s, pg);
  s.addNotes("誠實補充:AE/VAE 嚴格是自監督(答案=自己),對學生只要記:它們會生成,前兩代不會。");
}
// ============ Section:非監督 ML ============
function sectionDivider(no, zh, en, sub) {
  const s = P.addSlide(); s.background = { color: NAVYBG };
  s.addShape(P.shapes.OVAL, { x: 0.9, y: 2.7, w: 1.5, h: 1.5, fill: { color: TEAL } });
  s.addText(no, { x: 0.9, y: 2.7, w: 1.5, h: 1.5, fontFace: FF, fontSize: 44, bold: true, color: WHITE, align: "center", valign: "middle" });
  s.addText(zh, { x: 2.8, y: 2.85, w: 9.8, h: 0.9, fontFace: FF, fontSize: 40, bold: true, color: WHITE });
  s.addText(en, { x: 2.85, y: 3.8, w: 9.8, h: 0.5, fontFace: FF, fontSize: 16, color: ICE, charSpacing: 2 });
  if (sub) s.addText(sub, { x: 2.85, y: 4.35, w: 9.8, h: 0.5, fontFace: FF, fontSize: 14, color: MUTE });
  return s;
}
sectionDivider("3", "非監督 ML 的原理", "PCA · t-SNE · ISOMAP", "共同任務:把高維壓到 2D;差別在「保留什麼」");
// PCA
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "3.1 · PCA"); title(s, "PCA:找城市變化的「主軸」");
  figCard(s, "out/figs/concept/c3_pca_axes.png", 0.7, 1.7, 7.0, 4.9);
  const pts = [
    ["直覺", "問「這團城市點雲最長的方向是哪條」——那就是 PC1", TEAL],
    ["可解讀", "PC1 = 強度/資本軸(負=高聳密集,正=瘦長細碎)", NAVY],
    ["規劃意涵", "PC1 佔 47% 方差 → 城市差異近半能用單一強度軸解釋", CORAL],
  ];
  pts.forEach((c, i) => {
    const y = 1.9 + i * 1.55;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: 8.0, y, w: 4.6, h: 1.35, fill: { color: LIGHT }, rectRadius: 0.06, shadow: shadow() });
    s.addText(c[0], { x: 8.2, y: y + 0.12, w: 4.2, h: 0.35, fontFace: FF, fontSize: 14, bold: true, color: c[2], margin: 0 });
    s.addText(c[1], { x: 8.2, y: y + 0.5, w: 4.25, h: 0.8, fontFace: FF, fontSize: 12.5, color: INK, margin: 0 });
  });
  pageNum(s, pg); s.addNotes("PCA 是唯一有可解讀軸、又能逆回原空間的統計法。");
}
// t-SNE
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "3.2 · t-SNE"); title(s, "t-SNE:只在乎「誰跟誰是鄰居」");
  figCard(s, "out/figs/concept/c4_tsne_neighbors.png", 0.7, 1.7, 8.2, 4.6);
  s.addText([
    { text: "像把同學按「常一起玩」分小圈圈:", options: { breakLine: true, color: INK } },
    { text: "圈內對,但圈跟圈之間的距離沒意義。", options: { color: CORAL, bold: true } },
  ], { x: 9.2, y: 2.1, w: 3.5, h: 1.4, fontFace: FF, fontSize: 14 });
  s.addText([
    { text: "適合:", options: { bold: true, color: TEAL } }, { text: "看聚類  ", options: { color: INK } },
    { text: "小心:", options: { bold: true, color: CORAL } }, { text: "別讀團間距離、參數敏感不穩定", options: { color: INK } },
  ], { x: 9.2, y: 4.0, w: 3.5, h: 1.8, fontFace: FF, fontSize: 13 });
  pageNum(s, pg); s.addNotes("perplexity 換一下圖就變。只看聚類、不解讀距離。");
}
// Isomap
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "3.3 · Isomap"); title(s, "Isomap:把摺起來的紙攤平");
  figCard(s, "out/figs/concept/c5_isomap_swissroll.png", 0.7, 1.7, 8.6, 4.6);
  s.addText([
    { text: "資料像捲起來的紙,", options: { breakLine: true, color: INK } },
    { text: "直線距離會「穿牆」。", options: { breakLine: true, color: CORAL, bold: true } },
    { text: "Isomap 沿紙面(測地線)量距離再攤平,保全局結構。", options: { color: INK } },
  ], { x: 9.5, y: 2.4, w: 3.2, h: 2.6, fontFace: FF, fontSize: 13.5 });
  pageNum(s, pg); s.addNotes("和 t-SNE 一樣沒有可解讀軸、不能逆回城市。");
}
// 三方法比較 + 真實圖
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "3 · 小結"); title(s, "同一份語料,三種投影的結構");
  figCard(s, "out/figs/01_method_compare.png", 0.7, 1.7, 8.7, 4.5);
  const tbl = [
    [{ text: "", options: { fill: { color: NAVY } } }, { text: "全局", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "center" } }, { text: "可解讀軸", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "center" } }, { text: "能生成", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "center" } }],
    ["PCA", "✔", "✔", "✔"],
    ["t-SNE", "✘", "✘", "✘"],
    ["Isomap", "✔", "✘", "✘"],
  ];
  s.addTable(tbl, { x: 9.5, y: 2.0, w: 3.2, colW: [1.1, 0.7, 0.7, 0.7], rowH: 0.5, fontFace: FF, fontSize: 13, align: "center", valign: "middle", border: { pt: 0.5, color: "D6DEE8" }, color: INK });
  s.addText("PCA 能說「PC1 是什麼」;t-SNE/Isomap 只給座標、回不去。",
    { x: 9.5, y: 4.5, w: 3.2, h: 1.2, fontFace: FF, fontSize: 12.5, color: MUTE, italic: true });
  pageNum(s, pg); s.addNotes("顏色=家族,形狀=體制。");
}
// ============ Section:神經網路 ============
sectionDivider("4", "神經網路的原理", "AE · VAE · CNN", "以 AE、VAE 為主");
// AE
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "4.1 · AE 自編碼器"); title(s, "AE:學徒「臨摹 → 只憑幾個數字默寫」");
  figCard(s, "out/figs/concept/c6_ae_hourglass.png", 0.7, 1.75, 9.0, 4.6);
  s.addText([
    { text: "瓶頸 = 潛碼", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "壓到最窄還能畫回來,代表那幾個數字抓住了城市形態的精華。", options: { color: INK } },
  ], { x: 9.9, y: 2.3, w: 2.9, h: 2.0, fontFace: FF, fontSize: 13.5 });
  s.addText("非線性壓縮 → 同 2 維,重建遠比 PCA 準(見結果)。",
    { x: 9.9, y: 4.6, w: 2.9, h: 1.3, fontFace: FF, fontSize: 13, color: TEAL, bold: true });
  pageNum(s, pg); s.addNotes("編碼器→潛碼→解碼器。潛碼=城市形態DNA。");
}
// VAE
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "4.2 · VAE 變分自編碼器"); title(s, "VAE:學會「風格」才能創作");
  figCard(s, "out/figs/concept/c7_vae_continuity.png", 0.7, 1.75, 9.0, 4.6);
  s.addText([
    { text: "把潛空間規範成連續高斯雲 N(0,I)", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "→ 隨手在雲裡點一下,就解碼出沒看過但合理的城市。這就是生成。", options: { color: INK } },
  ], { x: 9.9, y: 2.3, w: 2.9, h: 2.4, fontFace: FF, fontSize: 13 });
  s.addText("代價:重建略遜 AE(多了規範化)。",
    { x: 9.9, y: 5.0, w: 2.9, h: 0.8, fontFace: FF, fontSize: 12.5, color: MUTE });
  pageNum(s, pg); s.addNotes("AE 潛空間有洞、踩進去解亂碼;VAE 填滿、處處可採樣。");
}
// CNN
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "4.3 · CNN(Conv-VAE)"); title(s, "CNN:看空照圖的「紋理」");
  // 迷你流程圖:影像 → 卷積 → 潛碼 → 生成
  const flow = [["整城\n高度影像", TEAL], ["卷積\n讀紋理", NAVY], ["潛碼", CORAL], ["生成\n新紋理", TEAL]];
  flow.forEach((f, i) => {
    const x = 0.9 + i * 3.0;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x, y: 2.5, w: 2.4, h: 1.5, fill: { color: f[1] }, rectRadius: 0.08, shadow: shadow() });
    s.addText(f[0], { x, y: 2.5, w: 2.4, h: 1.5, fontFace: FF, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
    if (i < 3) s.addText("→", { x: x + 2.4, y: 2.5, w: 0.6, h: 1.5, fontFace: FF, fontSize: 26, color: MUTE, align: "center", valign: "middle" });
  });
  s.addText([
    { text: "前面把每棟樓當一列數字,丟了空間排列。", options: { breakLine: true, color: INK } },
    { text: "把整座城燒成高度影像,用卷積像看空照圖一樣讀街廓紋理、疏密。", options: { color: INK } },
  ], { x: 0.9, y: 4.5, w: 11.5, h: 1.1, fontFace: FF, fontSize: 15 });
  s.addText("誠實邊界:高度圖 86% 空白、樣本少 → 本例僅示意,潛碼部分塌縮。",
    { x: 0.9, y: 5.9, w: 11.5, h: 0.5, fontFace: FF, fontSize: 12.5, color: CORAL, italic: true });
  pageNum(s, pg); s.addNotes("換表示(表格→影像)、換架構(MLP→CNN),模型看到的東西完全不同。");
}
// ============ Section:訓練結果 ============
sectionDivider("5", "訓練結果", "REAL DATA · 60 樣本 × 9 特徵", "直接讀圖");
// 重建
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "5.1 · 重建"); title(s, "AE 完勝:2 維就贏 PCA 4 維");
  figCard(s, "out/figs/03_recon_compare.png", 0.7, 1.7, 8.4, 4.7);
  const stats = [["0.061", "AE 壓到 2 維", CORAL], ["0.104", "PCA 壓到 4 維", MUTE], ["0.033", "AE 壓到 3 維", TEAL]];
  stats.forEach((st, i) => {
    const y = 1.85 + i * 1.55;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: 9.4, y, w: 3.3, h: 1.35, fill: { color: LIGHT }, rectRadius: 0.06, shadow: shadow() });
    s.addText(st[0], { x: 9.55, y: y + 0.12, w: 2.0, h: 0.9, fontFace: FF, fontSize: 34, bold: true, color: st[2], margin: 0, valign: "middle" });
    s.addText([{ text: "重建 MSE", options: { breakLine: true, color: MUTE } }, { text: st[1], options: { color: INK, bold: true } }],
      { x: 11.2, y: y + 0.2, w: 1.4, h: 1.0, fontFace: FF, fontSize: 11, margin: 0, valign: "middle" });
  });
  s.addText("越低越好 · 非線性瓶頸更省維", { x: 9.4, y: 6.5, w: 3.3, h: 0.3, fontFace: FF, fontSize: 11, color: MUTE, italic: true });
  pageNum(s, pg); s.addNotes("這是「NN 比統計法強在哪」最有感的一格。");
}
// 訓練損失
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "5.2 · 訓練"); title(s, "AE/VAE 怎麼訓練出來的");
  figCard(s, "out/figs/02_ae_vae_loss.png", 0.7, 1.7, 12.0, 3.7);
  const c = [["(a) 過擬合", "沒正則的 AE 會背答案(驗證變差)", NAVY], ["(b) 損失拆解", "VAE 重建與 KL 此消彼長", TEAL], ["(c) 塌縮", "β 太大壓垮 KL → 潛碼失效;退火避免", CORAL]];
  c.forEach((k, i) => {
    const x = 0.7 + i * 4.05;
    s.addText([{ text: k[0], options: { breakLine: true, bold: true, color: k[2] } }, { text: k[1], options: { color: INK } }],
      { x, y: 5.6, w: 3.9, h: 1.2, fontFace: FF, fontSize: 13, valign: "top" });
  });
  pageNum(s, pg); s.addNotes("訓練不是按一下,是調正則、調 β。");
}
// 群聚 vs regimes
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "5.3 · 群聚(最關鍵)"); title(s, "資料自己會分出「權力體制」嗎?");
  figCard(s, "out/figs/08_cluster_vs_regime.png", 0.7, 1.7, 8.4, 4.6);
  const tbl = [
    [{ text: "潛空間", options: { fill: { color: NAVY }, color: WHITE, bold: true } }, { text: "ARI", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "center" } }, { text: "silhouette", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "center" } }],
    ["原始 9 維", { text: "0.70", options: { bold: true, color: CORAL, align: "center" } }, "0.32"],
    ["VAE-2", "0.56", { text: "0.58", options: { bold: true, color: TEAL, align: "center" } }],
    ["PCA-2", "0.55", "0.49"],
    ["AE-2", "0.34", "0.55"],
  ];
  s.addTable(tbl, { x: 9.3, y: 1.95, w: 3.4, colW: [1.5, 0.95, 0.95], rowH: 0.42, fontFace: FF, fontSize: 12.5, valign: "middle", align: "center", border: { pt: 0.5, color: "D6DEE8" }, color: INK });
  s.addText("不給標籤,KMeans 只看形態就大致復原 05 手訂的 5 種體制(ARI 0.70)→ 這些 regime 有形態資料支撐。",
    { x: 9.3, y: 4.5, w: 3.4, h: 1.8, fontFace: FF, fontSize: 12.5, color: INK });
  pageNum(s, pg); s.addNotes("這就是資料驅動有別於手動 lookup 的價值:客觀檢驗規則,甚至發現新族群。");
}
// 誰在成簇
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "5.4 · 解釋性"); title(s, "誰在成簇:體制,不是城市家族");
  figCard(s, "out/figs/04_cluster_interpret.png", 0.7, 1.7, 9.2, 4.6);
  s.addText([
    { text: "silhouette 按體制 ≫ 按家族", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "(0.24–0.42 vs ≈0)", options: { color: MUTE, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "同一套權力算子把不同城推向一致方向;城市原本的類型反而被打散。", options: { color: INK, breakLine: true } },
    { text: "權力 > 地方性。", options: { bold: true, color: NAVY } },
  ], { x: 10.1, y: 2.2, w: 2.7, h: 3.5, fontFace: FF, fontSize: 13 });
  pageNum(s, pg); s.addNotes("PC1 載荷=強度/資本 vs 細粒自建。");
}
// 生成能力
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "5.5 · 生成"); title(s, "只有 VAE 真的會「造城」");
  figCard(s, "out/figs/06_gen_capability.png", 0.7, 1.7, 6.3, 4.7);
  figCard(s, "out/figs/07_gen_cloud.png", 7.1, 1.7, 5.5, 3.3);
  s.addText([
    { text: "能生成 = 有沒有 decoder。", options: { bold: true, color: CORAL, breakLine: true } },
    { text: "t-SNE/Isomap 只會投影、回不去 → 不能生成。VAE 有先驗、生成最真(1.03)、100% 守物理約束、插值最連續。", options: { color: INK } },
  ], { x: 7.1, y: 5.2, w: 5.6, h: 1.6, fontFace: FF, fontSize: 13, valign: "top" });
  pageNum(s, pg); s.addNotes("AE 無先驗、採樣落洞(2.11)最差。");
}
// 權力重分配
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "5.6 · 權力重新分配"); title(s, "每種體制把城市推向哪裡");
  figCard(s, "out/figs/05_regime_signature.png", 0.7, 1.7, 8.6, 4.6);
  const sig = [["開發商", "拔高塔化 h_mean↑", CORAL], ["政府", "向公共重心收攏 concentration↑", TEAL], ["居民自建", "細粒化 grain↓、棟數↑", NAVY], ["共享", "釋放開放地面 coverage↓", MUTE]];
  sig.forEach((g, i) => {
    const y = 1.85 + i * 1.15;
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: 9.5, y, w: 3.2, h: 1.0, fill: { color: LIGHT }, rectRadius: 0.06, shadow: shadow() });
    s.addText(g[0], { x: 9.65, y: y + 0.1, w: 3.0, h: 0.35, fontFace: FF, fontSize: 14, bold: true, color: g[2], margin: 0 });
    s.addText(g[1], { x: 9.65, y: y + 0.48, w: 3.0, h: 0.45, fontFace: FF, fontSize: 11.5, color: INK, margin: 0 });
  });
  pageNum(s, pg); s.addNotes("「權力重寫城市」不再只是文字,而是可量化、可比較的位移。");
}
// ============ §6 光譜 ============
{
  const s = P.addSlide(); s.background = { color: WHITE }; pg++;
  kicker(s, "6 · 綜合比較"); title(s, "方法光譜:哪個工具解哪種規劃問題");
  figCard(s, "out/figs/concept/c8_method_spectrum.png", 0.7, 1.7, 12.0, 3.4);
  s.addText([
    { text: "沒有最好的方法,只有對應規劃問題的方法:", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "要可控 → 規則 Lookup    ·    要看結構 → PCA/UMAP/Isomap    ·    要生成/做方案 → VAE    ·    要空間紋理 → Conv-VAE", options: { color: INK } },
  ], { x: 0.9, y: 5.4, w: 11.5, h: 1.4, fontFace: FF, fontSize: 15, align: "center" });
  pageNum(s, pg); s.addNotes("越左越可控可審,越右越能壓縮·重建·生成。");
}
// ============ §7 啟發 + 邊界(深色結尾) ============
{
  const s = P.addSlide(); s.background = { color: NAVYBG }; pg++;
  s.addText("啟發 & 誠實邊界", { x: 0.9, y: 0.6, w: 11, h: 0.8, fontFace: FF, fontSize: 32, bold: true, color: WHITE });
  const insp = [["1", "把「權力→形態」從描述升級成可量化、可比較、可生成的空間"], ["2", "資料驅動群聚能檢驗人訂規則(ARI 0.70),甚至提出沒想到的族群"], ["3", "VAE 連續潛空間 = 情景生成器雛形:規劃師出概念,NN 出候選"]];
  insp.forEach((it, i) => {
    const y = 1.6 + i * 1.05;
    s.addShape(P.shapes.OVAL, { x: 0.9, y, w: 0.6, h: 0.6, fill: { color: TEAL } });
    s.addText(it[0], { x: 0.9, y, w: 0.6, h: 0.6, fontFace: FF, fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle" });
    s.addText(it[1], { x: 1.7, y: y + 0.02, w: 6.9, h: 0.9, fontFace: FF, fontSize: 15, color: ICE, valign: "middle" });
  });
  s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: 8.9, y: 1.6, w: 3.7, h: 3.2, fill: { color: "1B3350" }, rectRadius: 0.06 });
  s.addText("誠實邊界", { x: 9.1, y: 1.75, w: 3.3, h: 0.4, fontFace: FF, fontSize: 15, bold: true, color: CORAL });
  s.addText([
    { text: "樣本小 → AE/VAE/CNN 皆教學演示,非嚴謹預測", options: { breakLine: true, color: ICE } },
    { text: "體制簽名是算子設定的必然結果(教學語言)", options: { breakLine: true, color: ICE } },
    { text: "NN 是放大鏡不是水晶球:價值判斷仍是規劃師的責任", options: { color: ICE } },
  ], { x: 9.1, y: 2.2, w: 3.35, h: 2.5, fontFace: FF, fontSize: 12, valign: "top", paraSpaceAfter: 8 });
  s.addText("NN 讓 05 資料能做更深的分析 —— 而規劃專業,始終是主角。",
    { x: 0.9, y: 5.4, w: 11.5, h: 0.7, fontFace: FF, fontSize: 18, bold: true, italic: true, color: WHITE, align: "center" });
  pageNum(s, pg); s.addNotes("務必對學生說誠實邊界。");
}

P.writeFile({ fileName: "潛空間教學簡報.pptx" }).then(f => console.log("WROTE", f));
