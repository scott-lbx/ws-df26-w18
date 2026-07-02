// build_workshop.js —— 工作營全本(31頁,深底 數據/終端 學術密集風)
// node build_workshop.js → 工作營_潛空間.pptx
const pptx = require("pptxgenjs");
const P = new pptx(); P.layout = "LAYOUT_WIDE"; P.title = "URBAN LATENT · workshop";
const W = 13.33, H = 7.5;
const BG = "0A0F1A", PANEL = "0E1727", PANEL2 = "12202F", LINE = "223145", DIM = "3D4B60";
const TX = "E6EDF3", MUT = "8A97A8", GRN = "4EC9B0", CYAN = "5AB0FF", AMB = "F2C14E", MAG = "C586C0", RED = "F96167";
const MONO = "Consolas", CJK = "Microsoft JhengHei";

const AR = { p_bldg_pca:1.053,p_bldg_tsne:1.053,p_bldg_isomap:1.053,p_recon_bar:1.380,p_recon_feat:1.550,p_confusion:1.260,
 p_ari:1.237,p_sil_cluster:1.231,p_cloud_pca:1.062,p_cloud_ae:1.062,p_cloud_vae:1.062,p_signature:1.903,p_regime_arrows:1.146,
 p_loadings:1.760,p_silhouette:1.446,p_loss_ae:1.368,p_loss_vae:1.260,p_loss_collapse:1.260,p_gen_plaus:1.149,
 m_pca:1.389,m_tsne:1.959,m_isomap:2.373,m_ae:1.888,m_vae:2.083,m_cnn:2.166,m_point:2.171,m_latentgrid:0.991,m_cae_recon:3.184,
 m_thesis:2.649,m_graph:2.332,m_gnn_mp:2.431,regime_3d_pair:1.286,city_3d:0.989 };
const DIR = { p_bldg_pca:"slides",p_bldg_tsne:"slides",p_bldg_isomap:"slides",p_recon_bar:"slides",p_recon_feat:"slides",
 p_confusion:"slides",p_ari:"slides",p_sil_cluster:"slides",p_cloud_pca:"slides",p_cloud_ae:"slides",p_cloud_vae:"slides",
 p_signature:"slides",p_regime_arrows:"slides",p_loadings:"slides",p_silhouette:"slides",p_loss_ae:"slides",p_loss_vae:"slides",
 p_loss_collapse:"slides",p_gen_plaus:"slides",m_pca:"concept_ml",m_tsne:"concept_ml",m_isomap:"concept_ml",m_ae:"concept_ml",
 m_vae:"concept_ml",m_cnn:"concept_ml",m_point:"concept_ml",m_latentgrid:"concept_ml",m_cae_recon:"concept_ml",
 m_thesis:"extra",m_graph:"extra",m_gnn_mp:"extra" };

function t(s,txt,o){ s.addText(txt,Object.assign({fontFace:CJK,color:TX,margin:0},o)); }
function mo(s,txt,o){ s.addText(txt,Object.assign({fontFace:MONO,color:MUT,margin:0},o)); }
function hair(s,x,y,w,c=LINE){ s.addShape(P.shapes.LINE,{x,y,w,h:0,line:{color:c,width:0.75}}); }
function vhair(s,x,y,h,c=LINE){ s.addShape(P.shapes.LINE,{x,y,w:0,h,line:{color:c,width:0.75}}); }
function panel(s,x,y,w,h,hdr,hc){ s.addShape(P.shapes.RECTANGLE,{x,y,w,h,fill:{color:PANEL},line:{color:LINE,width:0.75}});
 if(hdr){ s.addShape(P.shapes.RECTANGLE,{x,y,w,h:0.34,fill:{color:PANEL2},line:{color:LINE,width:0.75}});
  mo(s,hdr,{x:x+0.12,y:y+0.02,w:w-0.24,h:0.3,fontSize:10,color:hc||GRN,charSpacing:1,valign:"middle"}); } }
function fig(s,key,x,y,w,h){ const ar=AR[key]; let iw=w,ih=iw/ar; if(ih>h){ih=h;iw=ih*ar;}
 s.addImage({path:`out/figs/${DIR[key]}/${key}.png`,x:x+(w-iw)/2,y:y+(h-ih)/2,w:iw,h:ih}); }
function pfitJ(s,path,ar,x,y,w,h){ let iw=w,ih=iw/ar; if(ih>h){ih=h;iw=ih*ar;} s.addImage({path,x:x+(w-iw)/2,y:y+(h-ih)/2,w:iw,h:ih}); }
function metric(s,x,y,w,label,val,delta,dc){ mo(s,label,{x,y,w,h:0.28,fontSize:10,color:MUT,charSpacing:1});
 s.addText(val,{x,y:y+0.26,w,h:0.6,fontFace:MONO,fontSize:30,bold:true,color:TX,margin:0});
 if(delta) mo(s,delta,{x,y:y+0.9,w,h:0.28,fontSize:11,color:dc||GRN}); }
let PG=0;
function chrome(sec,run){ const s=P.addSlide(); PG++; s.background={color:BG};
 hair(s,0.5,0.62,W-1,LINE); mo(s,sec,{x:0.5,y:0.28,w:8,h:0.3,fontSize:11,color:GRN,charSpacing:2});
 mo(s,run,{x:W-6.5,y:0.28,w:6,h:0.3,fontSize:11,color:MUT,charSpacing:1,align:"right"});
 hair(s,0.5,H-0.55,W-1,LINE); mo(s,"URBAN-LATENT // 城市潛空間編碼",{x:0.5,y:H-0.48,w:8,h:0.3,fontSize:9.5,color:DIM,charSpacing:1});
 mo(s,String(PG).padStart(2,"0")+" / 31",{x:W-3,y:H-0.48,w:2.5,h:0.3,fontSize:9.5,color:DIM,align:"right"}); return s; }
function head(s,id,title,sub){ s.addText(id,{x:0.5,y:0.95,w:1.7,h:0.9,fontFace:MONO,fontSize:40,bold:true,color:LINE,margin:0});
 t(s,title,{x:2.2,y:1.0,w:10.6,h:0.6,fontSize:25,bold:true,color:TX});
 if(sub) mo(s,sub,{x:2.22,y:1.62,w:10.6,h:0.35,fontSize:12,color:MUT,charSpacing:1}); }
function noteBar(s,y,txt){ panel(s,0.5,y,12.33,6.9-y>1.9?1.9:6.9-y,"[ NOTE ]"); t(s,txt,{x:0.75,y:y+0.45,w:11.8,h:1.3,fontSize:13.5,color:TX,lineSpacingMultiple:1.2,valign:"top"}); }
function divider(sec,id,title,sub){ const s=P.addSlide(); PG++; s.background={color:BG};
 hair(s,0.5,0.62,W-1,LINE); mo(s,sec,{x:0.5,y:0.28,w:8,h:0.3,fontSize:11,color:GRN,charSpacing:2});
 mo(s,String(PG).padStart(2,"0")+" / 31",{x:W-3,y:0.28,w:2.5,h:0.3,fontSize:11,color:MUT,align:"right"});
 s.addText(id,{x:0.6,y:2.6,w:3.2,h:2,fontFace:MONO,fontSize:150,bold:true,color:PANEL2,margin:0});
 t(s,title,{x:3.6,y:3.0,w:9.4,h:1.0,fontSize:44,bold:true,color:TX});
 mo(s,sub,{x:3.65,y:4.15,w:9.4,h:0.5,fontSize:15,color:CYAN,charSpacing:2});
 hair(s,3.65,3.9,6.5,GRN); return s; }
function caseFull(img,idx,ttl,sub,cite){ const s=P.addSlide(); PG++; s.background={color:BG};
 s.addImage({path:`out/case/${img}.jpg`,x:0,y:0,w:W,h:H,sizing:{type:"cover",w:W,h:H}});
 s.addShape(P.shapes.RECTANGLE,{x:0,y:0,w:W,h:0.62,fill:{color:BG,transparency:12}});
 mo(s,`§07 · CASE STUDY ${idx}/04`,{x:0.5,y:0.22,w:8,h:0.3,fontSize:11,color:GRN,charSpacing:2});
 mo(s,String(PG).padStart(2,"0")+" / 31",{x:W-3,y:0.22,w:2.5,h:0.3,fontSize:11,color:MUT,align:"right"});
 s.addShape(P.shapes.RECTANGLE,{x:0,y:5.15,w:W,h:2.35,fill:{color:BG,transparency:10}}); hair(s,0,5.15,W,GRN);
 t(s,ttl,{x:0.6,y:5.42,w:12,h:0.7,fontSize:24,bold:true,color:TX});
 mo(s,sub,{x:0.62,y:6.18,w:12.1,h:0.5,fontSize:13,color:CYAN});
 mo(s,cite,{x:0.62,y:6.95,w:12.1,h:0.35,fontSize:9.5,color:MUT}); }
function pill(s,x,y,txt,c){ const wpx=0.16*txt.length+0.35; s.addShape(P.shapes.RECTANGLE,{x,y,w:wpx,h:0.34,fill:{color:PANEL2},line:{color:c,width:0.75}}); mo(s,txt,{x:x+0.1,y:y+0.02,w:wpx,h:0.3,fontSize:11,color:c,valign:"middle"}); return wpx; }

// ============ 01 COVER ============
{ const s=P.addSlide(); s.background={color:BG}; PG++;
 hair(s,0.5,0.6,W-1,LINE); hair(s,0.5,H-0.6,W-1,LINE);
 mo(s,"WORKSHOP · DEEP LEARNING × URBAN MORPHOLOGY",{x:0.5,y:0.3,w:12,h:0.3,fontSize:11,color:GRN,charSpacing:3});
 mo(s,"2026 / v2",{x:W-3,y:0.3,w:2.5,h:0.3,fontSize:11,color:MUT,align:"right"});
 t(s,"城市潛空間編碼",{x:0.5,y:1.6,w:12,h:1.0,fontSize:52,bold:true,color:TX});
 s.addText("URBAN LATENT ENCODING",{x:0.52,y:2.78,w:12,h:0.5,fontFace:MONO,fontSize:20,color:CYAN,charSpacing:4});
 t(s,"編碼 · 權力區分 · 重新生成",{x:0.5,y:3.42,w:12,h:0.5,fontSize:18,color:MUT});
 panel(s,0.5,4.25,8.0,2.55,"[ ABSTRACT ]");
 t(s,"把每棟建築當一個高維點,用 PCA/t-SNE/Isomap 與 AE/VAE/CNN/GNN 壓進潛空間;\n在潛空間裡量化「權力如何區分城市形態」,再從潛空間採樣重新生成。\n方法對照、訓練診斷、生成能力、與 4 個 DL-urban 案例研究。",
  {x:0.7,y:4.72,w:7.6,h:1.9,fontSize:13.5,color:TX,lineSpacingMultiple:1.25,valign:"top"});
 panel(s,8.7,4.25,4.13,2.55,"[ META ]");
 const meta=[["DATASET","lujiazui · 3263 blds × 16"],["CORPUS","12 cities × 5 = 60 × 9"],["METHODS","PCA·TSNE·ISO·AE·VAE·CNN·GNN"],["KEYWORDS","latent · morphology · generative"]];
 meta.forEach((r,i)=>{ const y=4.78+i*0.49; mo(s,r[0],{x:8.85,y,w:1.5,h:0.3,fontSize:9.5,color:MUT,charSpacing:1}); mo(s,r[1],{x:8.85,y:y+0.19,w:3.9,h:0.3,fontSize:10.5,color:CYAN}); }); }

// ============ 02 INDEX ============
{ const s=chrome("§00 · CONTENTS","目次 / INDEX"); head(s,"00","目次","SECTIONS 01–07 · 31 SLIDES");
 const toc=[["01","資料 → 向量","每棟建築 = 一個高維點",GRN],["02","統計壓縮","PCA · t-SNE · Isomap",CYAN],["03","深度壓縮","AE · VAE · CNN · GNN",AMB],["04","訓練診斷","loss · 過擬合 · collapse",MAG],["05","群聚 × 權力","KMeans vs regimes · 簽名",GRN],["06","生成能力","sampling · plausibility",CYAN],["07","案例研究","embeddings·VAE·GNN·GAN",AMB]];
 toc.forEach((r,i)=>{ const y=2.2+i*0.66; mo(s,r[0],{x:0.7,y,w:0.8,h:0.5,fontSize:22,bold:true,color:r[3],valign:"middle"});
  t(s,r[1],{x:1.7,y,w:4.5,h:0.5,fontSize:19,bold:true,color:TX,valign:"middle"});
  mo(s,r[2],{x:6.3,y,w:6.0,h:0.5,fontSize:12.5,color:MUT,valign:"middle"}); hair(s,0.7,y+0.58,11.9,LINE); }); }

// ============ 03 論點 ============
{ const s=chrome("§00 · THESIS","論點 / framing"); head(s,"","本簡報的論點:城市背後的資訊層","ENCODE → DIFFERENTIATE → RE-GENERATE");
 panel(s,0.5,2.15,12.33,2.55,"[ FRAMEWORK ]"); fig(s,"m_thesis",0.7,2.6,11.9,2.0);
 const cc=[["編碼","把城市壓成潛空間座標(資訊層)",GRN],["權力區分","不同權力體制在潛空間分開",AMB],["重新生成","從潛空間採樣造出新城市",CYAN]];
 cc.forEach((c,i)=>{ const x=0.5+i*4.15; panel(s,x,4.95,3.9,1.75); mo(s,"0"+(i+1),{x:x+0.2,y:5.1,w:1,h:0.4,fontSize:16,bold:true,color:c[2]}); t(s,c[0],{x:x+0.8,y:5.12,w:3,h:0.4,fontSize:16,bold:true,color:c[2]}); t(s,c[1],{x:x+0.2,y:5.6,w:3.55,h:1,fontSize:12.5,color:TX}); }); }

// ============ 04 workflow ============
{ const s=chrome("§01 · PIPELINE","NN 介入 05 流程"); head(s,"01","NN 在哪裡介入你的工作流程","05 OUTPUT → NN: ENCODE / ANALYSE / GENERATE");
 const steps=["多源\n拼接","角色\n映射","高度\n情景","權力\n四體制","3D\n量體"];
 steps.forEach((st,i)=>{ const x=0.6+i*1.62; s.addShape(P.shapes.RECTANGLE,{x,y:2.6,w:1.4,h:1.2,fill:{color:PANEL},line:{color:CYAN,width:0.75}}); t(s,st,{x,y:2.6,w:1.4,h:1.2,fontSize:12,bold:true,color:TX,align:"center",valign:"middle"}); if(i<4) s.addShape(P.shapes.RIGHT_ARROW,{x:x+1.4,y:3.0,w:0.2,h:0.3,fill:{color:MUT}}); });
 mo(s,"05 EXISTING PIPELINE",{x:0.6,y:2.2,w:8,h:0.3,fontSize:11,color:CYAN,charSpacing:1});
 s.addShape(P.shapes.RIGHT_ARROW,{x:8.9,y:3.0,w:0.6,h:0.4,fill:{color:GRN}});
 const nn=[["ENCODE","壓縮→潛空間",GRN],["ANALYSE","群聚結構",AMB],["GENERATE","新方案",CYAN]];
 nn.forEach((n,i)=>{ const y=2.35+i*0.55; panel(s,9.7,y,3.1,0.5); mo(s,n[0],{x:9.85,y:y+0.08,w:1.5,h:0.35,fontSize:11,bold:true,color:n[2]}); t(s,n[1],{x:11.1,y:y+0.05,w:1.6,h:0.4,fontSize:11,color:TX,valign:"middle"}); });
 noteBar(s,4.6,"NN 不改資料來源——讀的就是 05 拼接好的 buildings.parquet 與 9 個形態指標。三種資料粒度:每棟樓→向量 · 整城→高度影像 · 每城×體制→9 維形態。"); }

// ============ 05 三代方法 ============
{ const s=chrome("§01 · METHOD LINEAGE","三代方法"); head(s,"01","查表 → 非監督 ML → 神經網路","RULE-BASED · UNSUPERVISED ML · NEURAL NETS");
 const g=[["01","規則 Lookup","lookup table","人訂規則;透明可控,不會自己發現結構",CYAN],["02","非監督 ML","PCA · t-SNE · Isomap","數學找結構;找族群,多半不能生成",AMB],["03","神經網路","AE · VAE · CNN · GNN","網路學壓縮;能壓、能重建、能生成",GRN]];
 g.forEach((c,i)=>{ const x=0.5+i*4.15; panel(s,x,2.2,3.9,3.6); mo(s,c[0],{x:x+0.2,y:2.4,w:1,h:0.4,fontSize:18,bold:true,color:c[4]}); t(s,c[1],{x:x+0.85,y:2.42,w:3,h:0.4,fontSize:17,bold:true,color:c[4]});
  s.addShape(P.shapes.RECTANGLE,{x:x+0.2,y:3.0,w:3.5,h:0.42,fill:{color:PANEL2},line:{color:c[4],width:0.5}}); mo(s,c[2],{x:x+0.2,y:3.02,w:3.5,h:0.38,fontSize:12,bold:true,color:c[4],align:"center",valign:"middle"});
  hair(s,x+0.2,3.7,3.5,LINE); t(s,c[3],{x:x+0.2,y:3.85,w:3.5,h:1.8,fontSize:13,color:TX}); });
 mo(s,"> 關鍵差異:神經網路會「生成」——造沒看過但合理的城市;前兩代不會。",{x:0.5,y:6.05,w:12,h:0.4,fontSize:13,color:GRN}); }

// ============ 06 §02 divider ============
divider("§02 · UNSUPERVISED ML","02","統計壓縮","PCA · t-SNE · ISOMAP — 把高維壓到 2D,差別在保留什麼");

// ============ 07 建築=點 ============
{ const s=chrome("§02 · VECTORISE","建築 = 一個點"); head(s,"02","先把每一棟建築變成一個點","3263 BUILDINGS → 3263 POINTS IN 16-DIM");
 panel(s,0.5,2.15,12.33,2.7); fig(s,"m_point",0.7,2.45,11.9,2.15);
 const cc=[["為什麼","電腦只懂數字:把高度/面積/用途排成一串,建築=可計算向量",GRN],["這個點","座標=形態身分;兩點靠近=形態相似",AMB],["下一步","幾千點散在 16 維;人眼看不到→降維到 2D",CYAN]];
 cc.forEach((c,i)=>{ const x=0.5+i*4.15; panel(s,x,5.05,3.9,1.65); t(s,c[0],{x:x+0.2,y:5.18,w:3,h:0.4,fontSize:14,bold:true,color:c[2]}); t(s,c[1],{x:x+0.2,y:5.6,w:3.55,h:1,fontSize:12,color:TX}); }); }

// ============ 08-10 PCA/tSNE/Isomap ============
function principle(sec,id,tt,sub,key,cards){ const s=chrome(sec,id.run); head(s,"02",tt,sub);
 panel(s,0.5,2.15,7.6,4.6,id.hdr); fig(s,key,0.7,2.6,7.2,4.0);
 cards.forEach((c,i)=>{ const y=2.3+i*1.5; panel(s,8.3,y,4.5,1.35); t(s,c[0],{x:8.5,y:y+0.13,w:4.1,h:0.35,fontSize:14,bold:true,color:c[2]}); t(s,c[1],{x:8.5,y:y+0.55,w:4.15,h:0.75,fontSize:12.5,color:TX}); }); return s; }
principle("§02 · PCA",{run:"線性投影",hdr:"[ FIG.2.1 ] PCA — 主軸投影"},"PCA:找建築變化的「主軸」","PC1 = MAX-VARIANCE DIRECTION · 47% VAR","m_pca",
 [["直覺","問「這團建築點雲最長的方向」= PC1",GRN],["可解讀","PC1=強度/資本軸(負=高聳密集,正=瘦長細碎)",CYAN],["意涵","近半建築差異能用單一強度軸解釋",AMB]]);
principle("§02 · t-SNE",{run:"局部鄰域",hdr:"[ FIG.2.2 ] t-SNE — 局部簇"},"t-SNE:只在乎「誰跟誰是鄰居」","LOCAL NEIGHBOURHOOD · GLOBAL DISTANCE MEANINGLESS","m_tsne",
 [["直覺","像把同學按常一起玩分小圈圈",GRN],["小心","圈內對,圈間距離無意義",RED],["特性","perplexity 敏感、不穩定;只看聚類",AMB]]);
principle("§02 · ISOMAP",{run:"流形攤平",hdr:"[ FIG.2.3 ] Isomap — 流形"},"Isomap:把摺起來的紙攤平","GEODESIC DISTANCE · PRESERVES GLOBAL STRUCTURE","m_isomap",
 [["直覺","資料像捲起的紙,直線會穿牆",GRN],["做法","沿紙面(測地線)量距離再攤平",CYAN],["特性","保全局;無可解讀軸、不能逆回",AMB]]);

// ============ 11 三方法小結 ============
{ const s=chrome("§02 · SUMMARY","三投影同台 / 3263 blds"); head(s,"02","同一份建築語料,三種投影","COLOR = BUILDING HEIGHT");
 const cols=[["p_bldg_pca","FIG.2.1 // PCA",GRN],["p_bldg_tsne","FIG.2.2 // t-SNE",CYAN],["p_bldg_isomap","FIG.2.3 // ISOMAP",AMB]];
 cols.forEach((c,i)=>{ const x=0.5+i*3.35; panel(s,x,2.2,3.1,3.3,c[1]); fig(s,c[0],x+0.12,2.66,2.86,2.7); });
 panel(s,10.6,2.2,2.25,3.3,"[ TRADE-OFF ]");
 const rows=[["","GLB","AXIS","GEN"],["PCA","+","+","+"],["TSNE","-","-","-"],["ISO","+","-","-"]];
 rows.forEach((r,i)=>{ const y=2.75+i*0.62; r.forEach((cell,j)=>{ const x=10.75+j*0.5; const col=i===0?MUT:(cell==="+"?GRN:cell==="-"?RED:TX); mo(s,cell,{x,y,w:0.5,h:0.4,fontSize:j===0?11:13,bold:j===0||i===0,color:j===0?TX:col,align:j===0?"left":"center",valign:"middle"}); }); if(i===0) hair(s,10.72,y+0.42,2.0,LINE); });
 mo(s,"GLB=全局 · AXIS=可解讀軸 · GEN=能生成 // PCA 能說 PC1 是什麼;t-SNE/Isomap 只給座標、回不去",{x:0.5,y:5.95,w:12.3,h:0.4,fontSize:11,color:MUT}); }

// ============ 12 CASE embeddings ============
caseFull("embeddings","01","城市大數據的「神經嵌入」自動浮現空間結構","把 POI · 街景 · 移動資料丟進表徵學習 → 城市自動浮現社區/功能區結構(不需人工分類)","研究:Neural embeddings of urban big data reveal spatial structures, Nature HSSC 2024 · 示意圖:城市衛星影像 CC BY-SA · Wikimedia");

// ============ 13 §03 divider ============
divider("§03 · NEURAL NETS","03","深度學習壓縮","AE · VAE · CNN · GNN — 以 AE、VAE 為主");

// ============ 14 AE ============
{ const s=chrome("§03 · AE","自編碼器"); head(s,"03","AE:學徒「臨摹→只憑幾個數字默寫」","ENCODER → BOTTLENECK → DECODER");
 panel(s,0.5,2.15,8.4,4.55,"[ FIG.3.1 ] AE ARCHITECTURE"); fig(s,"m_ae",0.7,2.6,8.0,3.9);
 panel(s,9.1,2.15,3.73,4.55,"[ KEY ]"); t(s,"瓶頸 = 潛碼",{x:9.3,y:2.65,w:3.4,h:0.4,fontSize:15,bold:true,color:RED});
 t(s,"壓到最窄還能畫回來 → 抓住了城市形態的精華。",{x:9.3,y:3.15,w:3.4,h:1.1,fontSize:13,color:TX});
 metric(s,9.3,4.4,3.4,"RECON MSE · LATENT=2","0.061","▼ 81% vs PCA k=2",GRN); }

// ============ 15 VAE ============
{ const s=chrome("§03 · VAE","變分自編碼器"); head(s,"03","VAE:學會「風格」才能創作","ENCODER → μ/σ → SAMPLE z → DECODER");
 panel(s,0.5,2.15,8.8,4.55,"[ FIG.3.2 ] VAE — μ/σ SAMPLING"); fig(s,"m_vae",0.7,2.6,8.4,3.9);
 panel(s,9.5,2.15,3.33,4.55,"[ KEY ]"); t(s,"不是一個點,是一個分布",{x:9.7,y:2.65,w:3,h:0.7,fontSize:14,bold:true,color:RED});
 t(s,"潛空間規範成連續高斯雲 N(0,I) → 隨手採樣都生出合理城市 = 生成。",{x:9.7,y:3.4,w:3,h:1.5,fontSize:12.5,color:TX});
 mo(s,"COST: recon 略遜 AE",{x:9.7,y:5.6,w:3,h:0.4,fontSize:11,color:AMB}); }

// ============ 16 latent grid ============
{ const s=chrome("§03 · VAE LATENT","潛空間網格(概念示意)"); head(s,"03","潛空間網格:移動 → 形態平滑變化","CONCEPTUAL · IMAGE-VAE LATENT GRID");
 panel(s,0.5,2.15,5.6,4.55,"[ FIG.3.3 ] LATENT GRID (示意)"); fig(s,"m_latentgrid",0.7,2.55,5.2,4.0);
 panel(s,6.3,2.15,6.53,4.55,"[ READING ]");
 t(s,"每一格 = 潛空間一個座標解碼的形態。左→右密度增、下→上高度增,相鄰格子平滑過渡 = 潛空間連續 → 任一點採樣都合理。\n\n誠實邊界:本專案只有 12 城、高度圖多空白,資料量不足以訓練出這麼清晰的「城市影像 VAE」——此頁為概念示意,實際壓縮/生成用表格 AE/VAE。",
  {x:6.55,y:2.65,w:6.05,h:3.9,fontSize:13.5,color:TX,lineSpacingMultiple:1.25,valign:"top"}); }

// ============ 17 CASE vae ============
caseFull("vae_street","02","VAE 把街道網路壓進潛空間,並生成新網路","VAE 抓住街道網路指標於低維向量,並能生成風格一致的新街道網路","研究:Kempinska & Murcio (2019), Modelling urban networks using VAEs, Applied Network Science · 示意圖:街道網路圖 CC BY-SA · Wikimedia");

// ============ 18 CNN ============
{ const s=chrome("§03 · CNN","影像視角 / 誠實邊界"); head(s,"03","CNN 讀「紋理」:概念好,但資料不夠","① CONCEPT vs ② REALITY");
 panel(s,0.5,2.1,6.6,2.35,"[ ① CONCEPT ] CNN 讀紋理"); fig(s,"m_cnn",0.65,2.5,6.3,1.85);
 panel(s,7.3,2.1,5.5,2.35,"[ WHY IMAGE ]"); t(s,"把整座城燒成高度影像,卷積讀街廓排列、疏密——urban form 最自然的 NN 表示。",{x:7.5,y:2.55,w:5.1,h:1.8,fontSize:13,color:TX});
 panel(s,0.5,4.65,8.6,2.05,"[ ② REALITY ] 此資料 CNN 重建 = 模糊"); fig(s,"m_cae_recon",0.65,5.05,8.3,1.55);
 panel(s,9.3,4.65,3.53,2.05,"[ WHY BLUR ]",RED); t(s,"只有 12 城、86% 空白、建築高頻 → 小模型只學到密度分布。需數千張街廓影像才清晰;故壓縮/生成用表格 AE/VAE。",{x:9.5,y:5.05,w:3.15,h:1.5,fontSize:11.5,color:TX}); }

// ============ 19 GNN concept ============
{ const s=chrome("§03 · GNN","圖神經網路"); head(s,"03","城市是一張「圖」:建築=節點,鄰接=邊","TABLE → GRAPH · GNN READS RELATIONS");
 panel(s,0.5,2.15,12.33,3.1,"[ FIG.3.4 ] CITY AS GRAPH"); fig(s,"m_graph",0.7,2.6,11.9,2.5);
 noteBar(s,5.4,"表格把每棟樓當獨立一列,丟了「誰挨著誰」。圖(Graph)= 節點(建築/地塊)+ 邊(鄰接/街道連通)。GNN 讀的是「關係」——urban 尺度最自然的結構。"); }

// ============ 20 GNN mp ============
{ const s=chrome("§03 · GNN · MESSAGE PASSING","訊息傳遞"); head(s,"03","訊息傳遞:每棟建築聚合街坊資訊","AGGREGATE NEIGHBOURS → CONTEXTUAL EMBEDDING");
 panel(s,0.5,2.15,12.33,3.1,"[ FIG.3.5 ] MESSAGE PASSING"); fig(s,"m_gnn_mp",0.7,2.6,11.9,2.5);
 noteBar(s,5.4,"① 每個節點聚合鄰居特徵 → ② 多層堆疊 → 更新成「帶鄰域脈絡」的嵌入。一棟建築的向量 = 自己屬性 + 街坊關係。應用:功能分區辨識、全城建築屬性預測、街道網路分類。"); }

// ============ 21 CASE gnn ============
caseFull("gnn","03","GNN 從街坊脈絡預測全城建築屬性","把建築連成鄰接圖,GraphSAGE 聚合街坊+街景脈絡,城市尺度預測建築屬性/功能","研究:Predicting building characteristics at urban scale using GNNs, CEUS 2024 · 示意圖:住宅街廓空拍 CC BY-SA · Wikimedia");

// ============ 22 §05 divider ============
{ const s=divider("§05 · RESULTS","05","訓練結果","換分析單位:每座城 × 體制 = 一個點(60 × 9)");
 panel(s,9.0,1.9,3.83,4.6,"[ REAL CITY 3D ]"); pfitJ(s,"out/city/city_3d.png",AR.city_3d,9.15,2.35,3.55,3.9); }

// ============ 23 RECON ============
{ const s=chrome("§05 · RECONSTRUCTION","recon MSE"); head(s,"05","重建:AE 2 維勝 PCA 4 維","LOWER = BETTER · STANDARDIZED SPACE");
 metric(s,0.6,2.2,2.6,"AE · LATENT=2","0.061","▼ 81% vs PCA k=2",GRN); metric(s,3.3,2.2,2.6,"AE · LATENT=3","0.033","▼ 68% vs PCA k=4",GRN);
 metric(s,6.0,2.2,2.6,"PCA · k=4","0.104","linear baseline",MUT); metric(s,8.7,2.2,2.6,"VAE · LATENT=2","0.241","△ recon (KL cost)",AMB);
 vhair(s,3.15,2.25,1.05,LINE); vhair(s,5.85,2.25,1.05,LINE); vhair(s,8.55,2.25,1.05,LINE);
 panel(s,0.6,3.7,6.0,3.0,"[ FIG.5.1 ] RECON MSE · METHOD × LATENT"); fig(s,"p_recon_bar",0.75,4.15,5.7,2.4);
 panel(s,6.8,3.7,6.0,3.0,"[ FIG.5.2 ] PER-FEATURE RECON MSE"); fig(s,"p_recon_feat",6.95,4.15,5.7,2.4); }

// ============ 24 TRAINING ============
{ const s=chrome("§05 · TRAINING","訓練診斷"); head(s,"05","AE/VAE 怎麼訓練出來的","OVERFIT · LOSS DECOMPOSITION · POSTERIOR COLLAPSE");
 const L3=[["p_loss_ae","(a) 過擬合","裸 AE 背答案(驗證變差)",CYAN],["p_loss_vae","(b) 損失拆解","VAE recon 與 KL 此消彼長",GRN],["p_loss_collapse","(c) 塌縮","β 太大壓垮 KL;退火避免",RED]];
 L3.forEach((l,i)=>{ const x=0.5+i*4.15; panel(s,x,2.2,3.9,3.3,`[ ${l[1]} ]`); fig(s,l[0],x+0.12,2.66,3.66,2.7); t(s,l[2],{x:x+0.12,y:5.55,w:3.8,h:0.5,fontSize:12,color:l[3]}); });
 mo(s,"> 訓練不是按一下:要調正則(防過擬合)+ 調 β 退火(防 collapse)",{x:0.5,y:6.15,w:12,h:0.4,fontSize:12,color:MUT}); }

// ============ 25 CLUSTER ============
{ const s=chrome("§05 · CLUSTER × POWER","群聚 vs regimes"); head(s,"05","資料自己會分出權力體制嗎?","UNSUPERVISED KMEANS(k=5) vs MANUAL REGIMES");
 panel(s,0.6,2.2,5.0,4.4,"[ TABLE ] ARI / SILHOUETTE");
 const T=[["SPACE","ARI","SIL"],["原始 9 維","0.70","0.32"],["VAE-2","0.56","0.58"],["PCA-2","0.55","0.49"],["AE-2","0.34","0.55"]];
 T.forEach((r,i)=>{ const y=2.85+i*0.72; r.forEach((c,j)=>{ const x=0.85+j*1.55; const col=i===0?MUT:(j===1&&i===1?AMB:j===2&&i===2?GRN:TX); mo(s,c,{x:j===0?0.85:x,y,w:j===0?2.3:1.4,h:0.5,fontSize:i===0?11:15,bold:i===0||(j>0&&(i===1||i===2)),color:col,valign:"middle",align:j===0?"left":"center"}); }); hair(s,0.85,y+0.6,4.5,LINE); });
 t(s,"ARI 0.70 → KMeans 只看形態就大致復原 05 手訂的 5 種體制 → regime 有形態資料支撐(非憑空)",{x:0.85,y:5.75,w:4.5,h:0.8,fontSize:12,color:TX,lineSpacingMultiple:1.2});
 panel(s,5.9,2.2,3.4,4.4,"[ FIG.5.3 ] KMEANS × REGIME"); fig(s,"p_confusion",6.05,2.7,3.1,3.7);
 panel(s,9.6,2.2,3.25,4.4,"[ FIG.5.4 ] ARI BY SPACE"); fig(s,"p_ari",9.75,2.7,2.95,3.7); }

// ============ 26 INTERPRET ============
{ const s=chrome("§05 · INTERPRETABILITY","誰在成簇"); head(s,"05","誰在成簇:體制,不是城市家族","SILHOUETTE BY REGIME ≫ BY FAMILY");
 panel(s,0.5,2.15,7.4,2.3,"[ FIG.5.5 ] PCA LOADINGS"); fig(s,"p_loadings",0.65,2.5,7.1,1.85);
 panel(s,0.5,4.65,6.0,2.05,"[ FIG.5.6 ] SILHOUETTE 家族 vs 體制"); fig(s,"p_silhouette",0.65,5.05,5.7,1.55);
 panel(s,8.1,2.15,4.73,4.55,"[ READING ]"); t(s,"silhouette 按體制 ≫ 按家族(0.24–0.42 vs ≈0)。",{x:8.3,y:2.6,w:4.4,h:0.7,fontSize:14,bold:true,color:AMB});
 t(s,"同一套權力算子把不同城推向一致方向;城市原本的類型反而被打散。",{x:8.3,y:3.5,w:4.4,h:1.5,fontSize:13,color:TX});
 t(s,"權力 > 地方性。",{x:8.3,y:5.4,w:4.4,h:0.6,fontSize:18,bold:true,color:CYAN}); }

// ============ 27 GENERATION ============
{ const s=chrome("§06 · GENERATION","生成能力"); head(s,"06","能生成 = 有沒有 decoder","SAMPLE latent → DECODE · plausibility↓ validity↑");
 panel(s,0.6,2.2,5.4,2.5,"[ CAPABILITY MATRIX ]");
 const G=[["METHOD","DECODE","PRIOR","PLAUS","VALID"],["VAE","✔","✔ N(0,I)","1.03","100%"],["PCA","✔","~","1.40","97%"],["AE","✔","✗","2.11","93%"],["TSNE/ISO","✗","✗","—","—"]];
 G.forEach((r,i)=>{ const y=2.72+i*0.38; r.forEach((c,j)=>{ const x=0.8+j*1.05; const col=i===0?MUT:j===0?TX:(c.includes("✔")?GRN:c.includes("✗")?RED:i===1?GRN:TX); mo(s,c,{x:j===0?0.8:x+0.2,y,w:j===0?1.4:1.05,h:0.34,fontSize:i===0?9.5:11.5,bold:i===0||i===1,color:col,valign:"middle"}); }); if(i===0) hair(s,0.8,y+0.36,5.0,LINE); });
 const cl=[["p_cloud_pca","PCA"],["p_cloud_ae","AE"],["p_cloud_vae","VAE"]];
 cl.forEach((c,i)=>{ const x=6.3+i*2.2; panel(s,x,2.2,2.05,2.9,`[ ${c[1]} ] real·gen`); fig(s,c[0],x+0.1,2.66,1.85,2.3); });
 noteBar(s,4.95,"VAE 有先驗 N(0,I) → 生成點最貼真實流形、100% 守物理約束、插值最連續。AE 無先驗、隨機採樣落洞 → 偏離。t-SNE/Isomap 只投影、回不去 → 不可生成。"); }

// ============ 28 CASE generative ============
caseFull("generative","04","生成式都市設計:從草圖到即時真實渲染","從設計草圖/語義遮罩 → 生成模型即時渲染真實都市場景,支援人機互動迭代設計","研究:Deep Image of the City (MIT Media Lab);StreetGAN / DeepStreet · 示意圖:都市天際線 CC BY-SA · Wikimedia");

// ============ 29 POWER ============
{ const s=chrome("§05 · POWER REDISTRIBUTION","權力重新分配"); head(s,"05","每種體制把城市推向哪裡","FROM METRIC → REAL FORM");
 panel(s,0.5,2.15,4.0,4.55,"[ 3D 量體 ] 現狀 vs 開發商"); pfitJ(s,"out/city/regime_3d_pair.png",AR.regime_3d_pair,0.65,2.55,3.7,3.95);
 panel(s,4.7,2.15,8.13,2.35,"[ FIG.5.7 ] 形態簽名 Δ"); fig(s,"p_signature",4.85,2.5,7.83,1.85);
 const sig=[["開發商","拔高塔化 h_mean↑",RED],["政府","向公共重心 concentration↑",CYAN],["居民自建","細粒化 grain↓、棟數↑",GRN],["共享","釋放開放地面 coverage↓",AMB]];
 sig.forEach((g,i)=>{ const x=4.7+(i%2)*4.1,y=4.7+Math.floor(i/2)*1.05; panel(s,x,y,3.9,0.92); t(s,g[0],{x:x+0.2,y:y+0.1,w:3.5,h:0.35,fontSize:14,bold:true,color:g[2]}); t(s,g[1],{x:x+0.2,y:y+0.48,w:3.5,h:0.4,fontSize:12,color:TX}); }); }

// ============ 30 SPECTRUM ============
{ const s=chrome("§06 · SYNTHESIS","方法光譜"); head(s,"06","方法光譜:哪個工具解哪種規劃問題","CONTROLLABLE ←→ GENERATIVE");
 hair(s,0.9,3.6,11.5,DIM);
 const meth=[["Lookup",1.3,CYAN],["PCA",3.3,CYAN],["t-SNE/Iso",5.4,MUT],["AE",7.6,AMB],["VAE",9.7,GRN],["Conv/GNN",11.7,GRN]];
 meth.forEach(m=>{ s.addShape(P.shapes.OVAL,{x:m[1]-0.1,y:3.5,w:0.2,h:0.2,fill:{color:BG},line:{color:m[2],width:1.5}}); mo(s,m[0],{x:m[1]-1,y:4.0,w:2,h:0.4,fontSize:12,bold:true,color:m[2],align:"center"}); });
 mo(s,"可控 · 透明",{x:0.9,y:3.1,w:3,h:0.4,fontSize:13,bold:true,color:CYAN}); mo(s,"強大 · 能生成",{x:9.9,y:3.1,w:3,h:0.4,fontSize:13,bold:true,color:GRN,align:"right"});
 const grid=[["要可控","規則 Lookup"],["要看結構","PCA / Isomap"],["要生成/做方案","VAE"],["要空間紋理/關係","Conv-VAE / GNN"]];
 grid.forEach((g,i)=>{ const x=0.6+(i%2)*6.2,y=4.9+Math.floor(i/2)*0.85; panel(s,x,y,5.9,0.7); mo(s,g[0],{x:x+0.2,y:y+0.05,w:2.6,h:0.6,fontSize:13,color:AMB,valign:"middle"}); t(s,"→ "+g[1],{x:x+2.9,y:y+0.05,w:2.9,h:0.6,fontSize:13,bold:true,color:TX,valign:"middle"}); }); }

// ============ 31 CONCLUSION ============
{ const s=P.addSlide(); PG++; s.background={color:BG}; hair(s,0.5,0.62,W-1,LINE);
 mo(s,"§07 · TAKEAWAYS",{x:0.5,y:0.28,w:8,h:0.3,fontSize:11,color:GRN,charSpacing:2}); mo(s,"31 / 31",{x:W-3,y:0.28,w:2.5,h:0.3,fontSize:11,color:MUT,align:"right"});
 t(s,"啟發 & 誠實邊界",{x:0.5,y:1.0,w:12,h:0.7,fontSize:34,bold:true,color:TX});
 const ins=[["01","把「權力→形態」從描述升級成可量化、可比較、可生成的空間",GRN],["02","資料驅動群聚能檢驗人訂規則(ARI 0.70),甚至提出沒想到的族群",CYAN],["03","VAE 連續潛空間 = 情景生成器雛形:規劃師出概念,NN 出候選",AMB]];
 ins.forEach((it,i)=>{ const y=2.1+i*1.0; mo(s,it[0],{x:0.6,y,w:0.9,h:0.6,fontSize:22,bold:true,color:it[2],valign:"middle"}); t(s,it[1],{x:1.6,y,w:7.1,h:0.7,fontSize:15,color:TX,valign:"middle"}); hair(s,0.6,y+0.8,8.0,LINE); });
 panel(s,9.0,2.1,3.83,3.1,"[ 誠實邊界 ]",RED);
 t(s,"樣本小 → AE/VAE/CNN 皆教學演示,非嚴謹預測。\n\n體制簽名是算子設定的必然結果(教學語言)。\n\nNN 是放大鏡不是水晶球:價值判斷仍是規劃師的責任。",{x:9.2,y:2.6,w:3.45,h:2.5,fontSize:12,color:TX,lineSpacingMultiple:1.15,valign:"top"});
 hair(s,0.5,5.9,W-1,GRN); t(s,"NN 讓 05 資料能做更深的分析 —— 而規劃專業,始終是主角。",{x:0.5,y:6.15,w:12.3,h:0.7,fontSize:20,bold:true,italic:true,color:TX,align:"center"});
 mo(s,"URBAN-LATENT // 城市潛空間編碼 · 2026",{x:0.5,y:H-0.5,w:8,h:0.3,fontSize:10,color:DIM}); }

P.writeFile({ fileName: "工作營_潛空間.pptx" }).then(f => console.log("WROTE", f));
