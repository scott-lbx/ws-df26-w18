#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_v05_report.py — 組裝 ../V05_REPORT.html(研究生研究報告體例,自含、圖 base64)。
固定章節來自 report_static.py;結果章節在此生成,數字直接取自 results_e*.json(audit 已確認)。"""
import base64, json, os, sys

V05 = os.path.dirname(os.path.abspath(__file__))
TF = os.path.dirname(V05)
sys.path.insert(0, V05)
import report_static as ST

J = {n: json.load(open(os.path.join(V05, f"results_{n}.json"), encoding="utf-8"))
     for n in ["e0", "e1", "e2", "e3", "e4"]}
WIN = json.load(open(os.path.join(V05, "windows.json"), encoding="utf-8"))
AGR = json.load(open(os.path.join(V05, "agent_reports.json"), encoding="utf-8"))


def b64(name):
    with open(os.path.join(V05, "fig", name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def fig_html(name, caption):
    return (f'<figure><img src="{b64(name)}" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>")


def render_static(s):
    while "[[FIG:" in s:
        i = s.index("[[FIG:")
        j = s.index("]]", i)
        body = s[i + 6:j]
        name, cap = body.split("|", 1)
        s = s[:i] + fig_html(name.strip(), cap.strip()) + s[j + 2:]
    return s


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ================================================================ numbers
e1 = J["e1"]
runs = e1["runs"]


def r1(split, fs):
    return runs[f"{split}_{fs}"]


e2 = J["e2"]
prof = e2["window_profiles"]
e3 = J["e3"]
scr = e3["correlation_screen"]
loo = {dim: {fs: e3["loo"][fs][dim] for fs in ("FORM", "TYPE", "BOTH")}
       for dim in e3["loo"]["FORM"]}          # re-pivot: loo[dim][SET]
sens = {dim: {"r2": v["loo_r2"], "mae": v["mae"]}
        for dim, v in e3["sensitivity_form_loo_excl_zero_age"].items()
        if isinstance(v, dict) and "loo_r2" in v}
e4 = J["e4"]

# ================================================================ SEC4 — E1
conf_rows = ""
for split, fs, tag in [("strat5", "FORM", "分層隨機 5 折 / 純形態"), ("strat5", "FORM+CTX", "分層隨機 5 折 / 形態+脈絡"),
                       ("group24", "FORM", "留窗 / 純形態"), ("group24", "FORM+CTX", "留窗 / 形態+脈絡")]:
    m = r1(split, fs)
    conf_rows += (f"<tr><td>{tag}</td><td>{m['accuracy']:.3f}</td><td>{m['balanced_accuracy']:.3f}</td>"
                  f"<td>{m['macro_f1']:.3f}</td></tr>")
pwg = e1["power_supergroup"]

SEC4 = f"""
<h2 id="sec4">4 實驗一:建築層級的辨識——形態能認出類型嗎?能轉移嗎?</h2>

<h3>4.1 設計</h3>
<p>樣本:資料集中 {e1['n_buildings_labeled']:,} 棟有標籤建築(8 類;715 棟 unlabeled 剔除)。
模型:RandomForest(400 樹、類別平衡、seed 0)。特徵兩組:<b>FORM</b>(足印面積、高度、樓層、長寬比)
與 <b>FORM+CTX</b>(加 100 m 鄰里密度、鄰均高、四個 v0.4 遮罩距離——中位數插補 + 缺失旗標)。
交叉驗證兩種設計並列:<b>分層隨機 5 折</b>(StratifiedKFold,與舊 C 報告可比)與 <b>留窗(GroupKFold by window)</b>
——後者是誠實的空間轉移考試:模型沒見過該窗的任何建築。</p>
<div class="choice"><b>選擇:</b>POI/EULUC/AOI 衍生特徵一律排除。<b>備選:</b>納入 POI 計數
(C 報告作法)。<b>理由:</b>這些來源製造了標籤本身——拿它們當特徵是循環論證。
代價:本實驗測的是「形態(+空間脈絡)對融合標籤的可見度」,上界天然比 C 報告低。</div>
<div class="choice"><b>選擇:</b>雙 CV 設計並列而非擇一。<b>備選:</b>只報隨機 CV(數字好看)。
<b>理由:</b>融合標籤被 EULUC parcel 主導,同地塊建築共享標籤——隨機折把同一地塊拆進
訓練與測試,洩漏不可避免;兩個數字的<b>差</b>本身是研究發現(洩漏的大小)。</div>

<h3>4.2 結果</h3>
<table><tr><th>設計 / 特徵</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro-F1</th></tr>
<tr><td>多數類基準(全部猜 residential_estate)</td><td>{e1['baseline']['accuracy']:.3f}</td><td>0.125(8 類隨機)</td><td>—</td></tr>
{conf_rows}</table>
<p><b>主發現一:C 報告的 72% 不能複製。</b>在 14k 尺度 + 融合標籤 + 誠實切分下,
留窗 accuracy {r1('group24','FORM+CTX')['accuracy']:.3f} 只比 0.601 的多數類基準高 4 個百分點,
balanced accuracy {r1('group24','FORM+CTX')['balanced_accuracy']:.3f}(8 類隨機 = 0.125)——
跨窗可轉移的形態知識,大致只剩「集合住宅 vs 低層住宅」這一刀。
分層隨機 CV 比留窗高出 {100*(r1('strat5','FORM+CTX')['accuracy']-r1('group24','FORM+CTX')['accuracy']):.1f} 個百分點,
這個差就是空間自相關 + parcel 標籤洩漏的量。</p>
<p><b>主發現二:權力類建築在轉移下幾乎不可見。</b>{{public_admin, education, healthcare}}
合計 {pwg['strat5_FORM']['n_power_buildings']:,} 棟,留窗 pooled recall:純形態 {pct(pwg['group24_FORM']['pooled_recall'])}、
形態+脈絡 <b>{pct(pwg['group24_FORM+CTX']['pooled_recall'], 2)}</b>;healthcare 在兩種留窗設計下 recall 恰為 0(分層隨機 CV 加脈絡下也僅 1/128)。
更尖銳的是:脈絡特徵在分層隨機 CV 下把權力類 recall 增至 2.4 倍({pct(pwg['strat5_FORM']['pooled_recall'])}→{pct(pwg['strat5_FORM+CTX']['pooled_recall'])}),
在留窗下反而<b>更差</b>——距離特徵實際上是「窗內位置指紋」,幫模型背 parcel,不是可轉移的知識。</p>
{fig_html('fig_e1_confusion.png', '圖 4-1 — 留窗混淆矩陣(左:純形態;右:形態+脈絡)。三個權力類幾乎整列被吸進兩個住宅類。')}
{fig_html('fig_e1_transfer.png', '圖 4-2 — 兩種 CV × 兩組特徵的總表現、逐窗留出準確率(大致是住宅佔比的代理)、特徵重要度(鄰均高與 dist_green 居前——新脈絡特徵窗內有效、跨窗失效)。')}
<div class="failbox"><b>失敗作為結果:</b>(i) 72%(C 報告:3 窗、OSM 標籤、含 POI 特徵、隨機 CV)
→ 62.7%(v05:24 窗、融合標籤、純形態+脈絡、留窗)的縮水跨了三重變因,不可單獨歸因;
乾淨的洩漏量是 v05 內部的 10.2 個百分點差。結論方向一致:「電腦能從形讀出類型」應修正為
「電腦能從形讀出<b>家族</b>,而且只在它見過的街區」;
(ii) A32 城郊村 463 棟有標籤建築中 462 棟(99.8%)標為 residential_estate,模型卻預測 house 族——錯的可能是 parcel 級標籤
而不是模型,標籤粗糙度本身被實驗照出來。</div>
"""

# ================================================================ SEC5 — E2
def prow(wid):
    p = prof[wid]
    w = p["profile"]
    return (f"<tr><td>{wid} {p['name']}</td><td>{w['government']:.3f}</td><td>{w['developer_capital']:.3f}</td>"
            f"<td>{w['resident_community']:.3f}</td><td>{w['informal']:.3f}</td>"
            f"<td>{pct(p['low_confidence_gfa_share'],0)}</td><td>{pct(p['unlabeled_gfa_share'],0)}</td></tr>")


notable = ["A43", "A13", "A01", "A07", "A33", "A47"]
fc = e2["flagship_comparison"]

SEC5 = f"""
<h2 id="sec5">5 實驗二:權力歸屬鏈在 24 窗尺度運轉</h2>

<h3>5.1 設計</h3>
<p>先驗原樣沿用 v0.1(文獻常數,逐格出處+信度;§3.4),逐棟賦權
(residential_estate 按年代制度分格),GFA 加權聚合成視窗剖面;低信度 GFA 份額與
unlabeled GFA 份額作為兩個誠實度診斷隨剖面交付。語義×家族共現橋只從 18 個 TRAIN 窗計數
({e2['meta']['n_train_labeled']:,} 棟)。</p>

<h3>5.2 結果</h3>
<table><tr><th>視窗(節選)</th><th>gov</th><th>dev</th><th>res</th><th>inf</th><th>低信度GFA</th><th>未標GFA</th></tr>
{''.join(prow(w) for w in notable)}</table>
<p>先驗加權意義下,24 窗中 <b>9 窗政府主導、15 窗開發資本主導、0 窗居民或非正式主導</b>
(res 最高 {prof['A33']['profile']['resident_community']:.3f}、inf 最高 {prof['A33']['profile']['informal']:.3f},都在浦東城郊村)。
探針的曹楊意外原樣重現(dev {prof['A13']['profile']['developer_capital']:.3f} &gt; gov {prof['A13']['profile']['government']:.3f}
——同先驗同標籤,屬管線複製驗證,非獨立證據)。</p>
{fig_html('fig_e2_profiles.png', '圖 5-1 — 24 窗權力剖面(四欄 lollipop;權重加法非互斥,不用堆疊條)。空心點=低信度 GFA 份額 > 50%。')}
{fig_html('fig_e2_bridge.png', '圖 5-2 — 語義×家族共現橋(18 TRAIN 窗)與探針旗艦格對照。')}
<p><b>橋的尺度穩定性:</b>最乾淨的格在 6 倍樣本下保持——residential_lowrise→house
{fc['v05_train18_count']['residential_lowrise->house']:.3f}(v05 計數 1,355/1,357;探針 1.00);
estate 列照樣發散但眾數家族翻轉:slab {fc['v05_train18_count']['residential_estate->slab']:.3f}(探針 0.34)、
midrise {fc['v05_train18_count']['residential_estate->midrise_other']:.3f}(探針 0.50)——探針的三個 patch 高估了中層小區。</p>
<div class="failbox"><b>兩個結構性誠實警報:</b>
(i) <b>五個零年代郊區窗裡有四個(A29/A32/A33/A41)的「開發商主導」是 fallback 人造物</b>:
Age 全缺 → 全部 estate 落到 no_age 低信度格(gov .4/dev .5),A32 的低信度 GFA 份額高達
{pct(prof['A32']['low_confidence_gfa_share'],0)}——這些窗的剖面是先驗的回聲,不是證據。
(ii) <b>外灘是全 24 窗政府權重最低者({prof['A07']['profile']['government']:.3f})</b>——對一個
國家強管控的歷史濱江帶而言顯然荒謬;原因是先驗詞彙表裡<b>沒有遺產/國家保護這個權力通道</b>,
外灘的 office_finance/commercial 類把它讀成純資本。先驗的類別表不完備,實驗把缺口指名了。</div>
<div class="choice"><b>選擇:</b>GFA 加權聚合。<b>備選:</b>逐棟等權。<b>理由:</b>權力問題問的是
「誰主導了建成<b>體量</b>」;代價:小體量自建房攜帶的 res/inf 權重被體量壓制——
「0 窗居民主導」一半是先驗設計的結構性結果,必須與結果一起申報。</div>
"""

# ================================================================ SEC6 — E3
def srow(dim, label):
    cells = scr[dim]
    tops = sorted(cells.items(), key=lambda kv: -abs(kv[1]["rho"]))[:3]
    s = "、".join(f"{k} ρ={v['rho']:+.2f}{'※' if v['bh_fdr10'] else ''}" for k, v in tops)
    return f"<tr><td>{label}</td><td>{s}</td></tr>"


SEC6 = f"""
<h2 id="sec6">6 實驗三:三角的斜邊——視窗形態能回收權力剖面嗎?</h2>

<h3>6.1 設計</h3>
<p>這是 v05 的核心新實驗。E2 的權力剖面<b>由類型×年代構造而來</b>(asserted,非觀測);
真問題是:跳過 metadata,純形態簽名能回收多少?哪裡失敗?n=24(處處披露)。
三層分析:(1) Spearman 相關篩查(4 維 × 12 形態變數,permutation p 1 萬次,BH FDR 0.10);
(2) LOO Ridge 預測(巢狀 LOO 自 α∈{0.1, 1, 10, 100} 選定,折內標準化),特徵三組:FORM(10 個形態指標;PC1/PC2 只用於相關篩查、不進預測特徵)、
TYPE(語義 GFA 構成——設計時以為是循環上限)、BOTH;(3) 18 TRAIN→6 HELDOUT 二次檢核
+ 剔除四個零年代人造目標窗(E2 警報)的敏感性分析。</p>

<h3>6.2 結果</h3>
<table><tr><th>權力維度</th><th>最強的三個形態相關(※=BH FDR 0.10 存活;48 試驗中 {len(e3['bh_fdr10_survivors'])} 個存活)</th></tr>
{srow('government','government')}{srow('developer_capital','developer_capital')}
{srow('resident_community','resident_community')}{srow('informal','informal')}</table>
<p><b>主發現三:權力訊號掛在 PC2,不在 PC1。</b>所有人(包括我們自己的假設)以為承載資本的
是強度/垂直度軸(PC1),但四個維度與 PC1 的相關全部不顯著(最大 |ρ|=0.28);
分離 gov/dev 的是 <b>PC2(空曠尖峰 vs 密實毯)與 coverage</b>:
dev↔PC2 ρ=+{scr['developer_capital']['PC2']['rho']:.2f}、dev↔coverage ρ={scr['developer_capital']['coverage']['rho']:.2f};
gov 正好相反。開發商權重高的窗是「塔樓+大片空地」的新建肌理,政府權重高的窗是高覆蓋密實毯。</p>
{fig_html('fig_e3_cloud_power.png', '圖 6-1 — v0.3 簽名雲(PCA)上的四個權力維度。分離發生在縱軸(PC2),不在橫軸(PC1)。')}
<p><b>主發現四:只有 developer_capital 能從形態穩健回收。</b>LOO R²:
dev <b>+{loo['developer_capital']['FORM']['loo_r2']:.2f}</b>(剔除人造目標窗後 +{sens['developer_capital']['r2']:.2f},穩健)、
gov +{loo['government']['FORM']['loo_r2']:.2f}(敏感性分析翻負 {sens['government']['r2']:+.2f}——不可宣稱)、
res {loo['resident_community']['FORM']['loo_r2']:.2f}(單調相關存在但點預測失敗)、
inf {loo['informal']['FORM']['loo_r2']:.2f}(目標標準差僅 0.016——但注意 16 個 FDR 存活相關中
6 個屬 informal:它<b>看得見方向、量不準大小</b>,點預測失敗而秩相關存在,
與 resident_community 同一種失敗形)。</p>
<p><b>留出二次檢核(18 TRAIN→6 HELDOUT):</b>FORM 特徵 MAE = gov 0.069 / dev 0.089 /
res 0.023 / inf 0.006(n=6 的 R² 無意義,只報 MAE);誤差形態是典型的回歸均值——
全 24 窗政府權重最高的 A43(0.600)被預測成 0.459,其 dev 誤差 −0.215。</p>
<p><b>主發現五(本輪最大意外):TYPE 不是天花板。</b>設計時假定「類型構成→權力」是循環構造、
TYPE 特徵組必然滿分;實際 TYPE 對 gov R²={loo['government']['TYPE']['loo_r2']:+.2f}、
dev {loo['developer_capital']['TYPE']['loo_r2']:+.2f}——<b>失敗</b>。原因:gov/dev 的視窗間變異
主要由 residential_estate 的<b>年代條件格</b>驅動,而年代不在類型構成裡。
FORM 之所以能回收 dev,是因為高度/塔樓份額是<b>建造年代的代理</b>——
<b>三角不是經由「類型」閉合,是經由「時間」閉合</b>。TYPE 的天花板地位只在
年代無關的維度成立(res {loo['resident_community']['TYPE']['loo_r2']:.2f}、inf {loo['informal']['TYPE']['loo_r2']:.2f},循環如預期)。</p>
{fig_html('fig_e3_loo.png', '圖 6-2 — 上:FORM 特徵 LOO 預測散點(對角=完美);下:三組特徵的 LOO R² 對比——TYPE 在 gov/dev 上塌掉,年代是缺的那塊。')}
<div class="choice"><b>選擇:</b>permutation p 值 + BH FDR 而非裸 p。<b>備選:</b>常態近似 p。
<b>理由:</b>n=24 非常態,48 個同時檢定不控 FDR 必出假陽性;代價:功效低,真效應可能漏報。</div>
<div class="failbox"><b>效度警告(構念層級):</b>E3 測的是「<b>被斷言的</b>權力剖面之內部可回收性」,
不是上海的實際權力。它能證偽的是鏈條的自洽,不能證實權力本身——校準需要檔案證據(§10)。</div>
"""

# ================================================================ SEC7 — E4
dc = e4["direction_checks"]
DC_ZH = {"DEV_highest_tower_share": "DEV 的 tower 份額最高",
         "DEV_highest_T6T8_cell_share": "DEV 的 T6+T8 高強度格份額最高",
         "DEV_highest_h_p90": "DEV 的 h_p90 最高",
         "RES_highest_house_share": "RES 的 house 份額最高",
         "GOV_highest_large_footprint_share": "GOV 的 large_footprint 份額最高(fringe-belt 直覺)",
         "GOV_highest_public_gfa_share_rendered": "GOV 渲染後公共類 GFA 份額最高"}
dc_rows = "".join(
    f"<tr><td>{DC_ZH.get(k, k)}</td>"
    f"<td>{'= 平手' if v.get('tie') else ('✓ 成立' if v['held'] else '✗ 失敗')}</td>"
    f"<td>winner: {v['winner']};{' / '.join(f'{c} {x:.3g}' for c, x in v['values'].items())}</td></tr>"
    for k, v in dc.items())
pc = e4["per_config"]

SEC7 = f"""
<h2 id="sec7">7 實驗四:反事實生成——同一塊地,三種權力</h2>

<h3>7.1 設計</h3>
<p>把探針的 Generate 鏈在 v0.4 基建上重走一遍:權力配置 → (先驗點積)語義 mix →
(E2 橋)家族 GFA 需求 → (NNLS 對 token 家族構成)token 格數 → rank-match 到
A20 三林(留出窗)的 typed 骨架 → all-TRAIN 池 keep-out 拼貼。三組配置(GOV/DEV/RES,
加法非互斥),同一骨架、同一遮罩、各自獨立 seed 0。六個方向預期事先寫下再對答案。</p>

<h3>7.2 結果:鏈條傳得動「垂直度」與「house 性」,傳不動「身分」</h3>
<table><tr><th>指標</th><th>GOV</th><th>DEV</th><th>RES</th><th>真實 A20</th></tr>
<tr><td>FAR ※(重縮放人造物,不作實質解讀)</td><td>{pc['GOV']['indicators']['FAR']:.2f}</td><td>{pc['DEV']['indicators']['FAR']:.2f}</td><td>{pc['RES']['indicators']['FAR']:.2f}</td><td>{e4['real_site_indicators']['FAR']:.2f}</td></tr>
<tr><td>h_p90 (m)</td><td>{pc['GOV']['indicators']['height_p90']:.0f}</td><td><b>{pc['DEV']['indicators']['height_p90']:.0f}</b></td><td>{pc['RES']['indicators']['height_p90']:.0f}</td><td>{e4['real_site_indicators']['height_p90']:.0f}</td></tr>
<tr><td>house 份額</td><td>{pc['GOV']['indicators']['share_house']:.2f}</td><td>{pc['DEV']['indicators']['share_house']:.2f}</td><td><b>{pc['RES']['indicators']['share_house']:.2f}</b></td><td>{e4['real_site_indicators']['share_house']:.2f}</td></tr>
<tr><td>tower 份額</td><td>{pc['GOV']['indicators']['share_tower']:.2f}</td><td><b>{pc['DEV']['indicators']['share_tower']:.2f}</b></td><td>{pc['RES']['indicators']['share_tower']:.2f}</td><td>{e4['real_site_indicators']['share_tower']:.2f}</td></tr>
<tr><td>渲染後公共類 GFA 份額</td><td>{pc['GOV']['public_gfa_share_rendered']:.3f}</td><td>{pc['DEV']['public_gfa_share_rendered']:.3f}</td><td>{pc['RES']['public_gfa_share_rendered']:.3f}</td><td>—</td></tr></table>
<table><tr><th>方向預期(事先註冊)</th><th>裁定</th><th>讀法</th></tr>{dc_rows}</table>
{fig_html('fig_e4_counterfactual.png', '圖 7-1 — 同一骨架三種權力配置:RES=平鋪 house 布(h90 22 m)、DEV=最高最尖(h90 47 m)、GOV=居間。ILLUSTRATIVE, NOT PREDICTIVE。')}
<div class="failbox"><b>兩個失敗,都有內容:</b>
(i) <b>GOV 要了 {pct(sum(e4['semantic_mixes']['GOV'][c] for c in ('public_admin','education','healthcare')),1)} 的公共類語義,渲染出來只剩
{pct(pc['GOV']['public_gfa_share_rendered'],1)}</b>,甚至低於 RES——語義需求在「家族/token」翻譯中被剝掉了:
貼上來的格元帶著自己池捐贈者的語義,k=8 的形態詞彙在傳輸中剝離機構身分。
這是 v0.3「池稀釋」的語義版,也再次以生成端複現了 anti-physiognomy:<b>形態傳輸不了「誰的」</b>。
(ii) 預期「GOV 最大 large_footprint」(Conzen fringe-belt 直覺)輸給了資料:上海的經驗橋把
大底盤 GFA 配給商業/辦公(商場、裙樓),不配給 public_admin——文獻直覺被在地資料修正。</div>
<div class="choice"><b>選擇:</b>NNLS 解 family→token 的欠定分配,並把格數重縮放到 293 個建成格
(GOV ×{e4['per_config']['GOV']['rescale_factor_to_built_cells']:.2f} / DEV ×{e4['per_config']['DEV']['rescale_factor_to_built_cells']:.2f} /
RES ×{e4['per_config']['RES']['rescale_factor_to_built_cells']:.2f})。<b>備選:</b>不重縮放(各配置建成格數不同)。
<b>理由:</b>固定骨架=控制變因(差異全來自 token 重新分配);代價:GOV 的 FAR 偏高部分是重縮放人造物,
表中 FAR 行已加註 ※,不作實質解讀。</div>
"""

# ================================================================ SEC8-10
SEC8 = f"""
<h2 id="sec8">8 綜合討論</h2>
<p>四個實驗拼起來,對「分類—權力—形態」三角的回答是分層的:</p>
<table><tr><th>命題</th><th>證據</th><th>狀態</th></tr>
<tr><td>① 建築層級,形態看不見權力</td><td>E1:權力類留窗 recall ≤1.2%,healthcare=0;
72% 的舊樂觀被診斷為洩漏</td><td><b>強化確認</b>(A/B/C 的論點在 14k 尺度 + 誠實切分下更尖銳)</td></tr>
<tr><td>② 類型攜帶權力語義、橋可學</td><td>E2:鏈條在 24 窗跑通;橋的乾淨格尺度穩定
(lowrise→house 0.999);但先驗詞彙缺遺產通道(外灘異常)、零年代窗變先驗回聲</td>
<td><b>有條件成立</b>(鏈條自洽 ≠ 校準;缺口被指名)</td></tr>
<tr><td>③ 視窗層級,形態可部分回收權力</td><td>E3:只有 developer_capital 穩健(LOO R² 0.46),
訊號在 PC2/coverage 不在 PC1;informal 點預測失敗(秩相關存在但量不準)</td><td><b>部分成立,且路徑出乎意料</b></td></tr>
<tr><td>④ 權力配置可被生成為形態</td><td>E4:垂直度與 house 性傳得動(3/6 方向成立),
機構身分傳不動(語義在 token 翻譯中剝離)</td><td><b>半通</b>(傳形不傳名)</td></tr></table>
<p><b>貫穿四個實驗的同一條暗線是「時間」。</b>E3 的 TYPE 天花板塌掉,是因為 gov/dev 的變異
由年代條件格驅動;FORM 能回收 dev,是因為塔樓+空地的形態是 1998 後商品房時代的<b>年代簽名</b>;
E2 的最大誠實警報(零年代郊區)同樣是年代缺失;而資料側最弱的欄位恰好就是 Age
(39.2% 覆蓋、1984/85 截斷、曹楊 1950 年代工人新村的 residential_estate GFA 僅 6% 標為 pre-1985)。
<b>形態部分看得見「何時」,「何時」部分看得見「誰」——三角經由時間閉合,
而時間正是這份資料最殘缺的維度。</b>這把下一步的資料優先序定死了:建成年代
是整個研究槓桿率最高的單一資料改進。</p>
<p>對工作坊(DigitalFUTURES 2026)的含義:S2P 原論文的手編 affinity table 在這裡被替換成
一條<b>每一格可吵</b>的鏈(融合標籤 → 帶出處先驗 → 共現橋 → 生成),而 v05 的四個實驗
就是四套現成的課堂辯論材料——尤其是 E1 的「洩漏 vs 轉移」、E3 的「PC2 不是 PC1」、
E4 的「傳形不傳名」,每個都能讓學生在 2 分鐘內重跑改一格再吵一輪。</p>
"""

SEC9 = """
<h2 id="sec9">9 侷限與效度威脅</h2>
<table><tr><th>效度類別</th><th>威脅</th><th>影響範圍</th></tr>
<tr><td rowspan="2"><b>構念效度</b></td><td>權力剖面是文獻先驗的斷言,不是觀測;informal 維度全低信度,
里弄以城中村文獻類比(制度上不等同)</td><td>E2/E3/E4 的全部「權力」結論</td></tr>
<tr><td>先驗詞彙不完備:無遺產/國家保護通道(外灘異常)、無金融資本/全球資本之分</td><td>E2 剖面;E4 語義 mix</td></tr>
<tr><td rowspan="3"><b>內部效度</b></td><td>標籤循環:語義標籤由 EULUC/AOI/OSM/POI 融合而來,E1 即使排除這些特徵,
標籤本身的 parcel 粗糙度仍會把錯算到模型頭上(A32 案例)</td><td>E1 絕對數字</td></tr>
<tr><td>Age 截斷(1984/85 歸併、2018 後無資料、39% 覆蓋):pre-1985 是地板類不是真年代;
零年代窗的剖面是先驗回聲</td><td>E2 年代條件化;E3 的時間論證</td></tr>
<tr><td>生成端:NNLS 欠定(T2 在三組配置中永不被用,各配置另有 2–3 個 token 為 0)、重縮放扭曲 FAR、單 seed、T8 重尾</td><td>E4 全部</td></tr>
<tr><td rowspan="2"><b>外部效度</b></td><td>單一城市(上海)、24 窗、中心城偏置;郊區 OSM/EULUC 品質未驗證</td><td>全部</td></tr>
<tr><td>AI 高度低估超高層、郊區高度可疑(楓涇)</td><td>形態指標、簽名雲</td></tr>
<tr><td><b>統計結論效度</b></td><td>E3 的 n=24:LOO R² 小樣本脆弱;heldout n=6 的 R² 無意義(只報 MAE);
permutation+FDR 控了假陽性但功效低</td><td>E3</td></tr></table>
<p>與前輪一脈相承的兩個老天花板原樣在場,未被本輪觸碰:<b>池稀釋</b>(v0.3 命名;E4 的語義剝離
是它的新表現形)與<b>度量盲區</b>(v0.4 命名;E4 的指標表仍是非空間統計)。</p>
"""

SEC10 = """
<h2 id="sec10">10 結論與下一步</h2>
<p>回到業主的一句話任務:「怎麼去找到分類、權力與城市形態的關係」。v05 的答案是一張
<b>有方向、有條件、有失敗記錄的地圖</b>:建築層級的「形→權」這條邊在誠實檢驗下接近於不存在
(E1);「類→權」這條邊可以運轉但只是文獻斷言的傳聲筒,且詞彙有指名的缺口(E2);
視窗層級的「形→權」斜邊只對開發資本一個維度部分導通,且導通的機制是時間而非類型(E3);
「權→形」的生成方向傳得動量體、傳不動身分(E4)。<b>四個「不完全成功」合起來,
比一個表面成功更精確地畫出了這個問題的真實邊界。</b></p>
<p>下一步(按槓桿率排序):</p>
<ol>
<li><b>建成年代資料</b>——三角經由時間閉合,而 Age 是最殘缺的欄位;任何更完整的年代源
(歷史影像、測繪檔案、房產登記年份)都直接抬升 E2/E3 的全鏈。</li>
<li><b>檔案校準 pilot(Whitehand 協議)</b>——選 1–2 個窗,用規劃申請/出讓記錄逐筆標註
「發起行動者×物質變更」,把先驗從文獻常數變成有 ground truth 的校準值(B 報告 §4.5 的原路線)。</li>
<li><b>先驗詞彙修補</b>——加遺產/國家保護通道(修外灘異常),把 informal 的承載從
「視窗 GFA 權重」改成更合適的觀測單元(它在 GFA 加權下結構性不可見)。</li>
<li><b>語義條件化的生成</b>——E4 證明 k=8 形態詞彙剝離身分;讓拼貼按語義條件抽池
(或 token 詞彙帶語義維度),「傳形不傳名」才有解。</li>
</ol>
"""

APPENDIX = f"""
<h2 id="appendix">附錄</h2>
<h3>A. 重跑指令(全部 seed 0、有快取)</h3>
<pre>python3 build_dataset.py            # dataset.parquet + windows.json + fig_e0_*
python3 exp1_recognition.py          # results_e1.json + fig_e1_*
python3 exp2_power.py                # results_e2.json + power_priors_v05.json + fig_e2_*
python3 exp3_triangle.py             # results_e3.json + fig_e3_*
python3 exp4_counterfactual.py       # results_e4.json + cache_gen/cf_*.parquet + fig_e4_*
python3 make_diagrams.py             # fig_d1/d2/d3
python3 audit_v05.py                 # 獨立復算(6 項 spot check,全部 confirmed)
python3 make_v05_report.py           # 本報告</pre>
<h3>B. 與前輪的關係</h3>
<p>v05 重用 v0.3 的 24 窗融合表/簽名/token 詞彙與 v0.4 的遮罩/typed 格網(凍結輸入,
read-only),在其上全部重新設計實驗;先驗沿用探針 v0.1 未動。前輪文件:
TYPOLOGY_FOUNDATION.html(A/B/C+提案)、PROBE_REPORT.html、V02/V03/V04_REPORT.html。</p>
<h3>C. 獨立稽核</h3>
<p>一個未參與實驗的 audit agent 以獨立程式碼復算:A13 權力剖面(4 維至小數 4 位吻合,
順帶全 24 窗 max |diff| 4.9e-5)、E1 留窗 balanced accuracy(120 樹復算差 0.1pp)、
橋 lowrise→house(1355/1357 完全一致)、E3 dev 維 LOO R²(差 0.007)、E4 DEV 畫布 FAR
(幾何重算一致)——<b>6/6 confirmed,無數字偏差</b>;兩處分母歧義(dist_water NaN 份額
0.520 全集 vs 0.536 標籤子集)已在文中統一採全集口徑。</p>
"""

# ================================================================ assemble
CSS = """
  :root { --ink:#1a1d23; --muted:#6b7280; --line:#e5e7eb; --bg:#fafaf8; --card:#fff;
          --accent:#0f5e63; --accent2:#b45309; --warn:#9f1239; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15.5px/1.75 "Avenir Next","PingFang TC","Noto Sans TC",system-ui,sans-serif; }
  .wrap { max-width:1060px; margin:0 auto; padding:46px 30px 90px; }
  header { border-bottom:3px solid var(--ink); padding-bottom:18px; margin-bottom:8px; }
  h1 { font-size:27px; margin:0 0 6px; }
  .sub { color:var(--muted); font-size:13.5px; }
  h2 { font-size:21px; margin:48px 0 14px; padding-left:12px; border-left:5px solid var(--accent); }
  h3 { font-size:16.5px; margin:28px 0 8px; color:var(--accent); }
  code { background:#eef0f3; padding:1px 5px; border-radius:4px; font-size:.85em;
         font-family:"SF Mono",Menlo,monospace; }
  pre { background:#f4f4f2; border:1px solid var(--line); border-radius:8px; padding:12px 16px;
        font-size:12.5px; line-height:1.55; overflow-x:auto; font-family:"SF Mono",Menlo,monospace; }
  table { width:100%; border-collapse:collapse; background:var(--card); font-size:13px;
          border:1px solid var(--line); margin:14px 0; }
  th { background:#f1f3f4; text-align:left; padding:7px 10px; border-bottom:2px solid var(--ink); }
  td { padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  figure { margin:20px 0; background:var(--card); border:1px solid var(--line);
           border-radius:8px; padding:12px; }
  figure img { width:100%; height:auto; display:block; border-radius:4px; }
  figcaption { font-size:13px; color:var(--muted); margin-top:8px; }
  .abstract { background:#f0f7f7; border:2px solid var(--accent); border-radius:10px;
              padding:6px 22px 14px; margin:26px 0; }
  .choice { background:#fdf6ec; border-left:5px solid var(--accent2); border-radius:0 8px 8px 0;
            padding:10px 16px; margin:14px 0; font-size:13.8px; }
  .gapbox { background:#f5f5f4; border:1.5px dashed #9ca3af; border-radius:8px;
            padding:10px 18px; margin:16px 0; font-size:13.8px; }
  .failbox { background:#fff5f5; border-left:5px solid var(--warn); border-radius:0 8px 8px 0;
             padding:10px 16px; margin:14px 0; font-size:13.8px; }
  .toc { background:var(--card); border:1px solid var(--line); border-radius:8px;
         padding:12px 22px; margin:18px 0; font-size:13.5px; columns:2; }
  .toc a { color:var(--accent); text-decoration:none; display:block; margin:3px 0; }
  ol.refs { font-size:13px; color:#374151; } ol.refs li { margin:4px 0; }
  ul { padding-left:22px; } li { margin:4px 0; }
"""

n_feat = len(J["e0"]["new_columns"])
fills = {
    "@N_BUILDINGS@": f"{J['e0']['n_rows']:,}",
    "@N_FEATURES@": f"{17 + n_feat}",
    "@CLASS_TABLE@": (
        "<table><tr><th>語義類</th><th>n</th><th>語義類</th><th>n</th></tr>" + "".join(
            f"<tr><td>{a}</td><td>{J['e0']['semantic_counts'][a]:,}</td>"
            f"<td>{b}</td><td>{J['e0']['semantic_counts'][b]:,}</td></tr>"
            for a, b in [("residential_estate", "residential_lowrise"),
                         ("commercial_retail", "office_finance"),
                         ("public_admin", "education"),
                         ("industrial", "healthcare")])
        + f"<tr><td>unlabeled(剔除於 E1/E2)</td><td>{J['e0']['semantic_counts']['unlabeled']:,}</td>"
          f"<td>家族:slab/mid/house/tower/large</td>"
          f"<td>{J['e0']['family_counts']['slab']:,}/{J['e0']['family_counts']['midrise_other']:,}/"
          f"{J['e0']['family_counts']['house']:,}/{J['e0']['family_counts']['tower']:,}/"
          f"{J['e0']['family_counts']['large_footprint']:,}</td></tr></table>"),
    "@DATASET_NOTES@": (
        "兩個結構性事實在組裝時即確立:dist_water 在 52% 建築為缺值(24 窗中 10 窗無水體,"
        "缺值=「窗內無此類」而非未知);Age 僅 39.2% 覆蓋且 1984/85 為截斷地板"
        "(曹楊 1950 年代工人新村的 residential_estate GFA 僅 6% 標為 pre-1985)——年代欄的殘缺將貫穿後續所有實驗。"),
    "@ABSTRACT_FINDINGS@": (
        "(1) 建築層級,形態辨識在誠實的留窗轉移下塌回多數類基準附近,權力語義類"
        "(政府/教育/醫療)recall ≤1.2%、healthcare 恰為 0——「形態看不見權力」在 14k 尺度被強化確認,"
        "舊的 72% 樂觀被診斷為標籤洩漏;(2) 權力鏈在 24 窗跑通且橋尺度穩定,但五個零年代郊區窗中四個的"
        "剖面是先驗回聲,且先驗詞彙缺「遺產/國家保護」通道(外灘讀成 24 窗中政府權重最低,顯然荒謬);"
        "(3) 視窗層級只有開發資本權重能從形態穩健回收(LOO R² 0.46),且訊號掛在 PC2"
        "(空曠尖峰 vs 密實毯)而非預期的強度軸 PC1——三角經由「時間」而非「類型」閉合;"
        "(4) 反事實生成傳得動垂直度與 house 性,傳不動機構身分(GOV 配置要求 47.8% 公共類語義,"
        "渲染後僅剩 7.5%)。"),
}

static_parts = []
for sec in [ST.ABSTRACT, ST.SEC1, ST.SEC2, ST.SEC3]:
    s = sec
    for k, v in fills.items():
        s = s.replace(k, v)
    static_parts.append(render_static(s))

TOC = """
<div class="toc">
<a href="#sec1">1 研究問題與背景</a><a href="#sec2">2 資料</a><a href="#sec3">3 方法總覽</a>
<a href="#sec4">4 E1 建築層級辨識</a><a href="#sec5">5 E2 權力歸屬鏈</a>
<a href="#sec6">6 E3 三角斜邊測試</a><a href="#sec7">7 E4 反事實生成</a>
<a href="#sec8">8 綜合討論</a><a href="#sec9">9 侷限與效度威脅</a>
<a href="#sec10">10 結論與下一步</a><a href="#refs">參考文獻</a><a href="#appendix">附錄</a>
</div>"""

html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v05 — 分類、權力與城市形態:一次誠實的探索</title><style>{CSS}</style></head><body><div class="wrap">
<header>
<h1>分類、權力與城市形態:以上海開放資料的一次誠實探索</h1>
<div class="sub">typology_foundation / v05 · 2026-06-12 · 資料集 14,527 棟 × 24 個 1 km 視窗 ·
四個實驗 + 獨立稽核(6/6 confirmed)· 全程 seed 0 可重跑 ·
研究任務(業主原話):「我們有一些城市數據,怎麼去找到分類、權力與城市形態的關係的探索。
不一定要真的探索成功,也可以是失敗,然後變成一些有趣的實驗去呈現一些面向。」</div>
</header>
{static_parts[0]}
{TOC}
{static_parts[1]}
{static_parts[2]}
{static_parts[3]}
{SEC4}
{SEC5}
{SEC6}
{SEC7}
{SEC8}
{SEC9}
{SEC10}
{render_static(ST.SEC_REFS)}
{APPENDIX}
<footer style="margin-top:50px;border-top:2px solid var(--line);padding-top:14px;color:var(--muted);font-size:12.5px">
v05 · Urban-Stakeholder-Workshop / typology_foundation · 凍結輸入:v0.3 融合表+簽名、v0.4 遮罩+typed 格網、
探針 v0.1 先驗 · 所有圖表由 v05/ 內腳本生成,數字經獨立 audit agent 復算確認。</footer>
</div></body></html>"""

out = os.path.join(TF, "V05_REPORT.html")
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out)/1e6:.1f} MB")
