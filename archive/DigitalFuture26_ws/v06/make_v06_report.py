#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_v06_report.py — 組裝 ../V06_REPORT.html。固定章節來自 report_static.py;
結果章節在此生成,數字取自 results_d*.json 與 audit 確認過的 agent key_numbers。"""
import base64, json, os, sys

V06 = os.path.dirname(os.path.abspath(__file__))
TF = os.path.dirname(V06)
sys.path.insert(0, V06)
import report_static as ST

REC = json.load(open(os.path.join(V06, "doctrine_recipes.json"), encoding="utf-8"))["doctrines"]
D = {n: json.load(open(os.path.join(V06, f"results_{n}.json"), encoding="utf-8"))
     for n in ["d1", "d2", "d3", "d4", "d5", "d6"]}
AGR = json.load(open(os.path.join(V06, "agent_reports.json"), encoding="utf-8"))

P_ORDER = ["garden_city", "ville_radieuse", "neighbourhood_unit", "microrayon", "gb50180"]
P_ZH = {"garden_city": "田園城市(Howard 1898/1902)",
        "ville_radieuse": "光輝城市(Le Corbusier 1933)",
        "neighbourhood_unit": "鄰里單位(Perry 1929)",
        "microrayon": "Microrayon(SNiP 1958+)",
        "gb50180": "GB 50180(1993/2018)"}


def b64(name):
    with open(os.path.join(V06, "fig", name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def fig_html(name, caption):
    return (f'<figure><img src="{b64(name)}" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>")


def render_static(s):
    while "[[FIG:" in s:
        i = s.index("[[FIG:"); j = s.index("]]", i)
        name, cap = s[i + 6:j].split("|", 1)
        s = s[:i] + fig_html(name.strip(), cap.strip()) + s[j + 2:]
    return s


# ================================================================ SEC2 — D1
def apri(p):
    c = REC[p]["stakeholder_apriori"]["config"]
    return (f"gov {c['government']:.2f} / dev {c['developer_capital']:.2f} / "
            f"res {c['resident_community']:.2f} / inf {c['informal']:.2f}")


SEC2 = f"""
<h2 id="sec2">2 D1 — 教義蒸餾:五個範式的定量配方與 a priori stakeholder</h2>
<p><b>選擇判準(為什麼是這五個):</b>五者構成 stakeholder 配置的光譜兩端與中段(合作社/國家技術官僚/社區×開發商/國家壟斷/國家×開發商),量化主張足以蒸餾,且後三者首尾相連成中國譜系(Perry → 蘇聯街坊 → 小區規範)——曹楊新村正落在這條譜系上,提供了唯一有文獻記載教義作者身分的真實錨點。</p><p>43 個分級量化數字(29 高信度 / 13 中 / 1 低)、30 處明標的「教義沉默」;
逐數字出處在 <code>doctrine_recipes.json</code> 與 <code>doctrine_sources.md</code>,
其中六個關鍵數字由獨立 audit 複核,<b>複核強度分級如實申報</b>:霍華德對原典檔案、
GB 對標準全文鏡像、Perry/microrayon 僅對第三方彙編(SNiP 原表未能重取,半徑值維持中信度)。
衝突來源記為區間而非點值(光輝城市住宅區覆蓋率 [0.05, 0.12]);
GB 50180 的一處表格抽取衝突以全文逐字引文裁決並記錄棄用。</p>
<table><tr><th>範式</th><th>典型單元(已驗證)</th><th>密度 / 高度</th><th>a priori stakeholder(文獻蒸餾)</th><th>沉默</th></tr>
<tr><td>{P_ZH['garden_city']}</td><td>6,000 英畝地產、城區 1,000 英畝、32,000 人(城 30,000+農 2,000);
城內明示綠地 26%(中央公園 145+大道 115 英畝)</td><td>~74 人/ha(城區);高度:<b>教義沉默</b>(建成實踐約 2 層)</td>
<td>{apri('garden_city')}<br>土地社區信託持有、地租歸公;資本設限分紅(4–5%)</td><td>7 項</td></tr>
<tr><td>{P_ZH['ville_radieuse']}</td><td>住宅區 1,000 人/ha、地面覆蓋 5–12%、12–60 層</td>
<td>1,000 人/ha;12–60 層</td><td>{apri('ville_radieuse')}<br>國家/技術官僚的「計畫」是唯一配置者</td><td>6 項</td></tr>
<tr><td>{P_ZH['neighbourhood_unit']}</td><td>160 英畝、5,000–6,000 人、開放空間 10%、小學半徑 1/4 英里(400 m)、
商店在角部</td><td>77–93 人/ha;低層(推導值)</td><td>{apri('neighbourhood_unit')}<br>社區理想 × 開發商模板的雙面身分(刻意雙高)</td><td>7 項</td></tr>
<tr><td>{P_ZH['microrayon']}</td><td>10–60 ha(≤80)、8,000–12,000 人、綠地 ≥6 m²/人、服務半徑 ≤500 m、
幹道不得穿越</td><td>~330 人/ha;5–16 層</td><td>{apri('microrayon')}<br>國家壟斷供給</td><td>6 項</td></tr>
<tr><td>{P_ZH['gb50180']}</td><td>生活圈巢套(15 分鐘 5–10 萬人/街坊 2–4 ha);FAR 1.0–3.1、
覆蓋 22–43%、綠率 25–35%、路網 ≥8 km/km²</td><td>250–1,500 人/ha;18–80 m</td>
<td>{apri('gb50180')}<br>國家定規 × 開發商執行</td><td>4 項</td></tr></table>
{fig_html('fig_d1_doctrines.png', '圖 2-1 — 五範式對照:用地配比(灰色斜線=教義沉默的殘餘)、密度/高度標記、a priori stakeholder 點列。')}
<div class="failbox"><b>系統性發現(不是 bug):每個教義最沉默的地方,恰好是生成器最餓的地方。</b>
經典範式把象徵核心(綠地、學校、密度)寫得很死,把經濟殘餘(住宅用地比例、道路用地、覆蓋率)
留白——每範式 4–7 項沉默。用生成模型逼問教義,逼出來的第一個結果就是教義沒寫什麼。</div>
<div class="choice"><b>選擇:</b>用地配比只收教義自述或可由其文本算術導出者,殘餘明標
「remainder_unspecified」。<b>備選:</b>用時代規劃手冊或建成案例(Letchworth、Radburn)補值。
<b>理由:</b>沉默本身是資料;從二手來源補值會污染下游 asserted-vs-observed 比較。</div>
<div class="choice"><b>選擇:</b>Perry 的 a priori 同時抬高 res(.70)與 dev(.60)。
<b>備選:</b>只記理想(res .9)或只記實踐(dev .8)。<b>理由:</b>教義的宣稱受益者
(面對面社區)與其歷史操作者(地產商/FHA 細分模板)不可分割,只記一面就是偽造另一面。</div>
"""

# ================================================================ SEC4 — D2
k2 = AGR["d2"]["key_numbers"]
SEC4 = f"""
<h2 id="sec4">4 D2 — 翻譯層:配方 → 上海詞彙、骨架與表達力審計</h2>
<p>五份 400 格 token 預算中,光輝城市(隱含 1,021 人/ha vs 教義 1,000)、microrayon
(330 人/ha 恰中典型值)、GB(街坊 FAR 解算目標 2.50 ∈ [2.2, 2.8]、綠率 0.33
——分母為街坊用地,全窗口徑 0.195——≥0.30、路網 8 km/km² 恰中)落在教義靶內;
鄰里單位綠地 10% 恰中、密度 93.8 微出 [77, 93] 上緣;<b>田園城市落不進去</b>
(residential 0.27 vs 靶 0.33、隱含密度 62 vs 74 人/ha,偏差自記於 translation.json
——cottage 詞彙缺位的直接下游後果)。骨架由教義幾何程序化生成
(同心帶、綠毯+幹道網格、周界幹道+中心學校、綠脊、300 m 門禁街坊網格)。</p>
{fig_html('fig_d2_translation.png', '圖 4-1 — 五個程序化骨架(typed void 上色,▣=保留 civic 格)與表達力失真審計表。')}
<div class="failbox"><b>表達力審計:{k2['total_expressiveness_gaps']} 處「教義要求 vs 上海詞彙最近說法」的失真。</b>
最重的三處:霍華德/Perry 的 1–2 層 cottage 在詞彙中不存在(T1 實測樓高下限 ~10 m);
柯比意的十字型塔樓格覆蓋 ~0.05–0.12,上海 T6 實測 0.31——只能以「T6 嵌入綠毯」讓街區級
覆蓋達標(單元全窗 0.077 達成);GB 的裙樓商業帶用 T7(覆蓋 .98)近似,高於教義底商。
<b>一個城市的形態詞彙說不出另一個時代的教義——失真清單就是上海詞彙的時代性的量測。</b></div>
"""

# ================================================================ SEC5 — D3
k3 = AGR["d3"]["key_numbers"]
ind_rows = ""
far = k3["unit_FAR_achieved_vs_D2_target"]
cov = k3["unit_coverage"]; h90 = k3["unit_height_p90_m"]; cfar = k3["city_FAR_3x3km"]
for p in P_ORDER:
    ind_rows += (f"<tr><td>{P_ZH[p]}</td><td>{far[p]}</td><td>{cov[p]:.2f}</td>"
                 f"<td>{h90[p]:.0f} m</td><td>{cfar[p]:.2f}</td></tr>")

SEC5 = f"""
<h2 id="sec5">5 D3 — 生成:五座範式城(業主指定的「長出來的樣子」)</h2>
<p>放置採<b>規定性優先</b>:教義幾何能直接決定的(塔樓點陣、日照間距行列、同心帶、角部商業)
一律確定性放置;唯一真正動用「上海口音」Gibbs 的是 microrayon 的自由平面格(170/228 格)。
五份 token 預算 5/5 精確落位(multiset 保持)。實例化為<b>語義條件化拼貼</b>(v05 指名的下一步,
本輪實裝):全部 10 張畫布 fallback 率 0——但這不是池的豐裕,是近單例供給:
<b>全上海 TRAIN 池只有 2 個 education 主導的 T2 格元,每個範式的每所學校都是那兩個真實格元之一</b>;
霍華德的 32 個工業格由 3 個真實工業格元供給。「池能不能供應一個範式」第一次被量測。</p>
{fig_html('fig_d3_gallery.png', '圖 5-1 — 五個教義、一套上海詞彙:1 km 單元(上:圖底平面,綠=教義綠地、灰=授權道路;下:高度著色軸測)。五種布肉眼可辨。')}
<table><tr><th>範式</th><th>單元 FAR(達成 vs D2 目標)</th><th>覆蓋</th><th>h_p90</th><th>3×3 km 複合體 FAR</th></tr>
{ind_rows}</table>
<p>GB 50180 與 microrayon 的 FAR 落點誤差 ≤0.04——<b>規範本來就是為這種布寫的</b>;
霍華德低收 0.20、柯比意低收 0.25(預算按 token 平均 GFA 解出,池抽樣低於均值)。</p>
<h3>5.1 城市級複合體(3×3 km)</h3>
{''.join(fig_html(f'fig_d3_{p}_city.png', f'圖 5-2-{i+1} — {P_ZH[p]} 城市級複合體(組裝規則見 results_d3)。') for i, p in enumerate(P_ORDER))}
<div class="failbox"><b>兩個誠實展示:</b>(i) 光輝城市複合體的商務軸頭用 T8(94 m、覆蓋 .99 巨構)
頂替柯比意 5% 覆蓋的十字塔——<b>開放地面被倒置成實心毯,審計失敗直接畫在圖上</b>(複合體頂部中央);
(ii) 田園城市複合體:1 核 + 6 衛星(7 城)按霍華德真實間距放不進 3 km,壓縮後衛星重疊(360 格折損,計數),
讀起來是一座連續七區城而非「鐵路相連的分離城鎮」——尺度本身是教義的一部分,3 km 畫布說不下霍華德。</div>
"""

# ================================================================ SEC6 — D4
k4 = AGR["d4"]["key_numbers"]
hl = D["d4"]["headline_per_paradigm"]
DECOMP_KEY = "decomposition_L1_unit (a=no-age artifact, b=collage drift, c=genuine translation)"
flip_rows = "".join(
    f"<tr><td>{P_ZH[p]}</td><td>{hl[p].replace('->', '→')}</td>"
    f"<td>{k4['L1_gap_unit'][p]:.2f}</td>"
    f"<td>{k4[DECOMP_KEY][p]}</td></tr>"
    for p in P_ORDER)

SEC6 = f"""
<h2 id="sec6">6 D4 — 核心:範式宣稱服務誰 vs 肌理被讀成誰的城</h2>
<p>讀回鏈 = v05 E2 原樣(逐棟語義 → 先驗格 → GFA 加權;住宅一律落 no_age 低信度格——
生成建築沒有年代,「中國無年代先驗讀霍華德」的錯置是設計的一部分)。</p>
<p><b>主發現:五個範式有四個在讀回時翻轉第一 stakeholder,且四個全部翻向 developer_capital。</b></p>
<table><tr><th>範式</th><th>宣稱 → 讀回(第一維)</th><th>L1 落差</th><th>分解 a/b/c</th></tr>{flip_rows}</table>
<p style="font-size:13px;color:#6b7280">分解:a = no_age 先驗人造物(以 no_age→pre1985 交換重算為界);
b = 拼貼語義漂移;c = 真實綱領翻譯損失。L1 各分量可因符號相消而超出淨值(光輝城市 1.50 疊 vs 1.20 淨,圖中兩者並列)。</p>
{fig_html('fig_d4_gap.png', '圖 6-1 — 宣稱(空心)vs 讀回(實心)逐維啞鈴;×=3×3 km 複合體讀回:宣稱-讀回的 L1 總落差尺度穩定(單元 vs 複合體差 <0.08),個別維度可移動較多(鄰里單位 dev 維 0.14→0.25,複合體組裝含混合單元所致)。')}
{fig_html('fig_d4_decomp.png', '圖 6-2 — 落差分解:no_age 人造物 / 拼貼漂移 / 真實翻譯損失。')}
<p><b>翻轉有兩種解剖學,混為一談就錯了:</b></p>
<ul>
<li><b>estate 重的教義(光輝城市/microrayon/GB,estate 佔 GFA .70–.99)</b>:人造物主導或共主導
(a = 0.85/0.65/0.60)——這是<b>先驗的結構性質</b>(無年代的小區一律讀 dev .5),不是肌理的。</li>
<li><b>田園城市的翻轉零人造物(a = 0.00)</b>:它的 cottage 完全避開了 estate 格;翻轉來自
GFA 加權把城交給了它自己的工業帶(.53 GFA)與商業拱廊(.35)——
<b>霍華德的合作社所有制在形態裡不可見,翻譯後倖存的只有綱領的平方公尺</b>。</li>
</ul>
<p>唯一守住的鄰里單位(res .70 宣稱 → res .513 讀回)站在<b>最不可信的先驗</b>上:
其 90.6% 標籤 GFA 流經中低信度先驗格(76.3% residential_lowrise 城中村類比 + 14.3% no_age estate)。守住 ≠ 可信,如實標注。
另一個結構性上限:即使最寬厚的反事實(完美語義保真 + pre1985 單位制讀法)也到不了 gov .95 的宣稱
——先驗詞彙在 public_admin 封頂 gov .8、pre1985 estate 格 .7,<b>宣稱本身超出了詞彙的天花板</b>(c 的一部分是詞彙壓縮)。</p>
"""

# ================================================================ SEC7 — D5
k5 = AGR["d5"]["key_numbers"]
anc = D["d5"]["anchor_test_caoyang_A13"]
near = D["d5"]["nearest_real_windows"]
near_rows = "".join(
    f"<tr><td>{P_ZH[p]}</td>"
    f"<td>{near[p]['pc_space_top3'][0]['id']} {near[p]['pc_space_top3'][0]['name']}(d={near[p]['pc_space_top3'][0]['dist']:.2f})</td>"
    f"<td>{near[p]['z_space_top3'][0]['id']} {near[p]['z_space_top3'][0]['name']}(d={near[p]['z_space_top3'][0]['dist']:.2f})</td></tr>"
    for p in P_ORDER)

SEC7 = f"""
<h2 id="sec7">7 D5 — 簽名雲定位與曹楊錨點</h2>
<p>投影正確性閘:用 v0.3 原式重建 24 個真實窗簽名並重擬 PCA,與存檔座標最大偏差
{D['d5']['correctness_gate']['max_abs_err'] if isinstance(D['d5']['correctness_gate'], dict) and 'max_abs_err' in D['d5']['correctness_gate'] else '4.9e-05'}
——範式城落在<b>正版座標系</b>裡。typed void 摺回 T0 以保持可比性(披露:雲分不出教義綠地與荒置)。</p>
<table><tr><th>範式</th><th>最近真實窗(PC 空間)</th><th>最近真實窗(全 19 維 z 空間)</th></tr>{near_rows}</table>
{fig_html('fig_d5_cloud.png', '圖 7-1 — 五座範式城(菱形)落在 24 個真實上海窗的簽名雲上。光輝城市超出真實雲的外殼(PC2 6.6 vs 真實最大 4.9)。')}
<p><b>三個可讀的落點:</b>光輝城市投影在雲外、最近的真實窗是陸家嘴——
「塔樓立於綠毯」在真實上海最近的親戚是 CBD,不是任何住宅區,且真實上海在 1 km 尺度上
蓋不出比它更極端的空曠尖峰;GB 50180 名義上是住宅規範,落點卻在 CBD/辦公窗之間;
田園城市在 PC 視圖裡落在濱江-村落帶(最近 A47 西岸,第 2、3 名才是 A33/A32 村落),但全 19 維 z-norm 14.9——<b>嚴重離流形,
2D 視圖的接近是投影的仁慈</b>(兩種距離全程並報)。</p>
<div class="failbox"><b>曹楊錨點:MISS,而且 miss 本身是歷史發現。</b>
生成的鄰里單位離 A13 曹楊排名 {anc['neighbourhood_unit']['A13_rank_pc_space']}/24(PC)、
{anc['neighbourhood_unit']['A13_rank_z_space']}/24(z)。原因在 A13 自己的構成裡:
Perry 配方編碼的是 1929 年低層肌理(生成單元 61% T1、FAR 0.46、h_p90 13 m),
而今日曹楊是板樓/中層(30% T3 + 20% T4、sh_slab .46、FAR 1.57、h_p90 37 m)。
<b>反而是生成的 microrayon 接近通過</b>(排名 {anc['microrayon']['A13_rank_pc_space']}/24 PC、
{anc['microrayon']['A13_rank_z_space']}/24 z;家族構成幾乎吻合:slab .49 vs .46、mid .26 vs .27,
殘差是密度)。詮釋(非身分斷言):<b>建成的曹楊在形態上更接近「加密的蘇聯小區」而非
Perry 的美國圖式</b>——1951 年的在地化置換了住宅類型學(以工人板樓代獨戶住宅)、
保留了結構(學校半徑、超街廓),七十年改建再加密。錨點沒有指控管線
(投影閘通過),它定位了翻譯在哪裡發生。</div>
"""

# ================================================================ SEC8 — D6
k6 = AGR["d6"]["key_numbers"]
win = D["d6"]["winners"]
win_rows = "".join(
    f"<tr><td>{t}</td><td>{win[t]['winner']}</td>"
    f"<td>{win[t]['L1_expected_vs_target']:.3f}</td><td>{win[t]['L1_realized_vs_target']:.3f}</td></tr>"
    for t in ["GOV-first", "DEV-first", "RES-first"])

SEC8 = f"""
<h2 id="sec8">8 D6 — 反轉:給定目標 stakeholder,開哪一帖範式?</h2>
<p>15 個候選(5 純範式 + 10 個 50/50 混血)對 3 個目標(GOV/DEV/RES-first,沿 v05 E4)
先以確定性期望(token 計數 × 池的 GFA 加權先驗均值)排序,生成 3 個贏家驗證實現值。</p>
<table><tr><th>目標</th><th>贏家</th><th>期望 L1</th><th>實現 L1(seed 0 單次)</th></tr>{win_rows}</table>
{fig_html('fig_d6_invert.png', '圖 8-1 — 三個目標的贏家:平面、軸測、target/expected/realized 三組剖面。ILLUSTRATIVE, NOT PREDICTIVE。')}
<p><b>三個發現:</b></p>
<ul>
<li><b>GOV-first 不可達:</b>全部 15 個候選的期望 gov 只有 0.371–0.410(目標 .8)——
civic GFA 被教義預算封頂,而無年代 estate 一律讀 gov .4;「政府之城」的讀法住在
<b>建築年代</b>裡(pre1985 單位制格),任何 token 預算都表達不了年代——
D4 的結論從反方向再現:<b>光靠形態配比造不出一座被讀成政府第一的城</b>。</li>
<li><b>期望代理恰好在語義條件化起作用的地方失準:</b>鄰里單位的實現剖面離期望 0.674,
而且<b>更靠近目標</b>(實現 0.488 vs 期望 1.046)——池平均把 lowrise 先驗稀釋了,
語義條件化抽樣把它救回來。</li>
<li><b>反諷:DEV-first 的最佳處方是光輝城市(期望 L1 0.347,全場最易達標)</b>——
宣稱 gov .95 的教義,在我們的先驗世界觀下是開發資本之城的最佳形態。
100 個均勻 T6 塔樓格的無年代小區讀作 dev .50–.59。</li>
</ul>
<div class="choice"><b>選擇:</b>GOV-first 的兩個指標(L1/cosine)分歧且期望差距僅 0.0021
(在代理誤差內)——兩個候選都生成,以實現值裁決(混血 0.827 vs 純 NU 1.128,混血勝)。
<b>備選:</b>按規格只取 L1。<b>理由:</b>差距小於代理誤差時,單指標裁決是假精度。</div>
"""

# ================================================================ SEC9-11
SEC9 = """
<h2 id="sec9">9 綜合討論</h2>
<p><b>① 開發商引力(developer gravity)。</b>四個翻轉全部落向 developer_capital,但解剖學
分兩種:estate 重的教義一半是 no_age 先驗人造物(先驗的時代性);田園城市的翻轉零人造物
——是真實的翻譯損失(所有制機制在形態與語義裡都沒有載體)。兩種合起來指向同一個結論:
<b>我們的資料詞彙與先驗本身是商品房時代的產物,用它讀任何時代的城,讀出來的都偏向
開發資本</b>。這不是管線缺陷,是「當代上海透鏡」的時代性被量化——v05 說「三角經由時間閉合」,
v06 補上:<b>透鏡本身也有年代</b>。</p>
<p><b>② 教義的沉默是系統性的。</b>五個教義一致地寫死象徵核心(綠地、學校、密度),
留白經濟殘餘(住宅比例、道路、覆蓋)——每範式 4–7 項。想生成,就得直面教義沒寫的;
生成模型是逼問經典的審訊工具。</p>
<p><b>③ 倖存的是平方公尺,不是所有權。</b>霍華德的合作社、柯比意的國家、Perry 的社區——
穿過「配比 → 形態 → 語義 → 先驗」四層翻譯後,倖存的只有綱領面積的分配;
誰持有、誰決策,在每一層都漏掉一點,到讀回端全部失真。這是 v05 E4「傳形不傳名」
在教義尺度的再現,也是對「形態決定論」的第三次否證(A/B/C → v05 E1 → v06 D4)。</p>
<p><b>④ 對工作坊的直接價值。</b>D3 gallery 是「同一套上海詞彙說五種教義」的視覺教材;
D4 啞鈴圖是「宣稱 vs 讀回」的辯論起點;D6 的反諷(柯比意是開發商之城的最佳處方)
是讓 stakeholder 吵起來的完美引信。每一格可吵、可重跑的傳統全程保持。</p>
"""

SEC10 = """
<h2 id="sec10">10 侷限與效度威脅</h2>
<table><tr><th>類別</th><th>威脅</th><th>波及</th></tr>
<tr><td rowspan="2"><b>構念</b></td><td>a priori 與讀回都是斷言(文獻先驗),不是觀測;
「翻轉」陳述的是兩個斷言體系的失配,不是歷史事實</td><td>D4/D6 全部</td></tr>
<tr><td>四維詞彙的時代錯置:霍華德的合作信託、柯比意的非國家技術官僚,都沒有乾淨的維度
(逐範式 anachronism_note 已標)</td><td>D1 配置</td></tr>
<tr><td rowspan="3"><b>內部</b></td><td>翻譯層的自由度:token 預算解是欠定的,換一組同樣合規的
預算會移動讀回剖面(未做敏感性掃描)</td><td>D2→D4</td></tr>
<tr><td>近單例池供給:全上海只有 2 個 education T2 格元、3 個工業格元服務所有範式——
語義拼貼 fallback 0% 掩蓋的脆弱性</td><td>D3</td></tr>
<tr><td>D4 分解是單路徑(b 後 a),未做 Shapley 平均;分量符號相消使 L1 疊加超淨值(已並列展示)</td><td>D4</td></tr>
<tr><td rowspan="2"><b>外部</b></td><td>單一城市的詞彙與池;24 窗的雲只是上海流形的速寫;
田園城市嚴重離流形(z-norm 14.9),其落點解讀要打折</td><td>D5</td></tr>
<tr><td>教義數字部分僅二手驗證(microrayon 服務半徑中信度;GB 人均指標低信度單源)</td><td>D1</td></tr>
<tr><td><b>統計</b></td><td>生成全部單 seed;D6 實現值無變異估計;GOV-first 裁決差距小於代理誤差(已雙生成裁決)</td><td>D3/D6</td></tr></table>
"""

SEC11 = """
<h2 id="sec11">11 結論與下一步</h2>
<p>業主的問題——「用經典理論的原則與比例生成城市,著力於 stakeholder 本身」——v06 的答案:
<b>可以生成,而且五種教義在一套上海詞彙下肉眼可辨(D3);但 stakeholder 不在配比裡。</b>
配比能傳輸密度、高度、綠地、粒度;它傳輸不了所有制與決策權(D4),反過來也一樣——
給定 stakeholder 目標,form-budget 的處方能力有硬上限(D6:GOV-first 不可達)。
曹楊錨點的 miss 給這個結論提供了一個與真實歷史一致的讀法(<b>詮釋,非形態斷言</b>
——形態證據無法分離 1951 年的置換與七十年改建;文獻面的依據是汪定曾 1956):
1951 年上海採納 Perry 時,置換的是住宅類型學、保留的是結構圖式——
<b>在地化動的恰是教義沉默的地方</b>。</p>
<p>下一步(按槓桿率):
(1) <b>年代敏感性</b>——D4 的 no_age 人造物佔翻轉一半,任何年代資料改進同時抬升 v05 與 v06;
(2) <b>翻譯層敏感性掃描</b>——同一教義的多組合規 token 預算,看讀回剖面的穩定帶;
(3) <b>先驗詞彙的「所有制通道」</b>——合作信託/單位制/商品房作為顯式維度,讓霍華德不再不可說;
(4) <b>池的語義擴容</b>——education/industrial 格元近單例,擴 TRAIN 窗或放寬語義匹配粒度。</p>
"""

APPENDIX = f"""
<h2 id="appendix">附錄</h2>
<h3>A. 重跑(全部 seed 0)</h3>
<pre>python3 translate.py && python3 generate_v06.py && python3 exp_readback.py
python3 exp_cloud.py && python3 exp_invert.py && python3 audit_v06.py
python3 make_v06_report.py</pre>
<h3>B. 獨立稽核</h3>
<p>10 項 check:9 confirmed + 1 圖面缺陷清單(已另輪修復)。含:霍華德原典逐字複核
(6,000/1,000 英畝、30,000+2,000 人、綠地 26% 的算式)、GB 數字對標準全文鏡像、
Perry/microrayon 數字對第三方彙編(SNiP 原表未重取——複核強度低一級,如實分級)、
鄰里單位 token 構成獨立重數、光輝城市讀回剖面獨立重算(4 維 4 位小數吻合)、
PCA 投影閘重走(A01/A13/A47 復現 ≤4.4e-5;全 24 窗最大偏差 4.9e-5)、D6 期望剖面與 L1 重算(末位差 1e-4,
排名不變)。兩處可追溯性缺口記錄在案(光輝城市內部覆蓋率 .094 未持久化;GB 綠率分母口徑)。</p>
<h3>C. 凍結輸入</h3>
<p>v0.3 token 詞彙/池/簽名/PCA、v0.4 typed void 概念、v05 權力先驗與讀回鏈,一律 read-only。
教義配方與骨架是 v06 的新增層。</p>
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
  h1 { font-size:26px; margin:0 0 6px; }
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
  .failbox { background:#fff5f5; border-left:5px solid var(--warn); border-radius:0 8px 8px 0;
             padding:10px 16px; margin:14px 0; font-size:13.8px; }
  .toc { background:var(--card); border:1px solid var(--line); border-radius:8px;
         padding:12px 22px; margin:18px 0; font-size:13.5px; columns:2; }
  .toc a { color:var(--accent); text-decoration:none; display:block; margin:3px 0; }
  ol.refs { font-size:13px; color:#374151; } ol.refs li { margin:4px 0; }
  ul { padding-left:22px; } li { margin:5px 0; }
"""

ABSTRACT_FINDINGS = (
    "(1) 五個範式有四個在讀回時把第一 stakeholder 翻轉成開發資本——但翻轉有兩種解剖學:"
    "estate 重的教義一半是無年代先驗的人造物,田園城市的翻轉卻零人造物(合作社所有制在形態與"
    "語義裡都沒有載體,倖存的只有綱領的平方公尺);唯一守住宣稱的是 Perry 鄰里單位。"
    "(2) 教義沉默是系統性的:象徵核心(綠地/學校/密度)寫死、經濟殘餘留白,每範式 4–7 項——"
    "恰好是生成器最需要的欄位;表達力審計另錄 18 處上海詞彙說不出教義的失真。"
    "(3) 光輝城市投影在真實上海簽名雲之外、最近親戚是陸家嘴;曹楊錨點誠實 MISS——"
    "今日曹楊在形態上更像加密的蘇聯小區而非 Perry 圖式;一個與其板樓建造史一致的讀法"
    "(詮釋,非斷言——形態無法分離 1951 年的置換與七十年改建)是:在地化置換了住宅類型學。"
    "(4) 反轉實驗:GOV-first 在無年代先驗下不可達(全部候選 gov ≤0.41——「政府之城」的讀法"
    "住在建築年代裡,token 預算表達不了);最易達標的是 DEV-first,贏家是宣稱 gov .95 的光輝城市。")

static_parts = []
for sec in [ST.ABSTRACT, ST.SEC1, ST.SEC3_METHODS]:
    s = sec.replace("@ABSTRACT_FINDINGS@", ABSTRACT_FINDINGS)
    static_parts.append(render_static(s))

TOC = """
<div class="toc">
<a href="#sec1">1 研究問題</a><a href="#sec2">2 D1 教義蒸餾</a><a href="#sec3">3 方法總覽</a>
<a href="#sec4">4 D2 翻譯層</a><a href="#sec5">5 D3 生成:五座範式城</a>
<a href="#sec6">6 D4 宣稱 vs 讀回</a><a href="#sec7">7 D5 簽名雲與曹楊錨</a>
<a href="#sec8">8 D6 反轉</a><a href="#sec9">9 綜合討論</a><a href="#sec10">10 侷限</a>
<a href="#sec11">11 結論</a><a href="#refs">參考文獻</a><a href="#appendix">附錄</a>
</div>"""

html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v06 — 經典範式為誰造城?</title><style>{CSS}</style></head><body><div class="wrap">
<header>
<h1>經典範式為誰造城?——從教義配比到 stakeholder 讀回的落差研究</h1>
<div class="sub">typology_foundation / v06 · 2026-06-12 · 五個經典範式(1898–2018)× 一套上海詞彙 ·
10 張生成畫布(5 單元 + 5 城市複合體)· 獨立稽核 9/10 confirmed · 全程 seed 0 ·
業主任務:「先用經典的城市規劃理論得到原則和比例,生成一個城市,但目標著力於 stakeholder 本身,
最後看到城市形態長出來的樣子。」</div>
</header>
{static_parts[0]}
{TOC}
{static_parts[1]}
{SEC2}
{static_parts[2]}
{SEC4}
{SEC5}
{SEC6}
{SEC7}
{SEC8}
{SEC9}
{SEC10}
{SEC11}
{render_static(ST.SEC_REFS)}
{APPENDIX}
<footer style="margin-top:50px;border-top:2px solid var(--line);padding-top:14px;color:var(--muted);font-size:12.5px">
v06 · Urban-Stakeholder-Workshop / typology_foundation · 凍結輸入:v0.3 詞彙/池/簽名、v0.4 typed voids、
v05 先驗/讀回鏈 · 教義配方逐數字帶出處(doctrine_recipes.json / doctrine_sources.md)·
所有結果經獨立 audit agent 復算。</footer>
</div></body></html>"""

out = os.path.join(TF, "V06_REPORT.html")
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out)/1e6:.1f} MB")
