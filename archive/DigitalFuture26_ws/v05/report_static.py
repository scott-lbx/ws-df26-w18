# -*- coding: utf-8 -*-
"""report_static.py — V05 報告的固定章節(緒論/資料/方法)。
數字佔位符 @KEY@ 由 make_v05_report.py 以 results_e*.json 填入。
圖片佔位符 [[FIG:filename.png]] 由組裝器替換為 base64 <figure>。"""

ABSTRACT = """
<div class="abstract"><h3>摘要</h3>
<p>本研究以上海一座城市的開放資料(建築輪廓與 AI 解譯高度、土地利用分類、POI、AOI、
建成年代、路網、地表覆蓋)為材料,探索一個三角問題:<b>電腦可計算的建築「分類」,
能否成為「權力配置」與「城市形態」之間的橋樑?</b>我們組裝了一個 24 個 1 km 視窗、
@N_BUILDINGS@ 棟建築的分析資料集,執行四個實驗:E1 以形態(含路/水距離等脈絡特徵)
辨識語義類型,並用 leave-windows-out 檢驗空間轉移;E2 將文獻錨定的權力先驗
(年代條件化、逐格出處與信度)在 24 窗尺度上運轉,產出視窗權力剖面與語義×家族共現橋;
E3 直接檢驗三角的斜邊——視窗層級的形態簽名能否回收權力剖面(LOO 迴歸 + permutation 檢定
+ 「型作為橋」消融);E4 以反事實生成閉環:同一塊地、同一骨架,三種權力配置長出三種布。
主要發現:@ABSTRACT_FINDINGS@
本報告盡可能完整披露每一步的選擇與理由、失敗與意外;失敗本身視為一等結果。</p></div>
"""

SEC1 = """
<h2 id="sec1">1 研究問題與背景</h2>

<h3>1.1 問題</h3>
<p>業主給定的研究任務一句話:「我們有一些城市數據,怎麼去找到<b>分類、權力與城市形態</b>
的關係的探索。不一定要真的探索成功,也可以是失敗,然後變成一些有趣的實驗去呈現一些面向。」</p>
<p>我們把這句話操作化為一個三角(圖 1-1):三個頂點是<b>分類</b>(電腦可識別的建築語義類型)、
<b>權力</b>(誰主導了這塊建成環境:政府/開發資本/居民社群/非正式)、<b>形態</b>
(從單體幾何到 50 m 格網文法到視窗簽名的多尺度形式描述)。四個實驗各佔一條邊:</p>
<ul>
<li><b>E1(形態→分類)</b>:純形態與形態+脈絡能在多大程度認出語義類型?哪些類型塌縮?</li>
<li><b>E2(分類→權力)</b>:文獻先驗 × 建成年代,把語義類型翻譯成權力權重,聚合成視窗剖面。</li>
<li><b>E3(形態↔權力)</b>:跳過分類直接測斜邊——視窗形態簽名能否回收權力剖面?</li>
<li><b>E4(權力→形態)</b>:反方向走完全程——給定權力配置,生成端長出什麼布?</li>
</ul>
[[FIG:fig_d1_triangle.png|圖 1-1 — 研究設計:三角與四個實驗的對應。]]

<h3>1.2 前置工作(本研究吸收的內部迭代)</h3>
<p>本研究不是憑空開始。2026-06-11 起的一系列內部迭代確立了出發點;v05 的姿態是
<b>回到問題本身重新設計實驗</b>,而非在 v0.4 上加增量。吸收的結論如下表:</p>
<table>
<tr><th>迭代</th><th>確立的事</th><th>v05 的吸收方式</th></tr>
<tr><td>A/B/C 三路驗證</td><td>資料、文獻、統計三方獨立收斂:<b>形態看不見權力,
但 metadata 看得見類型,類型攜帶權力語義</b>;不存在從形態預測所有權的先例(文獻缺口)</td>
<td>成為三角假說本身;E1/E3 是它在 24 窗尺度的正式檢驗</td></tr>
<tr><td>探針 v0.1</td><td>雙層類型系統可跑通;玩具放置器的密度天花板會吃掉權力訊號;
權力先驗 v0.1(逐格文獻出處+信度)</td><td>先驗 JSON 原樣繼承為 v05 先驗;「生成端缺陷蓋過訊號」
成為 E4 的設計警惕</td></tr>
<tr><td>v0.2</td><td>50 m token 文法 + Gibbs 生長 + 格元拼貼;in-sample 3/3 是開卷考</td>
<td>生成機制繼承;「開卷/閉卷之分」成為 E1 雙重 CV 設計的靈感</td></tr>
<tr><td>v0.3</td><td>24 窗簽名雲(PC1 強度/垂直度、PC2 空曠尖峰 vs 密實毯);
out-of-sample 4/6;<b>池稀釋</b>命名為天花板</td><td>視窗庫、簽名、token 詞彙全部重用;
E3 直接站在簽名雲上</td></tr>
<tr><td>v0.4</td><td>有類型的空(E_ROAD/E_WATER/E_GREEN/E_VACANT)+ keep-out;
<b>非空間指標看不見空間修正</b>(度量盲區)</td><td>遮罩與 typed 格網成為 E1 新特徵
(路/水距離)與 E4 的基地約束;度量盲區教訓寫進 E3 的指標選擇</td></tr>
</table>

<h3>1.3 文獻定位(極簡版,詳見 B 報告)</h3>
<p>三條支撐線:(i) Conzen 學派 agents-of-change(Whitehand &amp; Whitehand 1984;Larkham 1988)
證明「行動者→物質變更」可被檔案化統計,但從未蒸餾出可移植的權重表;(ii) Willis (1995)、
Barr (2012) 把「投機資本→辦公塔樓的高度與外殼」做到機制+計量層級——型↔權的少數硬證據;
(iii) 中國脈絡提供最高信度錨點(單位大院=國家:Bray 2005;商品房=開發商×國家:Wu 2005、
Lu et al. 2020;城中村=村民/非正式:Hao et al. 2011),同時擊穿一對一映射(同形異權)。
一條紅線貫穿全研究:<b>權力歸屬永遠不從形態判讀</b>(反循環論證),只從 metadata+文獻+年代走;
E3 對斜邊的檢驗正是在測這條紅線之外還剩多少統計關聯。</p>
"""

SEC2 = """
<h2 id="sec2">2 資料</h2>

<h3>2.1 來源圖層與可掛接性</h3>
<p>所有資料來自 <code>data_collection/上海城市数据集/</code>(開放資料拷貝)。
逐層盤點與三窗實測掛接率出自 A 報告(2026-06-11),v05 直接沿用其結論:</p>
<table>
<tr><th>圖層</th><th>量</th><th>對建築的語義</th><th>實測可掛接率</th></tr>
<tr><td>AI 影像解譯-帶高度</td><td>1,366,427 棟</td><td>footprint + 高度(譜系層,主鍵 id)</td><td>100%</td></tr>
<tr><td>EULUC 建設用地分類</td><td>55,018 塊</td><td>11 類用地(行政辦公/教育/醫療…)</td><td>86–94%</td></tr>
<tr><td>AOI(百度)</td><td>25,680 面</td><td>小區/大院級類型 + 價格(19.3% 填充)</td><td>23–78%</td></tr>
<tr><td>OSM 帶 type 建築</td><td>145,923 棟</td><td>建築類型標籤(59.7% 非空)+ 名稱</td><td>10–19%</td></tr>
<tr><td>帶年份內測版</td><td>462,478 棟</td><td>Function 100% 填充 + <b>Age 1984–2018</b></td><td>22–45%</td></tr>
<tr><td>POI 2024</td><td>1,754,933 點</td><td>三級分類(含政府機構 54,444 點)</td><td>50 m 內 79–99%</td></tr>
<tr><td>路網合集(OSM fclass)</td><td>200,108 段</td><td>道路等級(v0.4 起入列)</td><td>buffer 遮罩</td></tr>
<tr><td>ESA10m 地表覆蓋</td><td>1,375,040 面</td><td>水體(Code 80,幾何驗證)</td><td>遮罩</td></tr>
</table>

<div class="gapbox"><b>已知資料缺口(誠實清單,全程約束推論強度)</b>
<ul>
<li><b>沒有任何產權/地籍/開發商圖層</b>——權力語義全部是從使用功能+名稱+年代推論的 proxy,不是產權事實。這是 E2 先驗鏈存在的根本原因,也是其天花板。</li>
<li>房價柵格與鏈家層在本拷貝缺失(資料夾 05–08 整批不在);唯一價格訊號是 AOI 的價格欄(19.3%)。</li>
<li>Age 在 1984/85 有歸併尖峰(≈25% =「1985 年以前」),2018 後無資料,且<b>郊區五窗 Age 覆蓋為 0</b>。</li>
<li>AI 高度低估超高層(上海中心 632 m 在層內只剩 338 m),郊區有可疑值(楓涇 h_p90 75.8 m)。</li>
<li>ESA 水體的屬性翻譯欄錯位——以幾何驗證(黃浦江完整出現於 Code 80)取代翻譯表。</li>
<li>OSM 標籤偏向地標/有名建築;EULUC 是 parcel 級先驗,整塊地同色。</li>
</ul></div>

<h3>2.2 研究區:24 個 1 km 視窗</h3>
<p>沿用 v0.3 的分層抽樣(50 個策展區、10 個類型組、每組 ≥2、四大組 +1,
重疊 >30% 剔除;18 TRAIN / 6 HELDOUT 按組分層,陸家嘴/老西門/曹楊強制入 TRAIN)。
類型組名稱只作抽樣框 metadata,不進任何模型。</p>
[[FIG:fig_e0_map.png|圖 2-1 — 研究區:24 個 1 km 視窗(圓=TRAIN,星=HELDOUT)。]]

<h3>2.3 v05 分析資料集的組裝</h3>
<p>v05 把 v0.3 的逐窗融合表、v0.4 的遮罩與 typed 格網合併成單一分析資料集
(<code>build_dataset.py → dataset.parquet</code>):@N_BUILDINGS@ 棟 × @N_FEATURES@ 個屬性欄位(另含 geometry),
新增四個「形態脈絡」特徵(到主幹路/全路網/水體/綠地的距離)與鄰里特徵
(100 m 內棟數、鄰均高)。@DATASET_NOTES@</p>
<p>八個語義類與五個形態家族的全域支持數如下(權力語義類的小樣本——healthcare 128 棟、
industrial 180 棟——直接約束 E1 對這些類的統計強度):</p>
@CLASS_TABLE@
[[FIG:fig_e0_dataset.png|圖 2-2 — 資料集概覽:逐窗語義構成(GFA 加權)、規模與標籤覆蓋率。]]
[[FIG:fig_d2_pipeline.png|圖 2-3 — 資料→標籤管線:五源融合級聯與幾何軌(紅框=反循環紅線)。]]
"""

SEC3 = """
<h2 id="sec3">3 方法總覽</h2>

<h3>3.1 雙層類型系統</h3>
<p>建築「類型」拆成兩層,各做擅長的事:<b>語義層</b>(8 類,從 metadata 融合而來,
權力先驗掛在這層)與<b>形態家族層</b>(5 族,從幾何規則而來,生成在這層操作),
中間以資料學得的共現橋連接(圖 3-1)。這個架構是 A/B/C 三路驗證的直接產物:
形態認得出家族、認不出機能;機能(類型)才攜帶權力語義。</p>
[[FIG:fig_d3_twolayer.png|圖 3-1 — 雙層類型系統:先驗掛語義層,生成走家族層,橋=共現計數。]]

<h3>3.2 語義類型:五源融合級聯</h3>
<p>優先序:EULUC parcel 先驗 → AOI 覆寫 → OSM-type/Function 覆寫 → POI 50 m 多數決補空(≥2 票)
→ 形態補底。住宅先記 generic,再以 floors≤3 且 fp&lt;300 m² 分流 lowrise/estate。</p>
<div class="choice"><b>選擇:</b>沿用 v0.3 的融合級聯而非 v0.2 的 POI 單一主判。
<b>備選:</b>POI 主判(教學上更簡單)。<b>理由:</b>v0.3 的實證——融合下 POI 只判 7.8%,
而 POI 稀疏的村/水鄉窗在 POI 主判下會大面積失明;郊區窗的標籤可得性決定了 24 窗研究的可行性。
代價:EULUC parcel 先驗把整塊地塗同色(標籤的空間自相關),這直接影響 E1 的 CV 設計(§4)。</div>

<h3>3.3 形態家族與 50 m token 文法</h3>
<p>家族規則級聯(先到先得):large_footprint(fp≥2000 m²)→ tower(h≥35 m 且長寬比&lt;2)
→ slab(長寬比≥2 且 4–12 層)→ house(≤3 層且 fp&lt;400 m²)→ midrise_other。
閾值由 pooled 分位數錨定(v0.1 紀錄)。注意語義層的 lowrise 分流(fp&lt;300 m²)與家族層的 house(fp&lt;400 m²)閾值刻意不同——兩者服務不同層,前者分流住宅語義、後者界定形態家族,並非筆誤。50 m 格網 token(k=8 + 有類型的空)沿 v0.3/v0.4:
詞彙只從 18 個 TRAIN 窗擬合;v0.4 把 EMPTY 拆成 E_VACANT/E_ROAD/E_WATER/E_GREEN。</p>

<h3>3.4 權力先驗的形式化</h3>
<p>每棟建築的權力向量 <b>w</b>(b) = prior(semantic(b), age_regime(b)) ∈ [0,1]⁴,
四維 {government, developer_capital, resident_community, informal} <b>加法、非互斥</b>
(文獻強制:Wu 2005 同形異權;Lu et al. 2020 國家在所有類型中在場 → gov 地板值)。
residential_estate 按年代條件化(≤1985 單位邏輯 / 1986–97 雙軌內插 / ≥1998 商品房邏輯 / 無年代低信度混合)。
視窗剖面 = 標籤建築的 GFA 加權平均;unlabeled GFA 排除並另行報告。
每格先驗帶 B 報告出處與信度,低信度 GFA 份額隨剖面一起交付。</p>
<div class="choice"><b>選擇:</b>先驗原樣沿用 v0.1(文獻常數),不做任何拍腦袋調整。
<b>備選:</b>按 v0.3/0.4 的新觀察微調權重。<b>理由:</b>先驗的唯一正當更新路徑是檔案證據
(Whitehand 式規劃申請/出讓記錄校準,B 報告 §4.5),用自己的中間結果調先驗會把探索變成循環。
代價:informal 維度全部低信度、里弄用城中村文獻類比的缺陷原樣保留,在 E2/E3 的解讀中反覆出現。</div>

<h3>3.5 視窗形態簽名</h3>
<p>每窗簽名 = 9 維 token 組成 + 10 個形態指標(FAR、coverage、h p50/p90、中位 footprint、
5 個家族份額),TRAIN 統計 z-score 後 PCA(只 fit TRAIN、全投影):
PC1 ≈ 強度/垂直度(29.8%)、PC2 ≈ 空曠尖峰 vs 密實毯(27.0%)。E3 在這個座標系上測權力。</p>
"""

# 結果章節 (SEC4-SEC7)、討論 (SEC8)、侷限 (SEC9)、結論 (SEC10) 由 make_v05_report.py
# 在實驗結果出爐後生成 — 見該檔。

SEC_REFS = """
<h2 id="refs">參考文獻(選列;完整註解書目見 B_literature_map.md,45+ 條)</h2>
<ol class="refs">
<li>Conzen, M.R.G. (1960). <i>Alnwick, Northumberland: A Study in Town-Plan Analysis</i>. IBG Publication 27.</li>
<li>Whitehand, J.W.R. &amp; Whitehand, S.M. (1984). The physical fabric of town centres: the agents of change. <i>Transactions of the IBG</i> NS 9(2), 231–247.</li>
<li>Larkham, P.J. (1988). Agents and types of change in the conserved townscape. <i>Transactions of the IBG</i> NS 13(2), 148–164.</li>
<li>Willis, C. (1995). <i>Form Follows Finance: Skyscrapers and Skylines in New York and Chicago</i>. Princeton Architectural Press.</li>
<li>Barr, J. (2012). Skyscraper height. <i>Journal of Real Estate Finance and Economics</i> 45(3), 723–753.</li>
<li>Bray, D. (2005). <i>Social Space and Governance in Urban China: The Danwei System from Origins to Reform</i>. Stanford University Press.</li>
<li>Wu, F. (2005). Rediscovering the 'gate' under market transition. <i>Housing Studies</i> 20(2), 235–254.</li>
<li>Lu, T., Zhang, F. &amp; Wu, F. (2020). The variegated role of the state in different gated neighbourhoods in China. <i>Urban Studies</i> 57(8).</li>
<li>Wu, F. (2016). State dominance in urban redevelopment. <i>Urban Affairs Review</i> 52(5).</li>
<li>Hao, P., Sliuzas, R. &amp; Geertman, S. (2011). The development and redevelopment of urban villages in Shenzhen. <i>Habitat International</i> 35(2), 214–224.</li>
<li>Steadman, P. (2014). <i>Building Types and Built Forms</i>. Matador.</li>
<li>Kropf, K. (2009). Aspects of urban form. <i>Urban Morphology</i> 13(2), 105–120.</li>
<li>Hillier, B. (1996). <i>Space is the Machine</i>. Cambridge University Press.</li>
<li>Webster, C. (2002). Property rights and the public realm. <i>Environment and Planning B</i> 29(3), 397–412.</li>
<li>Habraken, N.J. (1998). <i>The Structure of the Ordinary: Form and Control in the Built Environment</i>. MIT Press.</li>
<li>Biljecki, F. &amp; Chow, Y.S. (2022). Global building morphology indicators. <i>CEUS</i> 95, 101809.</li>
<li>「Predicting building types and functions at transnational scale」 arXiv:2409.09692 (2024) — 辨識端可行性基準(kappa 0.75–0.84)。</li>
</ol>
"""
