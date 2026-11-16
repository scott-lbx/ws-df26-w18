# -*- coding: utf-8 -*-
"""report_static.py — V06 報告固定章節。佔位符 @KEY@ 由 make_v06_report.py 填入;
[[FIG:name.png|caption]] 由組裝器替換為 base64 <figure>。"""

ABSTRACT = """
<div class="abstract"><h3>摘要</h3>
<p>經典城市規劃理論從來不只給「技術配比」——每一個範式背後都站著一組它要服務(或要馴化)的
stakeholder。本研究把五個經典範式(霍華德田園城市 1898、柯比意光輝城市/雅典憲章 1933、
Perry 鄰里單位 1929、蘇聯 microrayon、中國 GB 50180-2018)蒸餾成<b>逐數字帶出處的定量配方</b>
(用地配比、密度、高度制度、街廓粒度、幾何規則),翻譯進本專案的上海 50 m token 詞彙,
在 1 km² 畫布上生成五座「範式城市」(單元 + 城市級複合體,平面圖 + 3D 軸測),
然後做本研究的核心動作:<b>用權力歸屬鏈把生成的肌理「讀回去」——範式宣稱服務誰
(文獻先驗)vs 肌理被讀成誰的城(讀回剖面)——兩者的落差就是研究對象。</b>
另設兩個檢核:生成範式城投影到 24 個真實上海視窗的簽名雲上(錨點:曹楊新村 1951 年
按鄰里單位原則規劃,是有文獻記載的歷史事實);以及反轉實驗——給定目標 stakeholder,
反問哪個範式(或混血)最接近。主要發現:@ABSTRACT_FINDINGS@</p></div>
"""

SEC1 = """
<h2 id="sec1">1 研究問題</h2>
<p>業主任務(原話):「我們先用經典的城市規劃理論,得到城市規劃的原則和比例,比如說應該有
多少住宅、多少辦公,一些經典的範式,然後去生成一個城市。但是目標要著力於 stakeholder 本身。
最後也要看到城市形態長出來的樣子。」</p>
<p>我們把「著力於 stakeholder 本身」操作化為一個鏡像結構:</p>
<ul>
<li><b>正向(範式→城)</b>:範式配方 → 上海詞彙翻譯 → 生成肌理(D1–D3);</li>
<li><b>反向(城→stakeholder)</b>:生成肌理 → 權力歸屬鏈讀回 → 「這是誰的城」(D4);</li>
<li><b>落差 = 發現</b>:範式的 a priori stakeholder 宣稱(從規劃史文獻蒸餾)與肌理的
read-back 剖面之間的逐維落差、排序翻轉與其分解(D4);</li>
<li><b>定位與錨點</b>:範式城落在真實上海簽名雲的哪裡;曹楊新村=鄰里單位的歷史錨(D5);</li>
<li><b>反轉</b>:stakeholder-first——給定目標剖面,選範式/混血並生成(D6)。</li>
</ul>
<p>與前輪的連接:v05 E4 發現生成鏈「傳形不傳名」(k=8 形態詞彙在傳輸中剝離機構身分),
其結論的下一步「語義條件化的拼貼」<b>在本輪正式實裝</b>(D3);讀回鏈沿用 v05 E2
(文獻先驗 × GFA 加權,asserted 非 observed,全程照規矩標注)。</p>
[[FIG:fig_d0_design.png|圖 1-1 — 研究設計:正向生成 × 反向讀回,落差即對象。]]

<h3>1.1 為什麼經典範式是 stakeholder 綱領</h3>
<p>規劃史的常識但值得明說:霍華德的田園城市以<b>社區信託持有土地、地租歸公</b>為核心機制
(合作社主義);雅典憲章/光輝城市把配置權完全交給<b>國家與技術官僚</b>(「陽光、空間、綠地」
的供給者是計畫);Perry 的鄰里單位以家庭與社區為理想單元,但其圖式被美國開發商
即刻吸收為<b>細分開發模板</b>(社區理想 × 開發商執行的雙重身分);蘇聯 microrayon 是
國家標準化供給的極致;GB 50180 則是<b>國家規範訂定 × 開發商執行</b>的當代中國正式體制。
這些 a priori 配置在 D1 中逐一給出文獻出處與信度,並誠實標注我們的四維詞彙
(政府/開發資本/居民社群/非正式)在哪些範式上發生時代錯置(anachronism)。</p>
"""

SEC3_METHODS = """
<h2 id="sec3">3 方法總覽</h2>
<h3>3.1 翻譯層:教義配方 → 上海詞彙</h3>
<p>每個範式的用地配比+密度+高度制度被翻譯成 400 格(20×20 × 50 m)的 token 組成
({T1..T8} ∪ {E_VACANT, E_ROAD, E_GREEN}),逐條主張對應逐條分配、附理由
(D2 的 translation.json 是完整對照表)。範式的幾何教義(環帶、周界幹道、中心學校、
平行板樓行列)被寫成<b>程序化骨架</b>——typed void 格網 + 保留 civic 格。
教義說不出來的,不硬說:<b>表達力審計</b>記錄每一處「教義要求 vs 上海詞彙最近說法」的失真
(例:柯比意要求塔樓格覆蓋率 ~0.10,上海 T6 實測 0.31——只能用「T6 格嵌入綠毯」
讓街區級覆蓋率達標)。這些失真不是缺陷修飾,是發現:<b>一個城市的形態詞彙
說不出另一個時代的教義</b>。</p>
<h3>3.2 生成:規定性放置 + 語義條件化拼貼</h3>
<p>範式是規定性的(prescriptive),不是統計場——能由教義幾何直接放置的(塔樓點陣、
板樓行列、同心帶)一律直接放置;教義沉默處才用 v0.4 的 12-token 鄰接做 3 次 Gibbs 掃描補位
(標注為「上海口音」)。實例化用<b>語義條件化的格元拼貼</b>:每格優先抽語義相符的真實上海格元
(civic 格要求 education/public_admin 池),池空才退到同 token 任意格元並計數——
fallback 率本身是「上海池能不能供應這個範式」的量測。</p>
<h3>3.3 讀回鏈與落差分解</h3>
<p>讀回 = v05 E2 原樣:逐棟語義 → 先驗格(住宅一律落 no_age 低信度格——生成建築沒有年代,
這個「中國無年代先驗讀霍華德」的錯置是實驗設計的一部分,不是疏忽)→ GFA 加權聚合。
落差分解成三個來源:(a) no_age fallback 人造物(以 no_age→pre1985 交換重算為界),
(b) 語義拼貼 fallback(D3 計數),(c) 真實的綱領翻譯損失。</p>
<h3>3.4 簽名雲定位與歷史錨</h3>
<p>範式單元算 v0.3 同款 19 維簽名(typed void 摺回 T0,披露)、用 TRAIN 統計 z-score、
投影到既有 PCA(先驗證能 1e-3 復現 24 個真實窗的存檔座標,作為正確性閘)。
錨點檢驗:曹楊新村(A13)1951 年由汪定曾按鄰里單位原則規劃——生成的 neighbourhood_unit
範式城,離 A13 近不近?命中是全鏈的強驗證;失手則是「1951 年的在地化+70 年改建改掉了什麼」
的誠實提問。</p>
"""

SEC_REFS = """
<h2 id="refs">參考文獻(選列;教義數字的逐條出處見 doctrine_sources.md)</h2>
<ol class="refs">
<li>Howard, E. (1898/1902). <i>To-morrow: A Peaceful Path to Real Reform</i> / <i>Garden Cities of To-morrow</i>. London.</li>
<li>Le Corbusier (1933/1935). <i>La Ville Radieuse</i>; CIAM (1933). <i>Charte d'Athènes</i>.</li>
<li>Perry, C. (1929). The Neighborhood Unit. In <i>Regional Survey of New York and Its Environs</i>, Vol. 7.</li>
<li>СНиП II-60-75** 《Планировка и застройка городов…》(蘇聯城市規劃建設規範,microrayon 規範譜系)。</li>
<li>GB 50180-2018《城市居住区规划设计标准》。中华人民共和国住房和城乡建设部。</li>
<li>汪定曾 (1956). 上海曹杨新村住宅区的规划设计. 《建筑学报》1956(2)。</li>
<li>Hall, P. (1988). <i>Cities of Tomorrow</i>. Blackwell.(範式史的標準綜述)</li>
<li>Fishman, R. (1977). <i>Urban Utopias in the Twentieth Century</i>. Basic Books.(Howard/Wright/Corbusier 的 stakeholder 解讀)</li>
<li>Wu, F. (2005). Rediscovering the 'gate' under market transition. <i>Housing Studies</i> 20(2).(權力先驗詞彙的中國錨)</li>
<li>Lu, T., Zhang, F. &amp; Wu, F. (2020). The variegated role of the state… <i>Urban Studies</i> 57(8).</li>
<li>Lu, D. (2006). <i>Remaking Chinese Urban Form</i>. Routledge.(工人新村與鄰里單位的中國接受史)</li>
</ol>
"""
