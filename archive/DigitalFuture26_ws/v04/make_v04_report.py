#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_v04_report.py — build ../V04_REPORT.html (self-contained, figures base64)."""
import base64, json, os

V04 = os.path.dirname(os.path.abspath(__file__))
TF = os.path.dirname(V04)
res = json.load(open(os.path.join(V04, "results_v04.json"), encoding="utf-8"))


def b64(name):
    with open(os.path.join(V04, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


CSS = """
  :root { --ink:#1a1d23; --muted:#6b7280; --line:#e5e7eb; --bg:#fafaf8; --card:#fff;
          --accent:#0f5e63; --accent2:#b45309; --warn:#9f1239; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15.5px/1.7 "Avenir Next","PingFang TC","Noto Sans TC",system-ui,sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:44px 28px 80px; }
  header { border-bottom:3px solid var(--ink); padding-bottom:16px; margin-bottom:22px; }
  h1 { font-size:27px; margin:0 0 6px; }
  .sub { color:var(--muted); font-size:13.5px; }
  h2 { font-size:20px; margin:44px 0 12px; padding-left:12px; border-left:5px solid var(--accent); }
  h3 { font-size:16.5px; margin:26px 0 8px; color:var(--accent); }
  code { background:#eef0f3; padding:1px 5px; border-radius:4px; font-size:.85em;
         font-family:"SF Mono",Menlo,monospace; }
  table { width:100%; border-collapse:collapse; background:var(--card); font-size:12.8px;
          border:1px solid var(--line); margin:14px 0; }
  th { background:#f1f3f4; text-align:left; padding:7px 10px; border-bottom:2px solid var(--ink); }
  td { padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  blockquote { border-left:4px solid var(--accent); background:#f0f7f7; margin:14px 0;
               padding:10px 16px; border-radius:0 8px 8px 0; color:#374151; font-size:14px; }
  figure { margin:18px 0; background:var(--card); border:1px solid var(--line);
           border-radius:8px; padding:12px; }
  figure img { width:100%; height:auto; display:block; border-radius:4px; }
  figcaption { font-size:13px; color:var(--muted); margin-top:8px; }
  .verdict { background:#fff; border:2px solid var(--accent); border-radius:10px;
             padding:14px 20px; margin:18px 0; }
  .verdict b { color:var(--accent); }
  .warnbox { background:#fff5f5; border:2px solid var(--warn); border-radius:10px;
             padding:14px 20px; margin:18px 0; }
  .warnbox b { color:var(--warn); }
  ul { padding-left:22px; } li { margin:5px 0; }
"""

sp = res["spatial_metric"]["windows"]
HELD = ["A02", "A07", "A20", "A29", "A35", "A43"]
NAMES = {"A02": "南京東路", "A07": "外灘", "A20": "三林", "A29": "南橋",
         "A35": "漕河涇", "A43": "涼城"}
sp_rows = "".join(
    f"<tr><td>{h} {NAMES[h]}</td>"
    f"<td><b style='color:#9f1239'>{sp[h]['v03_free']['gfa_share_in_mask']:.1%}</b></td>"
    f"<td><b style='color:#0f5e63'>{sp[h]['v04_typed']['gfa_share_in_mask']:.1%}</b></td>"
    f"<td>{sp[h]['v03_free']['cell_gfa_pearson_r']:.2f}</td>"
    f"<td>{sp[h]['v04_typed']['cell_gfa_pearson_r']:.2f}</td></tr>"
    for h in HELD)

ex = res["exam_typed_skeleton"]
exam_rows = "".join(
    f"<tr><td>{h} {NAMES[h]}</td>"
    f"<td>{'✓' if ex['hit'][h] else '✗ → ' + ex['nearest'][h]}</td>"
    f"<td>{ex['grow_and_keepout_log'][h]['n_fixed_void_cells']}</td>"
    f"<td>{ex['grow_and_keepout_log'][h]['n_dropped']} 棟 "
    f"({ex['grow_and_keepout_log'][h]['dropped_gfa_share']:.1%} GFA)</td></tr>"
    for h in HELD)

cal = res["mask_calibration"]["windows"]
worst = sorted(cal.items(), key=lambda kv: -kv[1]["HARD"]["bld_share_gt20"])[:5]
cal_rows = "".join(
    f"<tr><td>{k} {v['name']}</td>"
    f"<td>{v['HARD_allroads']['bld_share_gt20']:.1%}</td>"
    f"<td><b>{v['HARD']['bld_share_gt20']:.1%}</b></td></tr>"
    for k, v in worst)

html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v0.4 報告 — 有類型的空與不可建遮罩</title><style>{CSS}</style></head><body><div class="wrap">
<header>
<h1>v0.4 — 有類型的空（typed voids）＋ keep-out 拼貼</h1>
<div class="sub">2026-06-12 · typology_foundation/v04 · 業主指示：「可以佔用原本的建成處，但不要佔『路』」 ·
seed 0 · 站在 v0.3 文法上（T1–T8 詞彙、池、β=8 Gibbs 逐字不動）</div>
</header>

<div class="verdict"><b>一句話裁定：</b>v0.3 的自由生長在外灘把 <b>56.4%</b> 的生成 GFA
蓋在江面和主幹路上——這就是「把空白當路蓋掉」的數字版。v0.4 把全部六個留出窗壓到
<b>&lt;1%</b>，移植的典範不對稱完整保留，而 hit rate（3/6 vs 4/6）「沒漲反跌」本身是
本輪最有教學價值的發現：<b>10 維非空間指標看不見修好的東西</b>。</div>

<h2>1. 遮罩與校準（先量真實，再定規則）</h2>
<p>路＝OSM fclass 按等級 buffer（motorway/trunk 30 m → service 6 m，footway 族排除）；
水＝ESA10m <b>Code 80</b>（屬性翻譯欄錯位，用幾何驗證：外灘窗 45% 水、楓涇 7% 河道、曹楊 0%——黃浦江完整出現）；
綠＝OSM 公園族＋EULUC 公园绿地（<b>farmland 留在 E_VACANT</b>：耕地是新城實際吃掉的可開發空）。</p>
<p><b>校準否決了全類路 keep-out</b>：10 m 全寬的「residential」巷會把老西門 40% 的真實建築標成違規。
規則改為 keep-out 只用主幹級（motorway→tertiary）∪ 水；巷弄是肌理、由格元自帶。校準後最差五窗：</p>
<table><tr><th>窗</th><th>全類路遮罩誤標真實建築</th><th>主幹∪水（採用）</th></tr>{cal_rows}</table>

<h2>2. T0 拆開之後（fig1 / fig2）</h2>
<p>「空白」大半不是可建地：<b>陸家嘴 254 空格只有 53 格真空地</b>（103 路、69 水、29 綠）；
南京東路 128 空格只有 16 格真空地；<b>西岸 302 空格中 192 水＋59 綠——v0.3 的 31% 標籤覆蓋率之謎當場解開</b>。
鄰接長出新結構：E_WATER 自鄰 .78、T8 巨構 21% 的鄰接面對路、T6 塔樓面路率（13.4%）是板樓的兩倍、
T5 高覆蓋老肌理幾乎不面任何 void。</p>
<figure><img src="{b64('fig1_typed_voids.png')}">
<figcaption>FIG 1 — typed void 地圖（遮罩線疊加）＋ 24 窗 T0 分解。</figcaption></figure>
<figure><img src="{b64('fig2_adjacency.png')}">
<figcaption>FIG 2 — 9-token vs 12-token 鄰接；右：每個建成 token 面對哪種空。</figcaption></figure>

<h2>3. 考試 v2（fig3）：{ex['hit_rate']}，與度量的盲區</h2>
<blockquote>誠實標注：typed-骨架考試<b>比 v0.3 自由生長簡單</b>（路/水/綠格按真實格網固定——公開可觀測資料，
但 PC2 主成分就是 void 結構），且 keep-out 改變了 GFA。兩個 hit rate <b>不是同一張考卷</b>，並排只為展示盲區。</blockquote>
<table><tr><th>窗</th><th>裁定</th><th>固定 void 格</th><th>keep-out 剔除</th></tr>{exam_rows}</table>
<p>A29 翻正（101/130 空格是真空地——「空的形狀」就是它的簽名）；A02 翻負（FAR 低估 40%＝池稀釋＋剔除，
被推向同樣高路佔比的外灘）；A35 是 1.436 vs 1.441 的<b>平手誤差</b>；A43→A35 毯子帶內互混原樣保留——
<b>池稀釋天花板不歸這版修，也確實沒被修掉</b>。</p>
<figure><img src="{b64('fig3_typed_exam.png')}">
<figcaption>FIG 3 — 中欄即重點：生成 GFA 壓在路∪水上的份額，v0.3（紅）vs v0.4（綠）。</figcaption></figure>

<h2>4. 空間度量：抱怨的量化版</h2>
<table><tr><th>窗</th><th>GFA 壓遮罩 v0.3</th><th>GFA 壓遮罩 v0.4</th>
<th>格 GFA 圖相關 r v0.3</th><th>r v0.4</th></tr>{sp_rows}</table>
<p>r 的增益集中在 void 結構強的窗（CBD/濱江 0.17–0.27），毯子帶接近 0——
<b>骨架在「空就是結構」的地方有用，在均勻毯子裡沒東西可錨</b>。</p>

<h2>5. 移植 keep-out（fig4）</h2>
<p>24 格全重跑：剔除成本 0–6.2% GFA、中位 FAR delta −0.11（正向波動是 T6/T8 重尾抽樣噪聲——keep-out 只能減不能加）。
陸家嘴 on 三林 FAR 2.42＝現況 ×1.66（v0.3 ×1.85）：<b>典範不對稱保留，但塔樓不再騎在幹道上</b>。
A36 on A02 的 v0.3 版有 39 棟、平均 59% footprint 壓在遮罩裡（fig4 紅色棟）。</p>
<figure><img src="{b64('fig4_keepout_transplant.png')}">
<figcaption>FIG 4 — before/after：藍＝路∪水遮罩，紅＝v0.3 站在遮罩裡的棟。
「土地格局打敗簽名」之外的新一條：路也打敗簽名。</figcaption></figure>

<h2>6. 誠實侷限</h2>
<div class="warnbox"><b>不是同一張考卷</b>：3/6 與 4/6 不可直接比大小；本輪的證據是空間度量，不是 hit rate。</div>
<ul>
<li>ESA 10 m 水線邊緣誤差：外灘 7.9% 真實建築被標違規 → A07 的 24.3% 剔除有一部分是遮罩錯不是生成錯。</li>
<li>OSM 路網郊區完整性未驗證（A32 路遮罩僅 6.6% 窗面積，存疑）；巷弄連續性／可穿越性完全沒進文法（street grammar 是 v0.5+ 候選）。</li>
<li>E_GREEN 只在格級擋、不做 sub-cell keep-out（真實公園裡有建築）。</li>
<li>單 seed；8 抽、>20% 剔除是選定不是推定；farmland→E_VACANT 是設計判斷。</li>
<li>池稀釋（v0.3 命名的天花板）原樣存在：A02 FAR 低估 40%。「簽名裡該有什麼」仍是下一個研究問題。</li>
</ul>

<h2>7. 重跑</h2>
<p><code>python3 run_v04.py prep → masks → calib → voids → adjacency → fig1 → fig2</code><br>
<code>python3 run_generate_v04.py exam → spatial → transplant → fig3 → fig4</code><br>
全部快取；遮罩自全市單次讀取（ESA where-filter 40 s）；FlatGeobuf 經 /tmp，不碰 GPKG。</p>
</div></body></html>"""

out = os.path.join(TF, "V04_REPORT.html")
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out)/1e6:.1f} MB")
