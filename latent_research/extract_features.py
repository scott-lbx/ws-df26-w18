"""
extract_features.py — 語料生成器:12 城 × 6 權力體制 → 72 樣本 × 9 形態特徵
=================================================================
接 05 引擎(common/operators/measure),對每個站點施加每種權力體制,
用 measure.diagnose 的 9 個形態指標量測,落盤成 out/latent_research/features.csv。

9 形態特徵(measure.diagnose):
  n  far  coverage  h_mean  h_max  h_cv  grain  slender  concentration
6 權力體制:current(現狀基線)+ regimes.yaml 的 5 種算子配方。

零 AI、純幾何。每個 (site, regime) 的 concentration 重心參照固定為該站現狀,故跨體制可比。
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:                                    # Windows cp950 打印中文會崩,強制 utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- 掛上 05 引擎(不重複資料/算子/形態邏輯)-----------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WS05 = PROJECT / "05_Shanghai Power to Form Students"
for _p in (str(WS05), str(WS05 / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as ws05_config   # noqa: E402  05 總開關(改 SLUG)
import common as C             # noqa: E402  取數據/角色/recs
import operators as ops        # noqa: E402  9 算子 + 配方引擎
import measure as M            # noqa: E402  形態指紋

OUT = PROJECT / "out" / "latent_research"
FEATURES = ["n", "far", "coverage", "h_mean", "h_max", "h_cv", "grain", "slender", "concentration"]


def discover_sites() -> list[str]:
    """自動找 05/data/<slug>/buildings.parquet 的所有站點,排序回傳。"""
    data = WS05 / "data"
    slugs = sorted(p.parent.name for p in data.glob("*/buildings.parquet"))
    return slugs


def regime_names() -> list[str]:
    """6 體制 = current + regimes.yaml 的 key(順序固定)。"""
    regs = ops.load_regimes(WS05 / "regimes.yaml")
    return ["current"] + list(regs.keys())


def diagnose_site(slug: str) -> list[dict]:
    """對一個站點跑全 6 體制,回傳 6 個 row(dict)。concentration 參照固定為該站現狀。"""
    ws05_config.SLUG = slug
    df = C.assign_all(C.current_buildings(slug))
    recs = C.to_recs(df)
    regs = ops.load_regimes(WS05 / "regimes.yaml")
    after = {name: ops.apply_regime(recs, regs[name]) for name in regs}
    rows, _ctr = M.compare(recs, after, slug)   # {regime: {9 feats}}, 現狀重心固定
    meta = C.site_meta(slug)
    area_km2 = float(meta.get("area_km2", 0.0))
    site_name = meta.get("name", slug)
    n_current = int(rows["current"]["n"])
    out = []
    for regime, d in rows.items():
        row = {"site": slug, "site_name": site_name, "regime": regime,
               "area_km2": round(area_km2, 4), "n_current": n_current}
        for f in FEATURES:
            row[f] = float(d[f])
        out.append(row)
    return out


def extract(sites: list[str] | None = None) -> "list[dict]":
    """跑全部站點 × 全部體制,回傳 flat 的 row 列表。單站失敗只記警告不中斷。"""
    sites = sites or discover_sites()
    rows: list[dict] = []
    for slug in sites:
        try:
            site_rows = diagnose_site(slug)
            rows.extend(site_rows)
            print(f"  [ok] {slug:16s} {len(site_rows)} regimes  (n_current={site_rows[0]['n_current']})")
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {slug:16s} {type(e).__name__}: {e}")
    return rows


def write_csv(rows: list[dict], path: Path = None) -> Path:
    import csv
    path = path or (OUT / "features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["site", "site_name", "regime", "area_km2", "n_current"] + FEATURES
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return path


def main():
    sites = discover_sites()
    regimes = regime_names()
    print(f"== 語料生成:{len(sites)} 城 × {len(regimes)} 體制 = {len(sites) * len(regimes)} 樣本 ==")
    print("   sites  :", ", ".join(sites))
    print("   regimes:", ", ".join(regimes))
    rows = extract(sites)
    p = write_csv(rows)
    print(f"== 完成:{len(rows)} 樣本 × {len(FEATURES)} 特徵 → {p.relative_to(PROJECT)} ==")
    # 摘要
    got_sites = sorted({r["site"] for r in rows})
    print(f"   實得站點 {len(got_sites)}/{len(sites)},樣本 {len(rows)}/{len(sites) * len(regimes)}")


if __name__ == "__main__":
    main()
