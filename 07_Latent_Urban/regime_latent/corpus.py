# -*- coding: utf-8 -*-
"""corpus.py —— 生成潜空间研究语料 features.csv

只用 05 config.DATASET_ROOT(上海城市数据集)现建每个城市的建筑数据,
对每城施加 baseline + 4 权力体制,measure.diagnose 出 9 维形态特征。

产出: 07_Latent_Urban/out/features.csv
  欄: slug,name,family,regime + 9 特征
  列: 12 城 × (baseline + 4 体制) = 60 样本
语料引擎(measure/operators/common/config)在 05;此处只借用,产物落在 07。
"""
import sys
import pandas as pd

from . import _paths

# 借用 05 的形态/体制引擎(data + engine),latent 分析产物仍留 07
sys.path.insert(0, str(_paths.O5))
sys.path.insert(0, str(_paths.O5 / "engine"))
# 05 与 07 都有名为 config 的模組:若 sys.modules 已缓存 07 的(无 SITES),
# 先清掉 config 及依赖它的 05 引擎,确保载入 05 版本(避免同 kernel 混用时撞名)。
_c = sys.modules.get("config")
if _c is not None and "05_Shanghai" not in getattr(_c, "__file__", "").replace("\\", "/"):
    for _m in ("config", "common", "measure", "operators", "pipeline"):
        sys.modules.pop(_m, None)
import config          # noqa: E402  (05 的总开关:DATASET_ROOT / SITES)
import common as C     # noqa: E402
import measure as M    # noqa: E402
import operators as ops  # noqa: E402

FEATURES = ["far", "coverage", "h_mean", "h_max", "h_cv",
            "grain", "slender", "concentration", "n"]

# 12 城:A 代表站 9 + B 扩充站 3(家族多样,ascii slug)
SLUGS = ["lujiazui", "caoyang", "yuyuan", "waitan", "nanjingxi", "kongjiang",
         "pengpu", "laoximen", "dapuqiao", "xujiahui", "nanjingdong", "beiwaitan"]

FAMILY = {s["slug"]: s["family"] for s in config.SITES}


def build():
    regs = ops.load_regimes()
    regime_names = list(regs.keys())
    rows = []
    for slug in SLUGS:
        try:
            df, src = C.build_or_load(slug)
            recs = C.to_recs(df)
        except Exception as e:
            print(f"[skip] {slug}: {e}")
            continue
        fam = FAMILY.get(slug, "?")
        name = config.site_name(slug)
        base = M.diagnose(recs, slug)
        rows.append({"slug": slug, "name": name, "family": fam,
                     "regime": "baseline", **{k: base[k] for k in FEATURES}})
        for rn in regime_names:
            try:
                after = ops.apply_regime(recs, regs[rn])
                d = M.diagnose(after, slug)
                rows.append({"slug": slug, "name": name, "family": fam,
                             "regime": rn, **{k: d[k] for k in FEATURES}})
            except Exception as e:
                print(f"[skip] {slug}/{rn}: {e}")
        print(f"[ok] {slug} ({src}) n={len(recs)}")

    _paths.OUT.mkdir(parents=True, exist_ok=True)
    dfo = pd.DataFrame(rows)
    dfo.to_csv(_paths.CSV, index=False, encoding="utf-8")
    print(f"\n[DONE] {len(dfo)} 样本 × {len(FEATURES)} 特征 → {_paths.CSV}")
    print(dfo.groupby('regime').size().to_string())
    return dfo


if __name__ == "__main__":
    build()
