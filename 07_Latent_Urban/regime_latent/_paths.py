# -*- coding: utf-8 -*-
"""_paths.py —— 路徑與中文字型(regime_latent 子套件共用)

regime_latent = 「城市 × 權力體制」的形態潛空間分析(有別於 07 的建築層 nn)。
語料來源:05 專案的 data + engine(measure.diagnose / operators.apply_regime);
latent 相關的分析、圖、報告一律留在 07,保持 05 乾淨。
"""
from pathlib import Path
import matplotlib

HERE = Path(__file__).resolve()
PKG = HERE.parent                       # .../07_Latent_Urban/regime_latent
ROOT07 = PKG.parent                     # .../07_Latent_Urban
REPO = ROOT07.parent                    # repo 根
O5 = REPO / "05_Shanghai Power to Form Students"   # 語料引擎在此(data + engine)

OUT = ROOT07 / "out"                    # 產物全部落在 07
FIGS = OUT / "figs"
CSV = OUT / "features.csv"
REPORT = OUT / "研究报告.md"


def set_cjk_font():
    """中文不變豆腐方塊。委派給 nn._font(單一來源);找不到就內建 fallback。"""
    import sys
    if str(ROOT07) not in sys.path:
        sys.path.insert(0, str(ROOT07))
    try:
        from nn._font import set_cjk_font as _f
        return _f()
    except Exception:
        avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        for f in ("Microsoft YaHei", "Microsoft JhengHei", "SimHei",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
            if f in avail:
                matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return f
        return None
