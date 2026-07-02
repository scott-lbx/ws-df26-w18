# -*- coding: utf-8 -*-
"""_font.py —— 讓 matplotlib 顯示中文,不變豆腐方塊(□□)。

問題:matplotlib 預設字型(DejaVu Sans)沒有中文字符 → 中文全變空方塊,
且負號會用到缺字。解法:挑一個系統實際有的 CJK 字型設進 rcParams,
並關掉 unicode 負號。`import nn` 時自動執行一次,全 notebook 生效。
"""
import matplotlib
import matplotlib.font_manager as fm

# 依序嘗試:Windows → macOS → Linux;第一個系統有的就用
_CANDIDATES = (
    "Microsoft YaHei", "Microsoft JhengHei", "SimHei", "DengXian",  # Windows
    "PingFang SC", "PingFang TC", "Arial Unicode MS", "Hiragino Sans GB",  # macOS
    "Heiti TC", "STHeiti", "Noto Sans CJK SC", "Noto Sans CJK TC",  # 其餘
    "Source Han Sans SC", "WenQuanYi Zen Hei",  # Linux
)


def set_cjk_font(verbose=False):
    """把第一個可用 CJK 字型設為 matplotlib 預設 sans-serif。回傳字型名(或 None)。"""
    avail = {f.name for f in fm.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in avail:
            # 放最前面但保留 DejaVu 當英數 fallback
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False  # 負號不缺字
            if verbose:
                print(f"[nn] CJK 字型 = {name}")
            return name
    if verbose:
        print("[nn] 找不到 CJK 字型;中文可能顯示為方塊。"
              "請安裝 Noto Sans CJK 或 Microsoft YaHei。")
    return None
