# -*- coding: utf-8 -*-
"""build05.py —— 需要時,呼叫 05 的多源拼接管線現建 buildings.parquet

05 的 01「數據-怎麼選與拼接」把 5 個 spine(輪廓+高度 / EULUC / Function / AOI…)
用空間 join 拼成 `data/<slug>/buildings.parquet`。07 一直讀的就是這個拼接結果。

本模組讓 07 在**快取缺該城**時,自動觸發 05 的拼接(build_or_load)補上——
於是 NB1-3 可用全市任一街道,而非只有隨包 12 城。

隔離:在**子行程**跑 05 的流程,避免 05/07 同名 `config` 在同一 kernel 撞名。
"""
import sys
import subprocess
from pathlib import Path

try:
    from .. import config
except (ImportError, ValueError):
    import config

_O5 = Path(__file__).resolve().parents[2] / "05_Shanghai Power to Form Students"


def city_parquet(slug):
    return config.DATA_ROOT / slug / "buildings.parquet"


def ensure_city(slug, rebuild=False):
    """確保 data/<slug>/buildings.parquet 存在;缺就用 05 拼接管線現建。
    rebuild=True 則不管有沒有都重拼(原始資料更新時用)。回傳 parquet 路徑。"""
    p = city_parquet(slug)
    if p.exists() and not rebuild:
        return p
    o5 = str(_O5).replace("\\", "/")
    code = (
        f"import sys; sys.path.insert(0, r'{o5}'); sys.path.insert(0, r'{o5}/engine');"
        f"import config;"
        f"assert getattr(config,'DATASET_ROOT',None), '05 config.DATASET_ROOT 未設,無法拼接';"
        f"import common as C; C.build_or_load('{slug}')"
    )
    print(f"[build05] 快取缺 {slug},呼叫 05 多源拼接管線現建…")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"05 拼接失敗({slug}):\n{r.stdout[-500:]}\n{r.stderr[-800:]}")
    if not p.exists():
        raise FileNotFoundError(f"拼接後仍找不到 {p}")
    print(f"[build05] OK → {p}")
    return p
