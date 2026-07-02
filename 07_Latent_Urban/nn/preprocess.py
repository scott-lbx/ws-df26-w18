# -*- coding: utf-8 -*-
"""
preprocess.py — 讀取建築資料並轉成模型可用的特徵矩陣（繁體中文註解）

流程：
    parquet → 衍生 slender → 補值(中位數) → z-score 標準化 → one-hot 類別
只依賴 pandas / numpy（parquet 由 pyarrow 讀取）。
"""

import numpy as np
import pandas as pd

# 支援「以套件方式匯入」與「直接執行」兩種情境
try:
    from .. import config
except (ImportError, ValueError):
    import config


def load_buildings(slug):
    """
    讀取指定站點的 buildings.parquet，回傳原始 DataFrame，
    並補上衍生欄位 slender = height_m / sqrt(area_m2)。

    參數
    ----
    slug : str
        站點代碼，例如 "lujiazui"。

    回傳
    ----
    pandas.DataFrame
    """
    path = config.DATA_ROOT / slug / "buildings.parquet"
    if not path.exists():
        # 快取缺該城 → 觸發 05 多源拼接管線現建(需 05 config.DATASET_ROOT)
        from . import build05
        path = build05.ensure_city(slug)
    df = pd.read_parquet(path)

    # 衍生「瘦高程度」：面積需 > 0，否則以 NaN 表示（之後會補中位數）
    area = pd.to_numeric(df.get("area_m2"), errors="coerce")
    height = pd.to_numeric(df.get("height_m"), errors="coerce")
    safe_area = area.where(area > 0)  # area <= 0 或 NaN → NaN
    df["slender"] = height / np.sqrt(safe_area)

    return df


def make_dataset(slug=None):
    """
    產生標準化後的特徵矩陣 X，以及對應的欄位名稱與輔助資訊。

    參數
    ----
    slug : str or None
        站點代碼；None 時使用 config.SLUG。

    回傳
    ----
    X : np.ndarray, shape [N, D], dtype float32
        標準化的數值特徵 + one-hot 類別特徵。
    feature_names : list[str], 長度 D
        展開後的欄位名稱（one-hot 會展成 "euluc=<類別>"）。
    aux : dict
        著色/檢視用的原始陣列，至少包含：
        {"euluc", "height_m", "aoi_price", "df"}。
    """
    if slug is None:
        slug = config.SLUG

    # 固定 numpy 亂數種子，確保可重現
    np.random.seed(config.SEED)

    df = load_buildings(slug)

    # ── 數值特徵：強制轉數字 → 中位數補值 → z-score ──────────
    num_cols = list(config.NUMERIC_FEATURES)
    num = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    medians = num.median(axis=0, skipna=True)
    medians = medians.fillna(0.0)          # 整欄皆 NaN 時退回 0
    num = num.fillna(medians)

    mean = num.mean(axis=0)
    std = num.std(axis=0, ddof=0)
    std_safe = std.replace(0.0, 1.0)       # 避免除以 0（常數欄）
    num_z = (num - mean) / std_safe

    num_mat = num_z.to_numpy(dtype=np.float32)
    num_names = list(num_cols)

    # ── 類別特徵：one-hot（NaN 視為一個獨立類別 "NA"）──────────
    cat_mats = []
    cat_names = []
    for col in config.CATEGORICAL_FEATURES:
        s = df[col].astype("object").where(df[col].notna(), "NA")
        dummies = pd.get_dummies(s, prefix=col, prefix_sep="=")
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)  # 欄位順序穩定
        cat_mats.append(dummies.to_numpy(dtype=np.float32))
        cat_names.extend(list(dummies.columns))

    if cat_mats:
        cat_mat = np.concatenate(cat_mats, axis=1)
        X = np.concatenate([num_mat, cat_mat], axis=1).astype(np.float32)
    else:
        X = num_mat

    feature_names = num_names + cat_names

    aux = {
        "euluc": df["euluc"].to_numpy() if "euluc" in df.columns else None,
        "height_m": pd.to_numeric(df.get("height_m"), errors="coerce").to_numpy(),
        "aoi_price": pd.to_numeric(df.get("aoi_price"), errors="coerce").to_numpy(),
        "df": df,
    }

    return X, feature_names, aux
