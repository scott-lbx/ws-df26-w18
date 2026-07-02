# -*- coding: utf-8 -*-
"""
config.py — 助教/學生可調整的參數（繁體中文註解）

這個檔案集中所有「旋鈕」，方便課堂上示範不同設定的效果。
其餘程式碼一律從這裡讀取設定，不要在別處寫死數值。
"""

from pathlib import Path

# ── 資料來源 ────────────────────────────────────────────────
# 要分析的站點 slug（對應 05 專案 data/<slug>/buildings.parquet）
SLUG = "lujiazui"

# 05 專案的 data 目錄（注意資料夾名稱中間有「空格」）。
# 相對於「本檔案」的位置解析，避免寫死絕對路徑而搬機器就壞掉。
DATA_ROOT = Path(__file__).parent.parent / "05_Shanghai Power to Form Students" / "data"

# ── 特徵設定 ────────────────────────────────────────────────
# 數值特徵（會做 z-score 標準化）。
#   slender 為衍生特徵：height_m / sqrt(area_m2)，代表建築的「瘦高程度」。
NUMERIC_FEATURES = ["height_m", "area_m2", "slender", "age", "aoi_price"]

# 類別特徵（會做 one-hot 展開）。euluc = 土地使用分類。
CATEGORICAL_FEATURES = ["euluc"]

# ── 模型設定 ────────────────────────────────────────────────
LATENT_DIM = 2  # 潛在空間維度（示範用固定為 2，方便畫散佈圖）
EPOCHS = 10000  # AE / VAE 訓練回合數
SEED = 0  # 亂數種子，確保課堂結果可重現
