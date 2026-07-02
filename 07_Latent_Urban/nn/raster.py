# -*- coding: utf-8 -*-
"""raster.py —— 把城市建築 footprint+高度 光柵化成「高度影像」(供 CNN 用)

流程:
    buildings.parquet(footprint 多邊形 + height_m)
      → rasterize(每格取建築高度)→ 城市高度圖 HxW
      → 切窗成小 patch(64×64…)→ CNN 訓練樣本 [N,1,P,P]

這把「都市形態」變成**影像**,讓卷積網路(Conv-VAE)能看空間紋理,
有別於前面把每棟樓當一列表格向量。教學重點:同一份 05 資料的**兩種張量化**。
"""
from pathlib import Path
import numpy as np

# 05 專案 data 目錄(與 config.DATA_ROOT 對齊)
_DATA = Path(__file__).resolve().parents[2] / "05_Shanghai Power to Form Students" / "data"

HCAP = 150.0   # 高度歸一化上限(m);>HCAP 截斷。lujiazui 有超高層,壓到 [0,1] 方便訓練


def city_height_map(slug, cell_m=10.0):
    """回傳 (img[H,W] 高度圖 float32, meta)。每格 = 落在該格的建築最大高度(m)。"""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.enums import MergeAlg
    from rasterio.transform import from_origin

    g = gpd.read_parquet(_DATA / slug / "buildings.parquet")
    minx, miny, maxx, maxy = g.total_bounds
    W = int((maxx - minx) / cell_m) + 1
    H = int((maxy - miny) / cell_m) + 1
    tr = from_origin(minx, maxy, cell_m, cell_m)
    shapes = [(geom, float(h)) for geom, h in zip(g.geometry, g.height_m)
              if geom is not None and h == h and h > 0]
    img = rasterize(shapes, out_shape=(H, W), transform=tr,
                    merge_alg=MergeAlg.replace, dtype="float32")
    return img, {"slug": slug, "cell_m": cell_m, "bounds": (minx, miny, maxx, maxy)}


def to_patches(img, patch=64, stride=32, min_fill=0.04, hcap=HCAP):
    """城市高度圖 → 歸一化 patch 堆疊 [n,1,patch,patch](值 ∈[0,1]=height/hcap)。
    只保留非空比例 ≥ min_fill 的窗,避免大片空地。"""
    H, W = img.shape
    norm = np.clip(img / hcap, 0, 1).astype("float32")
    out = []
    for y in range(0, max(1, H - patch + 1), stride):
        for x in range(0, max(1, W - patch + 1), stride):
            p = norm[y:y+patch, x:x+patch]
            if p.shape != (patch, patch):
                continue
            if (p > 0).mean() >= min_fill:
                out.append(p)
    if not out:
        return np.zeros((0, 1, patch, patch), "float32")
    return np.stack(out)[:, None, :, :]


def city_patch_dataset(slugs, patch=64, stride=32, cell_m=10.0,
                       min_fill=0.04, hcap=HCAP):
    """多城 → (X[N,1,P,P] float32, slug_labels[N])。CNN 訓練語料。"""
    Xs, labels = [], []
    for slug in slugs:
        img, _ = city_height_map(slug, cell_m)
        P = to_patches(img, patch, stride, min_fill, hcap)
        if len(P):
            Xs.append(P)
            labels += [slug] * len(P)
    X = np.concatenate(Xs, 0) if Xs else np.zeros((0, 1, patch, patch), "float32")
    return X, np.array(labels)
