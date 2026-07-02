# -*- coding: utf-8 -*-
"""
viz.py — 潛在空間視覺化工具（繁體中文註解）

提供三個課堂常用的畫圖 / 比較函式：
    scatter_latent    畫 2D 潛在散佈圖，可依類別或連續值著色。
    recon_error_table 各模型重建誤差比較表。
    vae_latent_grid   在潛在空間鋪網格解碼，看某特徵如何變化。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_latent(Z, color=None, title="", ax=None):
    """
    畫 2D 潛在空間散佈圖。

    參數
    ----
    Z : array [N, 2]
        潛在座標。
    color : array [N] or None
        著色依據；類別型(字串/物件)自動用分類色，數值型用連續色階。
    title : str
    ax : matplotlib Axes or None

    回傳
    ----
    matplotlib.axes.Axes
    """
    Z = np.asarray(Z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if color is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.6)
    else:
        color = np.asarray(color, dtype=object)
        is_numeric = np.issubdtype(np.asarray(pd.to_numeric(pd.Series(color),
                                    errors="coerce")).dtype, np.number) \
                     and not _looks_categorical(color)
        if is_numeric:
            vals = pd.to_numeric(pd.Series(color), errors="coerce").to_numpy()
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=vals, s=8, alpha=0.7, cmap="viridis")
            plt.colorbar(sc, ax=ax)
        else:
            # 類別著色：每個唯一值一種顏色
            cats = pd.Series(color).fillna("NA").astype(str)
            uniq = sorted(cats.unique())
            cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
            for i, u in enumerate(uniq):
                m = (cats == u).to_numpy()
                ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7,
                           color=cmap(i % cmap.N), label=str(u))
            ax.legend(fontsize=6, markerscale=1.5, loc="best")

    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title(title)
    return ax


def _looks_categorical(color):
    """判斷是否為類別型：非數字可轉換比例過高則視為類別。"""
    s = pd.Series(color)
    coerced = pd.to_numeric(s, errors="coerce")
    # 若無法轉成數字的比例高，視為類別
    return coerced.isna().mean() > 0.2


def recon_error_table(errors):
    """
    整理各模型的平均重建 MSE 成比較表。

    參數
    ----
    errors : dict[str, float]
        模型名稱 -> 平均重建 MSE。

    回傳
    ----
    pandas.DataFrame（依 MSE 由小到大排序），並印出。
    """
    df = pd.DataFrame(
        {"model": list(errors.keys()), "recon_mse": list(errors.values())}
    ).sort_values("recon_mse").reset_index(drop=True)
    print(df.to_string(index=False))
    return df


def vae_latent_grid(model, feature_names, grid=7, span=2.5, feature="height_m"):
    """
    在潛在空間 [-span, span]^2 上鋪 grid x grid 網格，逐點解碼，
    觀察某個特徵（預設 height_m 的 z-score）在潛在平面上如何變化。

    參數
    ----
    model : VAE
        需具備 decode(Z)->Xhat 的模型。
    feature_names : list[str]
        欄位名稱（用來定位要觀察的特徵）。
    grid : int
        每軸格點數。
    span : float
        潛在座標範圍（-span ~ span）。
    feature : str
        要觀察的特徵名稱，需存在於 feature_names。

    回傳
    ----
    (decoded, ax) :
        decoded : np.ndarray [grid*grid, D]，網格解碼結果。
        ax      : matplotlib Axes，該特徵在潛在平面上的熱圖。
    """
    axis = np.linspace(-span, span, grid)
    gx, gy = np.meshgrid(axis, axis)
    Z = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    decoded = model.decode(Z)                      # [grid*grid, D]

    if feature in feature_names:
        fi = feature_names.index(feature)
    else:
        fi = 0
        feature = feature_names[0]
    field = decoded[:, fi].reshape(grid, grid)

    _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field, origin="lower",
                   extent=[-span, span, -span, span], cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title(f"'{feature}' across latent grid")
    return decoded, ax
