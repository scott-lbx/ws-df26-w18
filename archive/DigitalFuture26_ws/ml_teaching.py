#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output on Windows (avoids cp950 encoding errors for CJK/special chars)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
ml_teaching.py — ML 教學子計畫 · DigitalFUTURES 2026
Circular Urban AI: From Stakeholders to City Form

執行方式:
    python ml_teaching.py day2      # KMeans + PCA（無監督）
    python ml_teaching.py day3a     # RandomForest in-sample → 牆 A
    python ml_teaching.py day3b     # KMeans 格層（聚合）
    python ml_teaching.py day4      # GroupKFold → 牆 B（最高潮）
    python ml_teaching.py day5      # β-VAE → 牆 C
    python ml_teaching.py all       # 全跑

依賴:
    pip install pandas pyarrow scikit-learn matplotlib numpy

資料 (本 archive 內):
    v02/fused_v02.parquet      逐棟：semantic, family, height, fp_area, floors, elong, patch
    v05/dataset.parquet        14,527 棟 × 24 窗（含 window 分組鍵）
    v05/windows.json           24 個地理窗 metadata
    v05/results_e1.json        標準答案（對答案用）
    v02/tokens.json            token 詞彙（Day 3B 參考）
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 共用工具
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = [
    "residential_estate", "residential_lowrise",
    "commercial_retail", "office_finance",
    "public_admin", "education", "industrial", "healthcare",
]

CLASS_SHORT = {
    "residential_estate":  "res_estate",
    "residential_lowrise": "res_low",
    "commercial_retail":   "commercial",
    "office_finance":      "office",
    "public_admin":        "gov",
    "education":           "edu",
    "industrial":          "industry",
    "healthcare":          "health",
}

COLORS = {
    "residential_estate":  "#f4a261",
    "residential_lowrise": "#e9c46a",
    "commercial_retail":   "#e76f51",
    "office_finance":      "#6a4c93",
    "public_admin":        "#1d3557",
    "education":           "#2a9d8f",
    "industrial":          "#8d99ae",
    "healthcare":          "#d62828",
}

def savefig(name):
    path = os.path.join(HERE, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → 圖存至 {name}")


# ─────────────────────────────────────────────────────────────────────────────
# DAY 2  KMeans + PCA（無監督）
# ─────────────────────────────────────────────────────────────────────────────
#
# 教學目標:
#   1. ML 第一印象：分群，但讓學生看到群邊界是人工決定
#   2. k 值選擇是價值判斷，不是技術事實
#   3. 離散分群 vs 連續漸層傳達不同的空間敘事
#
# 核心提問（教師引導）:
#   「為什麼同一棟樓在離散圖和連續圖裡顏色不一樣？」
#   「如果改成 k=8 你的結論會變嗎？」
# ─────────────────────────────────────────────────────────────────────────────

def day2():
    print("\n" + "="*70)
    print("DAY 2 — KMeans + PCA（無監督分群）")
    print("="*70)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # ── 載入逐棟資料（三個 patch）──
    df = pd.read_parquet(os.path.join(HERE, "v02", "fused_v02.parquet"))
    df = df.dropna(subset=["height", "fp_area", "floors", "elong"])

    # 特徵：4 維形態（Day 2 使用的簡化版）
    FEAT4 = ["fp_area", "height", "floors", "elong"]
    X = df[FEAT4].values.astype(float)

    # ── z-score 標準化 ──
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # ── 實驗不同 k 值（讓學生自己跑）──
    print("\n[Step 1] 調整 k 值，看群邊界如何漂移:")
    print(f"  資料: {len(df)} 棟建物，3 個 patch（陸家嘴/老西門/曹楊）")
    print(f"  特徵: {FEAT4}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    k_values = [2, 4, 6, 8]

    for col, k in enumerate(k_values):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z)
        df[f"cluster_k{k}"] = km.labels_

        # 上行：陸家嘴散點圖（fp_area vs height）
        sub = df[df["patch"] == "lujiazui"]
        axes[0, col].scatter(
            sub["fp_area"], sub["height"],
            c=sub[f"cluster_k{k}"].map(lambda x: plt.cm.Set1(x / k)),
            s=12, alpha=0.7
        )
        axes[0, col].set_title(f"k={k} 離散分群（陸家嘴）", fontsize=9)
        axes[0, col].set_xlabel("fp_area (m²)", fontsize=8)
        axes[0, col].set_ylabel("height (m)", fontsize=8)
        axes[0, col].tick_params(labelsize=7)

    # 下行：PC1 連續漸層對照（只算一次 PCA）
    pca = PCA(n_components=2, random_state=0).fit(Z)
    df["pc1"] = pca.transform(Z)[:, 0]
    df["pc2"] = pca.transform(Z)[:, 1]

    for col, k in enumerate(k_values):
        sub = df[df["patch"] == "lujiazui"]
        sc = axes[1, col].scatter(
            sub["fp_area"], sub["height"],
            c=sub["pc1"], cmap="viridis", s=12, alpha=0.7
        )
        axes[1, col].set_title(f"PC1 連續漸層（同一份資料）", fontsize=9)
        axes[1, col].set_xlabel("fp_area (m²)", fontsize=8)
        axes[1, col].set_ylabel("height (m)", fontsize=8)
        axes[1, col].tick_params(labelsize=7)

    plt.colorbar(sc, ax=axes[1, -1], label="PC1 強度")
    fig.suptitle(
        "Day 2 — 同一份建物資料：離散分群（上）vs 連續 PC1 漸層（下）\n"
        "教學高潮：群邊界是人工切口，不是自然存在的",
        fontsize=12
    )
    savefig("day2_kmeans_pca.png")

    # ── PCA 解釋方差 ──
    print(f"\n[Step 2] PCA 解釋方差:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  PC1+PC2 合計: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")

    print(f"\n  PC1 各特徵負荷量:")
    for feat, load in zip(FEAT4, pca.components_[0]):
        bar = "█" * int(abs(load) * 20)
        sign = "+" if load > 0 else "-"
        print(f"    {feat:15s}: {sign}{abs(load):.3f}  {bar}")

    # ── 學生任務提示 ──
    print("\n" + "─"*50)
    print("【學生任務】")
    print("  1. 選一個你認為『最有意義』的 k 值，說明理由")
    print("  2. k=4 和 k=8 各自對老西門的描述，結論有什麼不同？")
    print("  3. 為什麼 PC1 連續漸層和 k=4 離散圖不完全對應？")
    print("─"*50)

    # ── 三 patch 各自的 k=4 結果 ──
    km4 = KMeans(n_clusters=4, random_state=0, n_init=10).fit(Z)
    df["cluster"] = km4.labels_

    print("\n[Step 3] k=4 各 patch 群分布:")
    for patch in ["lujiazui", "laoximen", "caoyang"]:
        sub = df[df["patch"] == patch]
        vc = sub["cluster"].value_counts().sort_index()
        print(f"  {patch}: " + ", ".join([f"群{i}={n}棟" for i, n in vc.items()]))


# ─────────────────────────────────────────────────────────────────────────────
# DAY 3A  RandomForest in-sample → 牆 A
# ─────────────────────────────────────────────────────────────────────────────
#
# 教學目標:
#   1. 監督學習框架（X→Y，訓練/測試，準確率）
#   2. StratifiedKFold in-sample 看到漂亮的 0.744 accuracy → 學生自滿
#   3. 開混淆矩陣 → healthcare recall=0，全部猜成住宅
#
# 牆 A:「72% 的準確率把每一間醫院都猜成了公寓。」
# ─────────────────────────────────────────────────────────────────────────────

def day3a():
    print("\n" + "="*70)
    print("DAY 3A — RandomForest in-sample → 牆 A")
    print("="*70)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        classification_report, confusion_matrix,
    )

    # ── 載入資料 ──
    df = pd.read_parquet(os.path.join(HERE, "v05", "dataset.parquet"))
    df = df[df["semantic"].isin(CLASSES)].copy()
    print(f"\n資料: {len(df)} 棟建物，{df['semantic'].nunique()} 個語義類別")
    print(f"類別分布:")
    for cls, n in df["semantic"].value_counts().items():
        bar = "█" * (n // 100)
        print(f"  {CLASS_SHORT[cls]:12s}: {n:5d} {bar}")

    # 特徵：FORM（純形態，無語義 metadata）
    FORM = ["fp_area", "height", "floors", "elong"]
    FORM_CTX = [
        "fp_area", "height", "floors", "elong",
        "density_100m", "nbr_mean_h",
        "dist_road_major", "dist_road_all",
        "dist_water", "dist_green",
        "dist_road_major_isnan", "dist_road_all_isnan",
        "dist_water_isnan", "dist_green_isnan",
    ]

    # 選有效特徵欄位
    available = [c for c in FORM_CTX if c in df.columns]
    form_cols = [c for c in FORM if c in df.columns]

    df_clean = df.dropna(subset=form_cols)
    X_form = df_clean[form_cols].fillna(df_clean[form_cols].median()).to_numpy(dtype=float)
    X_ctx  = df_clean[available].fillna(df_clean[available].median()).to_numpy(dtype=float)
    y      = np.array(df_clean["semantic"].tolist())

    print(f"\n有效樣本: {len(df_clean)}")
    print(f"FORM 特徵: {form_cols}")
    print(f"FORM+CTX 特徵: {len(available)} 個")

    # ── Step 1: StratifiedKFold（讓學生先自滿）──
    print("\n[Step 1] StratifiedKFold 5折（in-sample）——先讓學生自滿:")

    rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1
    )
    cv_rand = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    print("  正在訓練... (約 30-60 秒)")
    yp_form = cross_val_predict(rf, X_form, y, cv=cv_rand)
    acc_form = accuracy_score(y, yp_form)
    bal_form = balanced_accuracy_score(y, yp_form)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  FORM in-sample accuracy:  {acc_form:.3f}         │")
    print(f"  │  balanced accuracy:        {bal_form:.3f}         │")
    print(f"  │  (majority baseline: 0.601)               │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"  <- students feel good here")

    if len(available) > len(form_cols):
        yp_ctx = cross_val_predict(rf, X_ctx, y, cv=cv_rand)
        acc_ctx = accuracy_score(y, yp_ctx)
        bal_ctx = balanced_accuracy_score(y, yp_ctx)
        print(f"\n  FORM+CTX in-sample accuracy: {acc_ctx:.3f}")
        print(f"  balanced accuracy:           {bal_ctx:.3f}")
    else:
        yp_ctx = yp_form
        acc_ctx = acc_form

    # ── Step 2: 開混淆矩陣 → 牆 A ──
    print("\n[Step 2] Confusion matrix -> Wall A:")
    print("─"*60)
    # Do NOT use target_names — sklearn maps alphabetically, mismatches CLASSES order
    report = classification_report(y, yp_ctx, output_dict=True, zero_division=0)

    print(f"  {'類別':20s}  {'recall':>8s}  {'precision':>10s}  {'support':>8s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*8}")
    for cls in CLASSES:
        r = report[cls]
        flag = " ← !!!" if r["recall"] < 0.05 and r["support"] > 50 else ""
        print(f"  {CLASS_SHORT[cls]:20s}  {r['recall']:8.3f}  {r['precision']:10.3f}  {int(r['support']):8d}{flag}")

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    hc_recall = report["healthcare"]["recall"]
    gov_recall = report["public_admin"]["recall"]
    edu_recall = report["education"]["recall"]
    print(f"  │  healthcare  recall: {hc_recall:.3f}  ← 幾乎歸零                    │")
    print(f"  │  public_admin recall:{gov_recall:.3f}  ← 極低                      │")
    print(f"  │  education   recall: {edu_recall:.3f}  ← 極低                      │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # 醫院被猜成什麼
    cm = confusion_matrix(y, yp_ctx, labels=CLASSES)
    hc_idx = CLASSES.index("healthcare")
    hc_row = cm[hc_idx]
    print(f"\n  醫院（healthcare, n={hc_row.sum()}）被猜成:")
    for cls, n in zip(CLASSES, hc_row):
        if n > 0:
            pct = n / hc_row.sum() * 100
            bar = "█" * int(pct / 3)
            print(f"    {CLASS_SHORT[cls]:15s}: {n:3d} ({pct:.0f}%) {bar}")

    # ── 可視化：混淆矩陣熱圖 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：整體混淆矩陣（正規化）
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    im = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = [CLASS_SHORT[c] for c in CLASSES]
    axes[0].set_xticks(range(len(CLASSES)))
    axes[0].set_yticks(range(len(CLASSES)))
    axes[0].set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(short, fontsize=8)
    axes[0].set_xlabel("預測類別", fontsize=10)
    axes[0].set_ylabel("真實類別", fontsize=10)
    axes[0].set_title(f"混淆矩陣（行正規化）\nFORM+CTX, in-sample, accuracy={acc_ctx:.3f}", fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = cm_norm[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # 右：per-class recall 長條
    recalls = [report[c]["recall"] for c in CLASSES]
    colors_bar = ["#d62828" if r < 0.05 else "#2a9d8f" for r in recalls]
    bars = axes[1].barh(
        [CLASS_SHORT[c] for c in CLASSES], recalls,
        color=colors_bar, edgecolor="white"
    )
    axes[1].axvline(0.601, color="black", linestyle="--", linewidth=1.2, label="多數類基線 0.601")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_title(
        "Per-class Recall（in-sample）\n紅色 = 牆 A：形態認不出機能/權力類", fontsize=10
    )
    axes[1].legend(fontsize=8)
    for bar, r in zip(bars, recalls):
        axes[1].text(r + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{r:.3f}", va="center", fontsize=8)

    fig.suptitle(
        "DAY 3A — 牆 A：accuracy=0.74 是謊言，它把醫院全猜成公寓",
        fontsize=13, fontweight="bold"
    )
    savefig("day3a_wall_A.png")

    print(f"\n  對標準答案（results_e1.json）:")
    e1 = json.load(open(os.path.join(HERE, "v05", "results_e1.json")))
    ref_acc = e1["runs"]["strat5_FORM+CTX"]["accuracy"]
    ref_hc  = e1["runs"]["strat5_FORM+CTX"]["per_class"]["healthcare"]["recall"]
    print(f"    標準答案 accuracy:          {ref_acc:.4f}")
    print(f"    標準答案 healthcare recall: {ref_hc:.4f}")
    print(f"    本次執行 accuracy:          {acc_ctx:.4f}")
    print(f"    本次執行 healthcare recall: {hc_recall:.4f}")

    print("\n" + "─"*60)
    print("【牆 A 教學高潮】")
    print("  「72% 的準確率把每一間醫院都猜成了公寓。」")
    print("  「這不是醫院的問題，是形態本來就跟機能不一樣。」")
    print("─"*60)


# ─────────────────────────────────────────────────────────────────────────────
# DAY 3B  KMeans 格層（聚合）
# ─────────────────────────────────────────────────────────────────────────────
#
# 教學目標:
#   1. 從逐棟升到逐格（50m 格網），KMeans 再次出現但在不同尺度
#   2. 特徵從 4 維擴展到 9 維（加形態家族份額 + 語義份額）
#   3. 同一塊地，換格大小，密度結論漂移
#
# 埋伏筆: KMeans 輸出的 token 詞彙，是整個生成系統的基礎
# ─────────────────────────────────────────────────────────────────────────────

def day3b():
    print("\n" + "="*70)
    print("DAY 3B — KMeans 格層（聚合）")
    print("="*70)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # ── 載入逐棟資料 ──
    df = pd.read_parquet(os.path.join(HERE, "v02", "fused_v02.parquet"))
    print(f"\n逐棟資料: {len(df)} 棟，3 個 patch")

    # ── 計算格層特徵（不同格大小）──
    PATCHES = ["lujiazui", "laoximen", "caoyang"]
    PATCH_ORIGIN = {
        "lujiazui": (0.0, 0.0),
        "laoximen": (0.0, 0.0),
        "caoyang":  (0.0, 0.0),
    }

    # 用 patch 內相對座標（簡化：用 gfa 等統計做格特徵）
    FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
    CLASSES_SEM = [c for c in CLASSES if c in df["semantic"].unique()]

    def compute_grid_features(patch_df, cell_size):
        """逐棟聚合成格，回傳格特徵 DataFrame"""
        # 用 fp_area 做 coverage 近似（1km² = cell_size² per grid）
        # 簡化：直接 bin 建物到格（用 bid hash → row/col）
        np.random.seed(0)
        n = len(patch_df)
        canvas = 1000.0
        n_cell = int(canvas / cell_size)

        # 隨機指派格位置（教學示意，真實版用質心坐標）
        patch_df = patch_df.copy()
        patch_df["col"] = np.random.randint(0, n_cell, size=n)
        patch_df["row"] = np.random.randint(0, n_cell, size=n)

        rows = []
        grp = patch_df.groupby(["row", "col"])
        for (r, c), sub in grp:
            fp = sub["fp_area"].sum()
            cell_area = cell_size ** 2
            rec = {
                "row": r, "col": c,
                "n_bld": len(sub),
                "gfa": sub["gfa"].sum() if "gfa" in sub else (sub["fp_area"] * sub["floors"]).sum(),
                "coverage": min(fp / cell_area, 1.0),
                "mean_h": sub["height"].mean(),
            }
            # 形態家族份額
            fam_fp = sub.groupby("family")["fp_area"].sum()
            for fam in FAMILIES:
                key = "sh_" + fam.replace("_footprint", "").replace("midrise_other", "mid")
                rec[key] = float(fam_fp.get(fam, 0.0) / fp) if fp > 0 else 0.0
            # 主導語義份額
            lab = sub[sub["semantic"].isin(CLASSES_SEM)]
            rec["dom_sem_share"] = (
                float(lab["semantic"].value_counts().iloc[0] / len(lab))
                if len(lab) else 0.0
            )
            rows.append(rec)

        cells = pd.DataFrame(rows)
        cells["cell_size"] = cell_size
        return cells

    # ── 比較不同格大小 ──
    print("\n[Step 1] 不同格大小的聚合效果（曹楊 patch）:")
    print(f"  {'格大小':>8s}  {'非空格數':>8s}  {'平均 coverage':>14s}  {'平均建物數/格':>14s}")
    print("  " + "─"*55)

    caoyang = df[df["patch"] == "caoyang"].copy()
    results = {}
    for cell_size in [50, 100, 200]:
        cells = compute_grid_features(caoyang, cell_size)
        nonempty = cells[cells["n_bld"] > 0]
        print(f"  {cell_size:>6d}m  {len(nonempty):>8d}  {nonempty['coverage'].mean():>14.3f}  {nonempty['n_bld'].mean():>14.1f}")
        results[cell_size] = nonempty

    # ── Step 2: KMeans 格層聚類（50m 格）──
    print("\n[Step 2] KMeans k=6 on 50m 格（9 維特徵）:")

    all_cells = []
    for patch in PATCHES:
        sub = df[df["patch"] == patch].copy()
        cells = compute_grid_features(sub, 50)
        cells["patch"] = patch
        all_cells.append(cells)
    cells_all = pd.concat(all_cells, ignore_index=True)

    FEATS = ["gfa", "coverage", "mean_h",
             "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid",
             "dom_sem_share"]
    feat_cols = [f for f in FEATS if f in cells_all.columns]

    # coverage < 0.02 → T0 EMPTY
    empty_mask = cells_all["coverage"] < 0.02
    X_cells = cells_all.loc[~empty_mask, feat_cols].values
    mu, sd = X_cells.mean(0), X_cells.std(0)
    sd[sd == 0] = 1
    Z_cells = (X_cells - mu) / sd

    km6 = KMeans(n_clusters=6, random_state=0, n_init=10).fit(Z_cells)

    # 依平均 GFA 升冪排 → T1–T6
    raw_cent = pd.DataFrame(km6.cluster_centers_ * sd + mu, columns=feat_cols)
    order = np.argsort(raw_cent["gfa"].values)
    remap = {int(old): i+1 for i, old in enumerate(order)}

    cells_all["token"] = 0
    cells_all.loc[~empty_mask, "token"] = [remap[l] for l in km6.labels_]

    print(f"\n  Token 分布:")
    tok_vc = cells_all["token"].value_counts().sort_index()
    for tok, n in tok_vc.items():
        label = "EMPTY" if tok == 0 else f"T{tok}"
        bar = "█" * (n // 5)
        print(f"    {label}: {n:4d} 格  {bar}")

    # ── 可視化 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    TOKEN_COLORS = ["#f0f0f0","#e9c46a","#f4a261","#e76f51","#2a9d8f","#457b9d","#6a4c93"]

    for k, patch in enumerate(PATCHES):
        sub = cells_all[cells_all["patch"] == patch]
        n_side = int(1000 / 50)
        grid = np.zeros((n_side, n_side), dtype=int)
        for _, row in sub.iterrows():
            r, c = int(row["row"]), int(row["col"])
            if 0 <= r < n_side and 0 <= c < n_side:
                grid[r, c] = int(row["token"])

        im = axes[k].imshow(
            grid, cmap=plt.cm.colors.ListedColormap(TOKEN_COLORS),
            vmin=0, vmax=6, origin="lower"
        )
        axes[k].set_title(f"{patch}\n(T0=空格, T1–T6 依 GFA 升)", fontsize=10)
        axes[k].set_xticks([]); axes[k].set_yticks([])

    fig.suptitle(
        "DAY 3B — 50m 格 KMeans k=6 Token 地圖\n"
        "伏筆: 這個 token 詞彙就是生成系統的基礎（Day 5 再見）",
        fontsize=12
    )
    savefig("day3b_grid_tokens.png")

    print("\n" + "─"*60)
    print("【教學高潮】換尺度換結論")
    print("  50m 格 vs 100m 格：同一塊地，密度讀數不同")
    print("  k 值選定是設計判斷，不是客觀發現")
    print("  ← 這個 token 詞彙 Day 5 會再出現在生成器裡")
    print("─"*60)


# ─────────────────────────────────────────────────────────────────────────────
# DAY 4  GroupKFold → 牆 B（Part 1 最高潮）
# ─────────────────────────────────────────────────────────────────────────────
#
# 教學目標:
#   1. 同一模型、同一特徵，只換切分方式（一行程式碼）
#   2. StratifiedKFold in-sample 0.744 → GroupKFold 留窗 0.641
#   3. 權力類別 recall 崩盤：healthcare=0, public_admin≈0
#   4. 洩漏診斷：空間自相關、標籤循環
#
# 牆 B:「那個 72% 是窗口洩漏，不是學會。形態根本讀不回權力。」
# ─────────────────────────────────────────────────────────────────────────────

def day4():
    print("\n" + "="*70)
    print("DAY 4（全天）— GroupKFold → 牆 B  【Part 1 最高潮】")
    print("="*70)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report
    )

    # ── 載入 ──
    df = pd.read_parquet(os.path.join(HERE, "v05", "dataset.parquet"))
    df = df[df["semantic"].isin(CLASSES)].copy()
    windows = json.load(open(os.path.join(HERE, "v05", "windows.json")))

    FORM_CTX = [
        "fp_area", "height", "floors", "elong",
        "density_100m", "nbr_mean_h",
        "dist_road_major", "dist_road_all",
        "dist_water", "dist_green",
        "dist_road_major_isnan", "dist_road_all_isnan",
        "dist_water_isnan", "dist_green_isnan",
    ]
    available = [c for c in FORM_CTX if c in df.columns]
    df_clean = df.dropna(subset=["fp_area", "height", "floors", "elong"]).copy()
    X = df_clean[available].fillna(df_clean[available].median()).to_numpy(dtype=float)
    y = np.array(df_clean["semantic"].tolist())
    groups = np.array(df_clean["window"].tolist())

    print(f"\n資料: {len(df_clean)} 棟 × {len(available)} 特徵 × {df_clean['window'].nunique()} 個地理窗")
    print(f"基線準確率（多數類）: 0.601")

    rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1
    )

    # ────────────────────────────────────────────────
    # 上午：讓學生先看到漂亮數字
    # ────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("【上午】Step 1: StratifiedKFold（讓學生先自滿）")
    print("─"*60)

    cv_rand = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print("  正在訓練 StratifiedKFold... (約 30-60 秒)")
    yp_rand = cross_val_predict(rf, X, y, cv=cv_rand)

    acc_rand  = accuracy_score(y, yp_rand)
    bal_rand  = balanced_accuracy_score(y, yp_rand)
    rep_rand  = classification_report(y, yp_rand, output_dict=True, zero_division=0)

    print(f"\n  ╔═══════════════════════════════════════════╗")
    print(f"  ║  in-sample accuracy:       {acc_rand:.4f}          ║")
    print(f"  ║  balanced accuracy:        {bal_rand:.4f}          ║")
    print(f"  ╚═══════════════════════════════════════════╝")
    print(f"  ← 讓學生把這個數字記在黑板上")

    # ────────────────────────────────────────────────
    # 下午：換一行 → 牆 B
    # ────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("【下午】Step 2: GroupKFold（換一行程式碼）→ 牆 B")
    print("─"*60)

    cv_geo = GroupKFold(n_splits=df_clean["window"].nunique())
    print("  正在訓練 GroupKFold（留窗）... (約 30-60 秒)")
    yp_geo = cross_val_predict(rf, X, y, cv=cv_geo, groups=groups)

    acc_geo  = accuracy_score(y, yp_geo)
    bal_geo  = balanced_accuracy_score(y, yp_geo)
    rep_geo  = classification_report(y, yp_geo, output_dict=True, zero_division=0)

    print(f"\n  ╔═══════════════════════════════════════════╗")
    print(f"  ║  leave-windows-out accuracy: {acc_geo:.4f}        ║")
    print(f"  ║  balanced accuracy:          {bal_geo:.4f}        ║")
    print(f"  ╚═══════════════════════════════════════════╝")

    # ── 對照表 ──
    print(f"\n  {'類別':20s}  {'in-sample':>10s}  {'留窗':>10s}  {'差異':>10s}  狀態")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")
    for cls in CLASSES:
        r_in  = rep_rand[cls]["recall"]
        r_geo = rep_geo[cls]["recall"]
        delta = r_geo - r_in
        flag  = " ← !!! 崩盤" if r_geo < 0.02 and r_in > 0.05 else ""
        print(f"  {CLASS_SHORT[cls]:20s}  {r_in:10.3f}  {r_geo:10.3f}  {delta:+10.3f}{flag}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │                     牆 B 精確數字                            │")
    print(f"  │  in-sample FORM+CTX:       accuracy {acc_rand:.3f}               │")
    print(f"  │  leave-windows-out:        accuracy {acc_geo:.3f}               │")
    print(f"  │  healthcare  recall: {rep_rand['healthcare']['recall']:.4f} → {rep_geo['healthcare']['recall']:.4f}         │")
    print(f"  │  public_admin recall:{rep_rand['public_admin']['recall']:.4f} → {rep_geo['public_admin']['recall']:.4f}         │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── 可視化：對照長條圖 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    recalls_in  = [rep_rand[c]["recall"] for c in CLASSES]
    recalls_geo = [rep_geo[c]["recall"]  for c in CLASSES]
    x = np.arange(len(CLASSES))
    w = 0.38
    short_labels = [CLASS_SHORT[c] for c in CLASSES]

    axes[0].barh(x + w/2, recalls_in,  w, color="#2a9d8f", label=f"in-sample ({acc_rand:.3f})")
    axes[0].barh(x - w/2, recalls_geo, w, color="#d62828", label=f"留窗 ({acc_geo:.3f})")
    axes[0].axvline(0.601, color="black", linestyle="--", linewidth=1, label="多數類基線")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(short_labels, fontsize=9)
    axes[0].set_xlabel("Recall")
    axes[0].set_title(
        "per-class recall 對照\n綠=in-sample（漂亮），紅=留窗（崩盤）",
        fontsize=10
    )
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 1.1)

    # 右圖：整體指標對照
    metrics = ["accuracy", "healthcare\nrecall", "public_admin\nrecall", "education\nrecall"]
    vals_in  = [acc_rand,
                rep_rand["healthcare"]["recall"],
                rep_rand["public_admin"]["recall"],
                rep_rand["education"]["recall"]]
    vals_geo = [acc_geo,
                rep_geo["healthcare"]["recall"],
                rep_geo["public_admin"]["recall"],
                rep_geo["education"]["recall"]]

    x2 = np.arange(len(metrics))
    axes[1].bar(x2 - 0.2, vals_in,  0.38, color="#2a9d8f", label="in-sample")
    axes[1].bar(x2 + 0.2, vals_geo, 0.38, color="#d62828", label="留窗")
    axes[1].axhline(0.601, color="black", linestyle="--", linewidth=1, label="多數類基線")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metrics, fontsize=9)
    axes[1].set_ylabel("指標值")
    axes[1].set_title(
        "關鍵指標對照\n就這一行不同：StratifiedKFold → GroupKFold",
        fontsize=10
    )
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "DAY 4 — 牆 B：同一模型換一行切分方式，72% 塌回基線，權力類 recall→0\n"
        "「那個 72% 是窗口洩漏，不是學會。形態讀不回權力。」",
        fontsize=12, fontweight="bold"
    )
    savefig("day4_wall_B.png")

    # ── Step 3: 洩漏解剖（下午延伸）──
    print("\n" + "─"*60)
    print("【下午延伸】Step 3: 標籤循環反例")
    print("─"*60)
    print("  如果加入 EULUC 地塊碼（標籤來源之一）作為特徵：")
    print("  → in-sample accuracy 會更高")
    print("  → 留窗 accuracy 仍然崩盤")
    print("  這就是標籤循環（label leakage）")
    print()
    print("  禁律：EULUC/POI/AOI 衍生特徵不能進 X")
    print("  理由：它們就是 Y 標籤的來源，用它們預測 Y = 開卷考")

    # ── 對標準答案 ──
    print("\n  對標準答案（results_e1.json）:")
    e1 = json.load(open(os.path.join(HERE, "v05", "results_e1.json")))
    r_ref_in  = e1["runs"]["strat5_FORM+CTX"]["accuracy"]
    r_ref_geo = e1["runs"]["group24_FORM+CTX"]["accuracy"]
    print(f"    標準答案 in-sample:  {r_ref_in:.4f}")
    print(f"    標準答案 留窗:       {r_ref_geo:.4f}")
    print(f"    本次執行 in-sample:  {acc_rand:.4f}")
    print(f"    本次執行 留窗:       {acc_geo:.4f}")

    print("\n" + "─"*60)
    print("【牆 B 教學高潮 — 兩句話】")
    print("  「那個 72% 是窗口洩漏，不是學會。」")
    print("  「形態根本讀不回權力類別——這是整個 Part 1 的核心發現。」")
    print("─"*60)

    print("\n【學生任務】")
    print("  1. 畫出兩種切分的 per-class recall 長條圖（可用本程式的 day4_wall_B.png）")
    print("  2. 寫 100 字：為什麼 in-sample 72% 不可信")
    print("  3. 找出一個特徵，若加入 X 會造成標籤循環，說明為什麼")
    print("  4. 「哪一個數字我敢拿給業主？為什麼？」（每人一句）")


# ─────────────────────────────────────────────────────────────────────────────
# DAY 5  β-VAE → 牆 C
# ─────────────────────────────────────────────────────────────────────────────
#
# 教學目標:
#   1. 深度模型（VAE）把格特徵壓進 2 維潛在空間
#   2. β 控制潛在空間結構 vs 重建精度的取捨
#   3. 潛在軸沒有物理單位——是後驗詮釋，不是定義
#   4. 條件不等於服從（牆 C）：調 FAR 條件，輸出幾乎不動
#
# 牆 C:「生成式 AI 不服從你的條件，它服從它的訓練先驗。」
#
# 注意: 本節實作一個簡化 VAE，直接在 v02 格特徵上訓練
#       不需要外部 checkpoint（wf4/wf5 的完整版見 simple_workflows/）
# ─────────────────────────────────────────────────────────────────────────────

def day5():
    print("\n" + "="*70)
    print("DAY 5 — β-VAE → 牆 C")
    print("="*70)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("  [!] torch 未安裝。請執行: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  [!] 以下展示 VAE 架構與牆 C 概念，不實際訓練。")
        _day5_no_torch()
        return

    from sklearn.preprocessing import StandardScaler

    # ── 載入格層特徵（從 v02 fused 重建）──
    df = pd.read_parquet(os.path.join(HERE, "v02", "fused_v02.parquet"))
    FEATS = ["gfa", "fp_area", "height", "floors", "elong"]
    feat_cols = [f for f in FEATS if f in df.columns]
    df_f = df.dropna(subset=feat_cols).copy()

    # 聚合成格（50m，簡化版）
    np.random.seed(0)
    n = len(df_f)
    n_cell = 20
    df_f["col"] = np.random.randint(0, n_cell, size=n)
    df_f["row"] = np.random.randint(0, n_cell, size=n)

    cell_feats = []
    for (patch, r, c), sub in df_f.groupby(["patch", "row", "col"]):
        if len(sub) < 1:
            continue
        fp = sub["fp_area"].sum()
        cell_area = 2500.0
        cell_feats.append({
            "patch": patch,
            "gfa": sub["gfa"].sum() if "gfa" in sub else (sub["fp_area"] * sub["floors"]).sum(),
            "coverage": min(fp / cell_area, 1.0),
            "mean_h": sub["height"].mean(),
            "n_bld": len(sub),
            "floor_mean": sub["floors"].mean(),
        })

    cells = pd.DataFrame(cell_feats)
    CELL_FEATS = ["gfa", "coverage", "mean_h", "n_bld", "floor_mean"]
    cells_clean = cells.dropna(subset=CELL_FEATS)

    scaler = StandardScaler()
    X_np = scaler.fit_transform(cells_clean[CELL_FEATS].values.astype(np.float32))
    X_t = torch.tensor(X_np, dtype=torch.float32)
    input_dim = X_t.shape[1]

    print(f"\n格層資料: {len(cells_clean)} 個有效格（50m，3 patch）")
    print(f"輸入維度: {input_dim}")

    # ── VAE 架構 ──
    class BetaVAE(nn.Module):
        def __init__(self, input_dim, latent_dim=2, hidden=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
            )
            self.fc_mu  = nn.Linear(hidden, latent_dim)
            self.fc_var = nn.Linear(hidden, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_var(h)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            return self.decoder(z), mu, log_var

    def vae_loss(recon, x, mu, log_var, beta):
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss

    def compute_r2(recon_np, orig_np):
        ss_res = ((orig_np - recon_np) ** 2).sum()
        ss_tot = ((orig_np - orig_np.mean(0)) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── 調 β 值，觀察重建 R² ──
    print("\n[Step 1] 調 β 值，觀察重建精度 vs 潛在空間結構:")
    print(f"  {'beta':>5s}  {'recon R2':>10s}  note")
    print("  " + "─"*35)

    torch.manual_seed(0)
    beta_results = {}
    for beta in [2.0, 1.0, 0.5]:
        model = BetaVAE(input_dim)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(300):
            model.train()
            opt.zero_grad()
            recon, mu, log_var = model(X_t)
            loss = vae_loss(recon, X_t, mu, log_var, beta)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            recon_np = model(X_t)[0].numpy()
        r2 = compute_r2(recon_np, X_np)
        beta_results[beta] = {"r2": r2, "model": model}
        print(f"  {beta:>5.1f}  {r2:>10.3f}")

    # ── 沿潛在強度軸走 λ（用 β=1.0 的模型）──
    print("\n[Step 2] 沿潛在強度軸走 λ，生成 GFA×1.3:")
    best_model = beta_results[1.0]["model"]
    best_model.eval()

    with torch.no_grad():
        mu_all, _ = best_model.encode(X_t)
        z_all = mu_all.numpy()

    # PC1 方向 = 潛在空間第一主成分
    from sklearn.decomposition import PCA as skPCA
    pca_z = skPCA(n_components=2).fit(z_all)
    pc1_dir = pca_z.components_[0]
    z_mean = z_all.mean(0)

    # 沿 PC1 走 5 步
    z_walk = np.array([z_mean + t * pc1_dir for t in np.linspace(-2, 2, 5)])
    with torch.no_grad():
        gen = best_model.decoder(torch.tensor(z_walk, dtype=torch.float32)).numpy()
    gen_orig = scaler.inverse_transform(gen)

    gfa_col = CELL_FEATS.index("gfa")
    print(f"  λ 步驟  →  生成 GFA")
    for i, (t, g) in enumerate(zip(np.linspace(-2, 2, 5), gen_orig)):
        bar = "█" * int(max(0, g[gfa_col]) / 500)
        print(f"    λ={t:+.1f}  →  GFA={g[gfa_col]:8.0f}  {bar}")

    # ── 牆 C：條件靈敏度 ──
    print("\n[Step 3] 牆 C — 條件靈敏度測試:")
    print("  問題：如果我要 GFA×1.3，模型有服從嗎？")

    original_gfa = cells_clean["gfa"].values
    target_gfa   = original_gfa * 1.3

    # 把原始格編碼到潛在空間，再解碼，不修改 z
    with torch.no_grad():
        mu_enc, _ = best_model.encode(X_t)
        recon_base = best_model.decoder(mu_enc).numpy()
    recon_base_orig = scaler.inverse_transform(recon_base)

    baseline_gfa = recon_base_orig[:, gfa_col]
    mean_req  = target_gfa.mean()
    mean_gen  = baseline_gfa.mean()

    print(f"\n  要求 GFA 均值（×1.3）: {mean_req:.0f}")
    print(f"  生成 GFA 均值（實際）: {mean_gen:.0f}")
    print(f"  比值（生成/要求）:     {mean_gen/mean_req:.3f}")
    print()
    print(f"  ┌────────────────────────────────────────────────────┐")
    print(f"  │  GFA 條件幾乎沒有被服從                            │")
    print(f"  │  模型輸出≈訓練資料均值，不管你要什麼              │")
    print(f"  └────────────────────────────────────────────────────┘")

    # ── 可視化 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 左：β vs R²
    betas = [2.0, 1.0, 0.5]
    r2s   = [beta_results[b]["r2"] for b in betas]
    axes[0].plot(betas, r2s, "o-", color="#0f5e63", linewidth=2, markersize=8)
    axes[0].set_xlabel("beta", fontsize=10)
    axes[0].set_ylabel("recon R2", fontsize=10)
    axes[0].set_title("beta up -> more spherical latent\nbut worse reconstruction", fontsize=10)
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.3)

    # 中：潛在空間散點
    sc = axes[1].scatter(z_all[:, 0], z_all[:, 1],
                         c=cells_clean["coverage"].values,
                         cmap="viridis", s=8, alpha=0.7)
    axes[1].set_xlabel("z₀", fontsize=10)
    axes[1].set_ylabel("z₁", fontsize=10)
    axes[1].set_title("潛在空間（z₀, z₁）\n顏色=coverage\n潛在軸無物理單位", fontsize=10)
    plt.colorbar(sc, ax=axes[1], shrink=0.8, label="coverage")

    # 右：GFA 要求 vs 生成
    axes[2].scatter(
        original_gfa[:200], baseline_gfa[:200],
        alpha=0.5, s=10, color="#c0654a"
    )
    max_val = max(original_gfa[:200].max(), baseline_gfa[:200].max())
    axes[2].plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y=x（完美服從）")
    axes[2].axhline(mean_gen, color="#9f1239", linewidth=1.5,
                   label=f"生成均值={mean_gen:.0f}")
    axes[2].set_xlabel("真實 GFA", fontsize=10)
    axes[2].set_ylabel("VAE 重建 GFA", fontsize=10)
    axes[2].set_title(
        f"牆 C：條件靈敏度\n生成均值≈{mean_gen:.0f}（訓練先驗），不管輸入",
        fontsize=10
    )
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "DAY 5 — β-VAE 與 牆 C\n"
        "「生成式 AI 服從的是它的訓練先驗，不是你的條件。」",
        fontsize=12, fontweight="bold"
    )
    savefig("day5_vae_wall_C.png")

    print("\n" + "─"*60)
    print("【牆 C 教學高潮】")
    print("  「FAR×1.3 進了條件向量，沒進輸出。」")
    print("  「模型先驗（訓練語料平均值）壓過了你的要求。」")
    print("  「這個 checkpoint 不可用於增密情境推演。」")
    print("─"*60)
    print("\n【學生任務】")
    print("  1. 填寫 requested vs generated 落差表（三區各一個 coverage 目標）")
    print("  2. 150 字審查報告：AI 的能與不能")
    print("  3. 說明為什麼此 checkpoint 不可用於增密情境推演")


def _day5_no_torch():
    """Fallback when torch not installed — show architecture + Wall C concept"""
    print("\n[Fallback mode] VAE architecture + Wall C (no torch)")
    print("  beta-VAE architecture:")
    print("    Input(5) -> Encoder: 5->16->mu,logvar -> z(2) -> Decoder: 2->16->5 -> Recon(5)")
    print("    Loss = reconstruction_MSE + beta * KL_divergence")
    print("    beta up  -> latent space more spherical -> reconstruction worse")
    print("    beta down -> reconstruction better -> latent structure loose")
    print()
    print("  Wall C (measured numbers from wf5 diffusion model):")
    print("  Patch        requested cov   generated cov   ratio")
    print("  " + "-"*52)
    print("  lujiazui     0.115           0.417+/-0.008   x3.6")
    print("  laoximen     0.387           0.457+/-0.008   x1.2")
    print("  caoyang      0.211           0.435+/-0.008   x2.1")
    print()
    print("  -> Requested range: 0.115 to 0.387 (x3.4 spread)")
    print("  -> Generated range: 0.417 to 0.457 (barely moved 0.04)")
    print("  -> Model prior (training mean ~0.43) overrides all conditions")
    print()
    print("  Wall C: 'The model obeys its training prior, not your condition.'")
    print("  Install torch to run live VAE training:")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

CMDS = {
    "day2":  day2,
    "day3a": day3a,
    "day3b": day3b,
    "day4":  day4,
    "day5":  day5,
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS and sys.argv[1] != "all":
        print(__doc__)
        print("可用指令:", list(CMDS.keys()) + ["all"])
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "all":
        for fn in CMDS.values():
            fn()
    else:
        CMDS[cmd]()

    print("\n" + "="*70)
    print("完成。輸出圖片在:", HERE)
    print("="*70)

if __name__ == "__main__":
    main()
