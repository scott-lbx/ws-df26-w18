# -*- coding: utf-8 -*-
"""regime_latent —— 「城市 × 权力体制」形态潜空间分析

有别于 07 的建築層 nn(單城建築向量),本子套件的分析單位是
**城市 × 体制的 9 维形态聚合**(measure.diagnose;60 样本)。

用法:
    from regime_latent import corpus, analysis, interpret
    corpus.build()        # 现建 60×9 语料 → out/features.csv
    analysis.main()       # PCA/AE/VAE 训练 + 出图 1-3
    interpret.main()      # 群聚可解释 + 4 体制 → 出图 4-5
产物全部落在 07_Latent_Urban/out/,保持 05 乾淨。
"""
from . import _paths, encoders, corpus, analysis, interpret, generate, cluster  # noqa: F401

__all__ = ["_paths", "encoders", "corpus", "analysis", "interpret",
           "generate", "cluster"]
