# -*- coding: utf-8 -*-
"""nn 套件：潛在壓縮教學模組的公開介面。"""

from ._font import set_cjk_font
set_cjk_font()   # import nn 時自動修好 matplotlib 中文(全 notebook 生效)

from .preprocess import load_buildings, make_dataset
from .models import PCA, AE, VAE, train_ae, train_vae
from .viz import scatter_latent, recon_error_table, vae_latent_grid
from . import raster, conv, io, build05

__all__ = [
    "set_cjk_font",
    "load_buildings", "make_dataset",
    "PCA", "AE", "VAE", "train_ae", "train_vae",
    "scatter_latent", "recon_error_table", "vae_latent_grid",
    "raster", "conv", "io", "build05",
]
