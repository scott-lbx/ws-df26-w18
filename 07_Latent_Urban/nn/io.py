# -*- coding: utf-8 -*-
"""io.py —— 儲存/載入訓練好的模型(給後續 notebook 重用,不必重訓)

存法:torch.save({state_dict, meta})。meta 記重建模型所需的超參,
載入時先用同參數重建骨架、再灌權重。教學:訓練一次、到處用。
"""
from pathlib import Path
import json
import torch

MODELS = Path(__file__).resolve().parent.parent / "out" / "models"


def save_model(model, name, meta=None):
    """存到 out/models/<name>.pt。meta:重建用超參(dict)。回傳路徑。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    p = MODELS / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, p)
    (MODELS / f"{name}.json").write_text(
        json.dumps(meta or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_into(model, name):
    """把 out/models/<name>.pt 權重灌進已建好的 model 骨架。回傳 (model, meta)。"""
    blob = torch.load(MODELS / f"{name}.pt", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})


def load_meta(name):
    """只讀 meta(超參),供重建骨架。"""
    blob = torch.load(MODELS / f"{name}.pt", weights_only=False)
    return blob.get("meta", {})


def exists(name):
    return (MODELS / f"{name}.pt").exists()
