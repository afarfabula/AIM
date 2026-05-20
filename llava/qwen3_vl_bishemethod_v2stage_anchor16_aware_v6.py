"""Qwen3-VL 上的 bishemethod_v2stage_anchor16_aware_v6 适配入口。"""

from __future__ import annotations

import importlib.util
import os
from typing import Any


def _load_apply_bishemethod_to_qwen3_vl():
    vz_path = os.environ.get("QWEN3_VISIONZIP_PATH", "/root/AIM/llava/qwen3_vl_visionzip.py")
    spec = importlib.util.spec_from_file_location("qwen3_vl_visionzip", vz_path)
    vz_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vz_mod)
    return vz_mod.apply_bishemethod_v2stage_anchor16_aware_v6_to_qwen3_vl


def apply_bishemethod_v2stage_anchor16_aware_v6_to_qwen3_vl(model: Any, **kwargs: Any) -> None:
    cfg = {
        "selector": kwargs.pop("selector", "bishe_v2stage_anchor16_aware_v6"),
        "method": kwargs.pop("method", "pred_2d"),
        "metric": kwargs.pop("metric", "max"),
        "threshold": kwargs.pop("threshold", 0.0),
        "vit_prune_layer": kwargs.pop("vit_prune_layer", 0),
        "anchored": kwargs.pop("anchored", True),
        "bishe_target_keep": kwargs.pop("bishe_target_keep", 96),
        "bishe_anchor_points": kwargs.pop("bishe_anchor_points", ["grid4x4"]),
        "bishe_merge_steps": kwargs.pop("bishe_merge_steps", [(192, 192), (144, 144), (96, 96), (48, 48)]),
        "bishe_protected_penalty": kwargs.pop("bishe_protected_penalty", 0.09),
        "bishe_qk_mix": kwargs.pop("bishe_qk_mix", 0.20),
        "bishe_pos_mix": kwargs.pop("bishe_pos_mix", 0.05),
        "bishe_metric_layer": kwargs.pop("bishe_metric_layer", -2),
        "bishe_anchor_order_mode": kwargs.pop("bishe_anchor_order_mode", "block"),
        "bishe_anchor_order_group_size": kwargs.pop("bishe_anchor_order_group_size", 3),
        "bishe_anchor_order_first_stage_only": kwargs.pop("bishe_anchor_order_first_stage_only", True),
    }
    cfg.update(kwargs)
    apply_bishemethod_to_qwen3_vl = _load_apply_bishemethod_to_qwen3_vl()
    apply_bishemethod_to_qwen3_vl(model, **cfg)
