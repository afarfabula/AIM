"""兼容层：旧的 bishe_anchor16_v6 Qwen3-VL token reduction 已废弃。

现在统一复用 AIM 内移植后的 hard-prune 实现，避免继续维护旧的 soft-mask /
多阶段 merge 版本。保留此文件仅为了兼容历史导入路径。
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any


def _load_apply_visionzip_to_qwen3_vl():
    vz_path = os.environ.get("QWEN3_VISIONZIP_PATH", "/root/AIM/llava/qwen3_vl_visionzip.py")
    spec = importlib.util.spec_from_file_location("qwen3_vl_visionzip", vz_path)
    vz_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vz_mod)
    return vz_mod.apply_visionzip_to_qwen3_vl


def apply_bishe_anchor16_v6_to_qwen3_vl(model: Any, **kwargs: Any) -> None:
    print(
        "[AIM Qwen3-VL token reduction] bishe_anchor16_v6 已废弃，"
        "自动切换到统一的 hard-prune 实现。"
    )
    apply_visionzip_to_qwen3_vl = _load_apply_visionzip_to_qwen3_vl()
    apply_visionzip_to_qwen3_vl(model, **kwargs)
