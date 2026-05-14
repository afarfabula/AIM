"""
Token Prune Strategies - 可扩展的 token 剪枝策略框架
支持多种策略：AIM、LLaVA-PruMerge、VisionZip
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import importlib


class TokenPruneStrategy(ABC):
    """Token 剪枝策略的基类"""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置字典
        """
        self.config = config

    @abstractmethod
    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        应用剪枝策略
        
        Args:
            model: 视觉模型 (CLIPVisionTower)
            images: 输入图像张量
            image_features: 原始图像特征
            
        Returns:
            剪枝后的图像特征
        """
        pass

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """获取策略的默认配置"""
        return {}


# 策略注册表
STRATEGY_REGISTRY: Dict[str, TokenPruneStrategy] = {}
# 全局策略实例（用于在推理时应用）
_GLOBAL_STRATEGY: Optional[TokenPruneStrategy] = None
_GLOBAL_STRATEGY_NAME: Optional[str] = None


def register_strategy(name: str):
    """策略注册装饰器"""
    def decorator(strategy_cls):
        STRATEGY_REGISTRY[name] = strategy_cls
        return strategy_cls
    return decorator


def get_strategy(name: str, config: Optional[Dict[str, Any]] = None) -> TokenPruneStrategy:
    """
    获取策略实例
    
    Args:
        name: 策略名称
        config: 策略配置，如果为 None 则使用默认配置
        
    Returns:
        策略实例
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_cls = STRATEGY_REGISTRY[name]
    if config is None:
        config = strategy_cls.get_default_config()
    
    return strategy_cls(config)


# ==================== 策略实现 ====================

@register_strategy("none")
class NoPruneStrategy(TokenPruneStrategy):
    """不进行任何剪枝的策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {}


@register_strategy("aim")
class AIMStrategy(TokenPruneStrategy):
    """AIM (AIM) 策略 - 双边软匹配合并"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 导入 AIM 所需的模块
        try:
            from llava.model.token_merge import bipartite_soft_matching_merge
            self.bipartite_merge = bipartite_soft_matching_merge
        except ImportError:
            self.bipartite_merge = None

    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            return image_features

        metric = None
        if model is not None and images is not None:
            try:
                if isinstance(images, list) and len(images) > 0:
                    images = images[0]
                if isinstance(images, torch.Tensor) and images.dim() == 3:
                    images = images.unsqueeze(0)

                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    vision_tower = getattr(model, "vision_tower", None)
                    inner = getattr(vision_tower, "vision_tower", vision_tower)
                    vision_model = getattr(inner, "vision_model", None)
                    if vision_model is not None:
                        outputs = {}

                        def hook_k(module, input, output):
                            outputs["desired_k"] = output

                        layers = vision_model.encoder.layers
                        hook_handle = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                        _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                        hook_handle.remove()
                        metric = outputs.get("desired_k", None)
            except Exception:
                metric = None

        if metric is None:
            metric = image_features
        elif metric.shape[1] == image_features.shape[1] + 1:
            metric = metric[:, 1:, :]
        elif metric.shape[1] != image_features.shape[1]:
            metric = image_features

        orig_num = image_features.shape[1]
        for r_feat, r_metric in self.config.get("merge_steps", [(288, 288), (36, 36)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            merged_feat = self.bipartite_merge(metric=metric, r=r_feat, x=image_features)
            if merged_feat is None:
                break
            merged_metric = self.bipartite_merge(metric=metric, r=r_metric, x=metric)
            image_features = merged_feat
            metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features

        if image_features.shape[1] != orig_num:
            print(f"AIM merged tokens: {orig_num} -> {image_features.shape[1]}")
        
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "merge_steps": [(288, 288), (36, 36)]  # (r_for_feature, r_for_key)
        }


@register_strategy("aim96")
class AIM96Strategy(AIMStrategy):
    """AIM-96: 在 AIM 的基础上增加 merge stage，把视觉 tokens 压到两位数（默认 96）。

    设计目标：
    - token 数从 576 -> 96（约 83% reduction），满足“两位数 token”要求
    - 仍尽量保持 AIM 的 query-aware 选择能力（比纯 vision-only pruning 更稳）
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        # bipartite soft matching merge 每步最多合并一半 tokens。
        # 576 -> 288 (r=288)
        # 288 -> 144 (r=144)
        # 144 ->  96 (r=48)
        return {
            "merge_steps": [(288, 288), (144, 144), (48, 48)]
        }


@register_strategy("bishemethod")
class BisheMethodStrategy(AIMStrategy):
    """BisheMethod: AIM96-style multi-stage soft-merge + Anchor protection.

    目标：
    - 两位数 token：默认 576 -> 96
    - GQA limit=2000 达到 0.57+（以 AIM96 的强表现为基线）
    - 算法自主创新点：在 soft-merge 的 matching 过程中引入“空间锚点保护”，避免关键位置被早期合并掉
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.anchor_points = config.get(
            "anchor_points",
            ["tl", "tr", "bl", "br", "center", "top", "bottom", "left", "right"],
        )

    def _anchor_indices(self, num_tokens: int, device) -> torch.Tensor:
        # Supports patch-only (576=24x24) and CLS+patch (577) layouts.
        side = int(num_tokens ** 0.5)
        offset = 0
        if side * side != num_tokens:
            num_spatial = num_tokens - 1
            side = int(num_spatial ** 0.5)
            if side * side != num_spatial:
                return torch.empty(0, device=device, dtype=torch.long)
            offset = 1
        point_to_idx = {
            "tl": 0,
            "tr": side - 1,
            "bl": (side - 1) * side,
            "br": (side - 1) * side + side - 1,
            "center": (side // 2) * side + side // 2,
            "top": side // 2,
            "bottom": (side - 1) * side + side // 2,
            "left": (side // 2) * side,
            "right": (side // 2) * side + side - 1,
        }

        def grid_points(n: int):
            if n <= 1:
                return [side // 2]
            return [round(i * (side - 1) / (n - 1)) for i in range(n)]

        idxs = []
        for name in self.anchor_points:
            if name.startswith("grid") and "x" in name:
                grid_name = name[len("grid"):]
                parts = grid_name.split("x", 1)
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    gy = int(parts[0])
                    gx = int(parts[1])
                    ys = grid_points(gy)
                    xs = grid_points(gx)
                    for y in ys:
                        for x in xs:
                            idxs.append(y * side + x + offset)
                    continue
            si = point_to_idx.get(name)
            if si is None:
                continue
            idxs.append(si + offset)
        if offset == 1:
            idxs.append(0)
        idxs = sorted(set([i for i in idxs if 0 <= i < num_tokens]))
        return torch.tensor(idxs, device=device, dtype=torch.long)

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            raise RuntimeError("bishemethod requires bipartite_soft_matching_merge to be available")

        metric = None
        if model is not None and images is not None:
            try:
                if isinstance(images, list) and len(images) > 0:
                    images = images[0]
                if isinstance(images, torch.Tensor) and images.dim() == 3:
                    images = images.unsqueeze(0)

                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    vision_tower = getattr(model, "vision_tower", None)
                    inner = getattr(vision_tower, "vision_tower", vision_tower)
                    vision_model = getattr(inner, "vision_model", None)
                    if vision_model is not None:
                        outputs = {}

                        def hook_k(module, input, output):
                            outputs["desired_k"] = output

                        layers = vision_model.encoder.layers
                        hook_handle = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                        _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                        hook_handle.remove()
                        metric = outputs.get("desired_k", None)
            except Exception:
                metric = None

        if metric is None:
            metric = image_features
        elif metric.shape[1] == image_features.shape[1] + 1:
            metric = metric[:, 1:, :]
        elif metric.shape[1] != image_features.shape[1]:
            metric = image_features

        orig_num = image_features.shape[1]
        protected_idx = self._anchor_indices(orig_num, image_features.device)

        # Only protect anchors on the first merge stage (when spatial structure is still intact).
        first = True
        for r_feat, r_metric in self.config.get("merge_steps", [(288, 288), (144, 144), (48, 48)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            merged_feat = self.bipartite_merge(
                metric=metric,
                r=r_feat,
                x=image_features,
                protected_idx=protected_idx if first else None,
            )
            if merged_feat is None:
                break
            merged_metric = self.bipartite_merge(
                metric=metric,
                r=r_metric,
                x=metric,
                protected_idx=protected_idx if first else None,
            )
            image_features = merged_feat
            metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features
            first = False

        if image_features.shape[1] != orig_num:
            print(f"BisheMethod merged tokens: {orig_num} -> {image_features.shape[1]}")

        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            # Use a schedule that lands exactly in two digits while leaving some unmerged tokens
            # in the first stage so protected anchors can survive.
            # 576 -> 304 (r=272), 304 -> 156 (r=148), 156 -> 96 (r=60)
            "merge_steps": [(272, 272), (148, 148), (60, 60)],
            # Lighter anchors by default (less interference, better accuracy).
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
        }


@register_strategy("bishemethod_soft")
class BisheMethodSoftStrategy(BisheMethodStrategy):
    """BisheMethod-Soft: soft anchor protection on top of AIM-style multi-stage merge.

    与 `bishemethod` 的区别：
    - 不再硬性禁止 anchor 参与第一段 merge
    - 只是在第一段里给 anchor 对应 source nodes 一个轻微 penalty，
      让它们“更不容易”被 merge，从而兼顾空间保真与匹配自由度
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protected_penalty = float(config.get("protected_penalty", 0.08))

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            raise RuntimeError("bishemethod_soft requires bipartite_soft_matching_merge to be available")

        metric = None
        if model is not None and images is not None:
            try:
                if isinstance(images, list) and len(images) > 0:
                    images = images[0]
                if isinstance(images, torch.Tensor) and images.dim() == 3:
                    images = images.unsqueeze(0)

                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    vision_tower = getattr(model, "vision_tower", None)
                    inner = getattr(vision_tower, "vision_tower", vision_tower)
                    vision_model = getattr(inner, "vision_model", None)
                    if vision_model is not None:
                        outputs = {}

                        def hook_k(module, input, output):
                            outputs["desired_k"] = output

                        layers = vision_model.encoder.layers
                        hook_handle = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                        _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                        hook_handle.remove()
                        metric = outputs.get("desired_k", None)
            except Exception:
                metric = None

        if metric is None:
            metric = image_features
        elif metric.shape[1] == image_features.shape[1] + 1:
            metric = metric[:, 1:, :]
        elif metric.shape[1] != image_features.shape[1]:
            metric = image_features

        orig_num = image_features.shape[1]
        protected_idx = self._anchor_indices(orig_num, image_features.device)

        first = True
        for r_feat, r_metric in self.config.get("merge_steps", [(272, 272), (148, 148), (60, 60)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            kwargs_merge = {}
            if first and protected_idx.numel() > 0:
                kwargs_merge = {
                    "protected_idx": protected_idx,
                    "protected_penalty": self.protected_penalty,
                }

            merged_feat = self.bipartite_merge(metric=metric, r=r_feat, x=image_features, **kwargs_merge)
            if merged_feat is None:
                break
            merged_metric = self.bipartite_merge(metric=metric, r=r_metric, x=metric, **kwargs_merge)
            image_features = merged_feat
            metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features
            first = False

        if image_features.shape[1] != orig_num:
            print(
                f"BisheMethodSoft merged tokens: {orig_num} -> {image_features.shape[1]} "
                f"(penalty={self.protected_penalty})"
            )
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "merge_steps": [(272, 272), (148, 148), (60, 60)],  # 576 -> 96
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
            "protected_penalty": 0.08,
        }


@register_strategy("bishemethod_v2")
class BisheMethodV2Strategy(BisheMethodSoftStrategy):
    """PACT-inspired bishemethod.

    相比 bishemethod_soft：
    - 使用 q_proj + k_proj 的混合 metric，而不是只用 k
    - 在 metric 中注入轻量 2D position code，使 merge 更偏向“内容相似 + 位置相近”
    - 保留第一阶段 soft anchor protection
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.qk_mix = float(config.get("qk_mix", 0.35))      # metric = (1-qk_mix)*k + qk_mix*q
        self.pos_mix = float(config.get("pos_mix", 0.08))    # append 2D coords scaled by pos_mix

    def _build_pos_code(self, num_tokens: int, device, dtype) -> Optional[torch.Tensor]:
        # patch-only 576=24x24 or CLS+patch 577
        side = int(num_tokens ** 0.5)
        offset = 0
        if side * side != num_tokens:
            num_spatial = num_tokens - 1
            side = int(num_spatial ** 0.5)
            if side * side != num_spatial:
                return None
            offset = 1
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [T, 2]
        if offset == 1:
            cls = torch.zeros(1, 2, device=device, dtype=dtype)
            coords = torch.cat([cls, coords], dim=0)
        return coords[:num_tokens]

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            raise RuntimeError("bishemethod_v2 requires bipartite_soft_matching_merge to be available")

        metric = None
        if model is not None and images is not None:
            try:
                if isinstance(images, list) and len(images) > 0:
                    images = images[0]
                if isinstance(images, torch.Tensor) and images.dim() == 3:
                    images = images.unsqueeze(0)

                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    vision_tower = getattr(model, "vision_tower", None)
                    inner = getattr(vision_tower, "vision_tower", vision_tower)
                    vision_model = getattr(inner, "vision_model", None)
                    if vision_model is not None:
                        outputs = {}

                        def hook_k(module, input, output):
                            outputs["desired_k"] = output

                        def hook_q(module, input, output):
                            outputs["desired_q"] = output

                        layers = vision_model.encoder.layers
                        hk = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                        hq = layers[-2].self_attn.q_proj.register_forward_hook(hook_q)
                        _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                        hk.remove()
                        hq.remove()

                        k = outputs.get("desired_k", None)
                        q = outputs.get("desired_q", None)
                        if k is not None and q is not None and k.shape == q.shape:
                            metric = (1.0 - self.qk_mix) * k + self.qk_mix * q
                        else:
                            metric = k if k is not None else q
            except Exception:
                metric = None

        if metric is None:
            metric = image_features
        elif metric.shape[1] == image_features.shape[1] + 1:
            metric = metric[:, 1:, :]
        elif metric.shape[1] != image_features.shape[1]:
            metric = image_features

        # PACT-inspired position-aware metric augmentation.
        pos_code = self._build_pos_code(metric.shape[1], metric.device, metric.dtype)
        if pos_code is not None:
            pos_code = pos_code.unsqueeze(0).expand(metric.shape[0], -1, -1)
            metric = torch.cat([metric, self.pos_mix * pos_code], dim=-1)

        orig_num = image_features.shape[1]
        protected_idx = self._anchor_indices(orig_num, image_features.device)

        first = True
        for r_feat, r_metric in self.config.get("merge_steps", [(272, 272), (148, 148), (60, 60)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            kwargs_merge = {}
            if first and protected_idx.numel() > 0:
                kwargs_merge = {
                    "protected_idx": protected_idx,
                    "protected_penalty": self.protected_penalty,
                }

            merged_feat = self.bipartite_merge(metric=metric, r=r_feat, x=image_features, **kwargs_merge)
            if merged_feat is None:
                break
            merged_metric = self.bipartite_merge(metric=metric, r=r_metric, x=metric, **kwargs_merge)
            image_features = merged_feat
            metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features
            first = False

        if image_features.shape[1] != orig_num:
            print(
                f"BisheMethodV2 merged tokens: {orig_num} -> {image_features.shape[1]} "
                f"(qk_mix={self.qk_mix}, pos_mix={self.pos_mix}, penalty={self.protected_penalty})"
            )
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "merge_steps": [(272, 272), (148, 148), (60, 60)],  # 576 -> 96
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
            "protected_penalty": 0.06,
            "qk_mix": 0.35,
            "pos_mix": 0.08,
        }


@register_strategy("bishemethod_v2qk")
class BisheMethodV2QKStrategy(BisheMethodV2Strategy):
    """Conservative v2: keep soft-v1 backbone, add only light q/k mixing.

    设计动机：
    - v2 掉分的主要嫌疑是 position code 太强；
    - 这个版本完全去掉 position mixing，只保留较弱的 q/k 混合，
      看 query-aware metric 是否能在不伤原 soft 版分数的前提下带来增益。
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "merge_steps": [(272, 272), (148, 148), (60, 60)],  # 576 -> 96
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
            "protected_penalty": 0.08,  # keep same as soft baseline
            "qk_mix": 0.15,
            "pos_mix": 0.0,
        }


@register_strategy("bishemethod_v2stage")
class BisheMethodV2StageStrategy(BisheMethodV2Strategy):
    """Stage-aware v2: only use q/k+position on the first merge stage.

    设计动机：
    - 第一阶段保留较强的结构/语义感知；
    - 后两阶段退回更接近 AIM/soft 的原始 metric，避免低 token 区域过拟合位置先验。
    """

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            raise RuntimeError("bishemethod_v2stage requires bipartite_soft_matching_merge to be available")

        metric_base = None
        if model is not None and images is not None:
            try:
                if isinstance(images, list) and len(images) > 0:
                    images = images[0]
                if isinstance(images, torch.Tensor) and images.dim() == 3:
                    images = images.unsqueeze(0)

                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    vision_tower = getattr(model, "vision_tower", None)
                    inner = getattr(vision_tower, "vision_tower", vision_tower)
                    vision_model = getattr(inner, "vision_model", None)
                    if vision_model is not None:
                        outputs = {}

                        def hook_k(module, input, output):
                            outputs["desired_k"] = output

                        def hook_q(module, input, output):
                            outputs["desired_q"] = output

                        layers = vision_model.encoder.layers
                        hk = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                        hq = layers[-2].self_attn.q_proj.register_forward_hook(hook_q)
                        _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                        hk.remove()
                        hq.remove()

                        k = outputs.get("desired_k", None)
                        q = outputs.get("desired_q", None)
                        if k is not None and q is not None and k.shape == q.shape:
                            metric_base = (1.0 - self.qk_mix) * k + self.qk_mix * q
                        else:
                            metric_base = k if k is not None else q
            except Exception:
                metric_base = None

        if metric_base is None:
            metric_base = image_features
        elif metric_base.shape[1] == image_features.shape[1] + 1:
            metric_base = metric_base[:, 1:, :]
        elif metric_base.shape[1] != image_features.shape[1]:
            metric_base = image_features

        pos_code = self._build_pos_code(metric_base.shape[1], metric_base.device, metric_base.dtype)
        metric_first = metric_base
        if pos_code is not None and self.pos_mix > 0:
            pos_code = pos_code.unsqueeze(0).expand(metric_base.shape[0], -1, -1)
            metric_first = torch.cat([metric_base, self.pos_mix * pos_code], dim=-1)

        orig_num = image_features.shape[1]
        protected_idx = self._anchor_indices(orig_num, image_features.device)

        first = True
        metric = metric_first
        for r_feat, r_metric in self.config.get("merge_steps", [(272, 272), (148, 148), (60, 60)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            kwargs_merge = {}
            if first and protected_idx.numel() > 0:
                kwargs_merge = {
                    "protected_idx": protected_idx,
                    "protected_penalty": self.protected_penalty,
                }

            merged_feat = self.bipartite_merge(metric=metric, r=r_feat, x=image_features, **kwargs_merge)
            if merged_feat is None:
                break
            merged_metric = self.bipartite_merge(metric=metric, r=r_metric, x=metric, **kwargs_merge)
            image_features = merged_feat

            # After the first stage, fall back to pure merged feature metric.
            if first:
                metric = image_features
            else:
                metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features
            first = False

        if image_features.shape[1] != orig_num:
            print(
                f"BisheMethodV2Stage merged tokens: {orig_num} -> {image_features.shape[1]} "
                f"(qk_mix={self.qk_mix}, pos_mix={self.pos_mix}, penalty={self.protected_penalty})"
            )
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "merge_steps": [(272, 272), (148, 148), (60, 60)],  # 576 -> 96
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
            "protected_penalty": 0.07,
            "qk_mix": 0.20,
            "pos_mix": 0.04,
        }


class _AnchorAwareOrderMixin:
    """Build anchor-aware token orders before bipartite split (a=even, b=odd).

    Why:
    - Current merge uses fixed even/odd split for source/target.
    - If anchor tokens collapse to one parity side, protection can become asymmetric.
    - This mixin injects a deterministic reorder step so anchors are distributed
      across both source and target groups before each merge stage.
    """

    def _build_anchor_aware_order(
        self,
        num_tokens: int,
        device: torch.device,
        mode: str = "alternate",
        group_size: int = 1,
    ) -> Optional[torch.Tensor]:
        if num_tokens <= 1:
            return None

        anchor_idx = self._anchor_indices(num_tokens, device)
        if anchor_idx is None or anchor_idx.numel() == 0:
            return None

        all_idx = torch.arange(num_tokens, device=device, dtype=torch.long)
        is_anchor = torch.zeros(num_tokens, device=device, dtype=torch.bool)
        is_anchor[anchor_idx.clamp(min=0, max=num_tokens - 1)] = True

        anchors = all_idx[is_anchor]
        non_anchors = all_idx[~is_anchor]

        if mode == "alternate":
            # Interleave anchor and non-anchor to avoid parity collapse.
            out = []
            ia = 0
            inon = 0
            while ia < anchors.numel() or inon < non_anchors.numel():
                if ia < anchors.numel():
                    out.append(anchors[ia])
                    ia += 1
                if inon < non_anchors.numel():
                    out.append(non_anchors[inon])
                    inon += 1
            return torch.stack(out) if len(out) > 0 else None

        if mode == "balanced_pairs":
            # Fill (even, odd) pairs so each pair tends to contain one anchor.
            out = []
            ia = 0
            inon = 0
            while ia < anchors.numel() or inon < non_anchors.numel():
                if ia < anchors.numel():
                    out.append(anchors[ia]); ia += 1
                elif inon < non_anchors.numel():
                    out.append(non_anchors[inon]); inon += 1

                if inon < non_anchors.numel():
                    out.append(non_anchors[inon]); inon += 1
                elif ia < anchors.numel():
                    out.append(anchors[ia]); ia += 1
            return torch.stack(out) if len(out) > 0 else None

        if mode == "block":
            g = max(1, int(group_size))
            out = []
            ia = 0
            inon = 0
            while ia < anchors.numel() or inon < non_anchors.numel():
                for _ in range(g):
                    if ia < anchors.numel():
                        out.append(anchors[ia]); ia += 1
                for _ in range(g):
                    if inon < non_anchors.numel():
                        out.append(non_anchors[inon]); inon += 1
            return torch.stack(out) if len(out) > 0 else None

        return None


class _AnchorAwareV2StageBase(_AnchorAwareOrderMixin, BisheMethodV2StageStrategy):
    """Anchor-aware base: same v2stage-litefirst pipeline with custom pre-merge ordering."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.anchor_order_mode = config.get("anchor_order_mode", "alternate")
        self.anchor_order_group_size = int(config.get("anchor_order_group_size", 1))
        self.anchor_order_first_stage_only = bool(config.get("anchor_order_first_stage_only", True))

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                # Match the existing litefirst backbone.
                "merge_steps": [(192, 192), (144, 144), (96, 96), (48, 48)],
                "anchor_points": ["grid4x4"],
                "protected_penalty": 0.09,
                "qk_mix": 0.20,
                "pos_mix": 0.05,
                "target_total_tokens": 640,
                "anchor_order_mode": "alternate",
                "anchor_order_group_size": 1,
                "anchor_order_first_stage_only": True,
            }
        )
        return cfg

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            raise RuntimeError("anchor-aware bishemethod requires bipartite_soft_matching_merge")

        # --- Copy v2stage metric build path ---
        metric_base = None
        if model is not None and images is not None:
            try:
                if isinstance(images, list) and len(images) > 0:
                    images = images[0]
                if isinstance(images, torch.Tensor) and images.dim() == 3:
                    images = images.unsqueeze(0)
                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    vision_tower = getattr(model, "vision_tower", None)
                    inner = getattr(vision_tower, "vision_tower", vision_tower)
                    vision_model = getattr(inner, "vision_model", None)
                    if vision_model is not None:
                        outputs = {}

                        def hook_k(module, input, output):
                            outputs["desired_k"] = output

                        def hook_q(module, input, output):
                            outputs["desired_q"] = output

                        layers = vision_model.encoder.layers
                        hk = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                        hq = layers[-2].self_attn.q_proj.register_forward_hook(hook_q)
                        _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                        hk.remove()
                        hq.remove()
                        k = outputs.get("desired_k", None)
                        q = outputs.get("desired_q", None)
                        if k is not None and q is not None and k.shape == q.shape:
                            metric_base = (1.0 - self.qk_mix) * k + self.qk_mix * q
                        else:
                            metric_base = k if k is not None else q
            except Exception:
                metric_base = None

        if metric_base is None:
            metric_base = image_features
        elif metric_base.shape[1] == image_features.shape[1] + 1:
            metric_base = metric_base[:, 1:, :]
        elif metric_base.shape[1] != image_features.shape[1]:
            metric_base = image_features

        pos_code = self._build_pos_code(metric_base.shape[1], metric_base.device, metric_base.dtype)
        metric_first = metric_base
        if pos_code is not None and self.pos_mix > 0:
            pos_code = pos_code.unsqueeze(0).expand(metric_base.shape[0], -1, -1)
            metric_first = torch.cat([metric_base, self.pos_mix * pos_code], dim=-1)

        orig_num = image_features.shape[1]
        protected_idx = self._anchor_indices(orig_num, image_features.device)

        first = True
        metric = metric_first
        for r_feat, r_metric in self.config.get("merge_steps", [(192, 192), (144, 144), (96, 96), (48, 48)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            kwargs_merge = {}
            if first and protected_idx.numel() > 0:
                kwargs_merge = {
                    "protected_idx": protected_idx,
                    "protected_penalty": self.protected_penalty,
                }

            token_order = None
            if (first or (not self.anchor_order_first_stage_only)) and protected_idx.numel() > 0:
                token_order = self._build_anchor_aware_order(
                    num_tokens=image_features.shape[1],
                    device=image_features.device,
                    mode=self.anchor_order_mode,
                    group_size=self.anchor_order_group_size,
                )

            merged_feat = self.bipartite_merge(
                metric=metric,
                r=r_feat,
                x=image_features,
                token_order=token_order,
                **kwargs_merge,
            )
            if merged_feat is None:
                break
            merged_metric = self.bipartite_merge(
                metric=metric,
                r=r_metric,
                x=metric,
                token_order=token_order,
                **kwargs_merge,
            )
            image_features = merged_feat
            if first:
                metric = image_features
            else:
                metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features
            first = False

        if image_features.shape[1] != orig_num:
            print(
                f"{self.__class__.__name__} merged tokens: {orig_num} -> {image_features.shape[1]} "
                f"(order_mode={self.anchor_order_mode}, penalty={self.protected_penalty})"
            )
        return image_features


@register_strategy("bishemethod_v2stage_anchor16_litefirst")
class BisheMethodV2StageAnchor16LiteFirstStrategy(BisheMethodV2StageStrategy):
    """Ours variant with denser spatial anchors and a gentler first merge stage.

    Design:
    - 16 anchors from a 4x4 spatial grid, so the first stage keeps broader coverage.
    - Slightly stronger position prior than the default v2stage.
    - First stage merges fewer tokens; later stages finish the compression to 96.
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                # 576 -> 384 -> 240 -> 144 -> 96
                "merge_steps": [(192, 192), (144, 144), (96, 96), (48, 48)],
                "anchor_points": ["grid4x4"],
                "protected_penalty": 0.09,
                "qk_mix": 0.20,
                "pos_mix": 0.05,
            }
        )
        return cfg


class _LlavaNextPerViewBudgetMixin:
    """Per-view budgeted pruning for LLaVA-NeXT anyres inputs.

    LLaVA-NeXT anyres first expands one image into multiple square views
    (global resized view + local crops). For these inputs we want to keep
    anchor coverage inside each view instead of pruning after the views have
    already been spatially mixed together.
    """

    def _dynamic_merge_steps(self, num_tokens: int, target_keep: int):
        steps = []
        cur = int(num_tokens)
        target_keep = max(1, min(int(target_keep), cur))
        while cur > target_keep:
            # bipartite merge can remove at most half of the current tokens.
            r = min(cur // 2, cur - target_keep)
            if r <= 0:
                break
            steps.append((r, r))
            cur -= r
        return steps

    def apply_anyres_views_after_projector(self, image_features: torch.Tensor):
        if not isinstance(image_features, torch.Tensor) or image_features.dim() != 3:
            return [image_features]

        num_views, num_tokens, _ = image_features.shape
        target_total_tokens = int(self.config.get("target_total_tokens", num_views * 128))
        base_keep = max(1, target_total_tokens // max(1, num_views))
        remainder = max(0, target_total_tokens - base_keep * num_views)
        outputs = []

        original_merge_steps = self.config.get("merge_steps")
        try:
            for view_idx in range(num_views):
                keep_this_view = base_keep + (1 if view_idx < remainder else 0)
                keep_this_view = min(keep_this_view, num_tokens)
                self.config["merge_steps"] = self._dynamic_merge_steps(num_tokens, keep_this_view)
                view_out = super().apply(None, None, image_features[view_idx : view_idx + 1])
                if isinstance(view_out, torch.Tensor) and view_out.dim() == 3 and view_out.shape[0] == 1:
                    view_out = view_out.squeeze(0)
                outputs.append(view_out)
        finally:
            self.config["merge_steps"] = original_merge_steps

        kept_total = sum(x.shape[0] for x in outputs if isinstance(x, torch.Tensor) and x.dim() == 2)
        print(
            f"{self.__class__.__name__}: per-view anyres pruning "
            f"{num_views}x{num_tokens} -> total {kept_total} tokens"
        )
        return outputs


@register_strategy("bishemethod_v2stage_anchor16_litefirst_next_t640")
class BisheMethodV2StageAnchor16LiteFirstNext640Strategy(
    _LlavaNextPerViewBudgetMixin, BisheMethodV2StageAnchor16LiteFirstStrategy
):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["target_total_tokens"] = 640
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_next_t320")
class BisheMethodV2StageAnchor16LiteFirstNext320Strategy(
    _LlavaNextPerViewBudgetMixin, BisheMethodV2StageAnchor16LiteFirstStrategy
):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["target_total_tokens"] = 320
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_next_t160")
class BisheMethodV2StageAnchor16LiteFirstNext160Strategy(
    _LlavaNextPerViewBudgetMixin, BisheMethodV2StageAnchor16LiteFirstStrategy
):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["target_total_tokens"] = 160
        return cfg


# =========================
# Anchor-Aware Variants (10)
# =========================

@register_strategy("bishemethod_v2stage_anchor16_aware_v1")
class BisheMethodV2StageAnchor16AwareV1Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Alternate anchor/non-anchor to avoid parity collapse (stage-1 only)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "alternate",
                "anchor_order_first_stage_only": True,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v2")
class BisheMethodV2StageAnchor16AwareV2Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Alternate ordering applied on every stage (strongest parity control)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "alternate",
                "anchor_order_first_stage_only": False,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v3")
class BisheMethodV2StageAnchor16AwareV3Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Balanced pairs: try to put anchors into even slots, non-anchors into odd slots."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "balanced_pairs",
                "anchor_order_first_stage_only": True,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v4")
class BisheMethodV2StageAnchor16AwareV4Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Balanced pairs + light protection (softer anchor bias)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "balanced_pairs",
                "anchor_order_first_stage_only": True,
                "protected_penalty": 0.05,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v5")
class BisheMethodV2StageAnchor16AwareV5Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Block ordering g=2 (two anchors, two non-anchors)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "block",
                "anchor_order_group_size": 2,
                "anchor_order_first_stage_only": True,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v6")
class BisheMethodV2StageAnchor16AwareV6Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Block ordering g=3 (stronger spatial grouping)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "block",
                "anchor_order_group_size": 3,
                "anchor_order_first_stage_only": True,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v7")
class BisheMethodV2StageAnchor16AwareV7Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Alternate ordering + slightly stronger position mixing."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "alternate",
                "anchor_order_first_stage_only": True,
                "pos_mix": 0.08,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v8")
class BisheMethodV2StageAnchor16AwareV8Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Alternate ordering + lower position prior (less geometric bias)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "alternate",
                "anchor_order_first_stage_only": True,
                "pos_mix": 0.02,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v9")
class BisheMethodV2StageAnchor16AwareV9Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Alternate ordering + higher q/k mix (more query-aware)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "alternate",
                "anchor_order_first_stage_only": True,
                "qk_mix": 0.30,
                "protected_penalty": 0.09,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v10")
class BisheMethodV2StageAnchor16AwareV10Strategy(
    _LlavaNextPerViewBudgetMixin, _AnchorAwareV2StageBase
):
    """Alternate ordering + lighter penalty (more merge freedom)."""

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg.update(
            {
                "anchor_order_mode": "alternate",
                "anchor_order_first_stage_only": True,
                "protected_penalty": 0.03,
            }
        )
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v6_next_t640")
class BisheMethodV2StageAnchor16AwareV6Next640Strategy(
    BisheMethodV2StageAnchor16AwareV6Strategy
):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["target_total_tokens"] = 640
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v6_next_t320")
class BisheMethodV2StageAnchor16AwareV6Next320Strategy(
    BisheMethodV2StageAnchor16AwareV6Strategy
):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["target_total_tokens"] = 320
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_aware_v6_next_t160")
class BisheMethodV2StageAnchor16AwareV6Next160Strategy(
    BisheMethodV2StageAnchor16AwareV6Strategy
):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["target_total_tokens"] = 160
        return cfg


@register_strategy("bishemethod_v2stage_anchor9_litefirst")
class BisheMethodV2StageAnchor9LiteFirstStrategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["anchor_points"] = ["grid3x3"]
        return cfg


@register_strategy("bishemethod_v2stage_anchor36_litefirst")
class BisheMethodV2StageAnchor36LiteFirstStrategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["anchor_points"] = ["grid6x6"]
        return cfg


@register_strategy("bishemethod_v2stage_anchor49_litefirst")
class BisheMethodV2StageAnchor49LiteFirstStrategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["anchor_points"] = ["grid7x7"]
        return cfg


# =========================
# Anchor16 Base Extensions
# =========================


@register_strategy("bishemethod_v2stage_anchor16_litefirst_t288")
class BisheMethodV2StageAnchor16LiteFirst288Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(192, 192), (96, 96)]  # 576 -> 384 -> 288
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_t192")
class BisheMethodV2StageAnchor16LiteFirst192Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(192, 192), (96, 96), (96, 96)]  # 576 -> 384 -> 288 -> 192
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_t144")
class BisheMethodV2StageAnchor16LiteFirst144Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(192, 192), (96, 96), (96, 96), (48, 48)]  # 576 -> 384 -> 288 -> 192 -> 144
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_t64")
class BisheMethodV2StageAnchor16LiteFirst64Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(192, 192), (144, 144), (96, 96), (64, 64), (16, 16)]  # 576 -> 384 -> 240 -> 144 -> 80 -> 64
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_t32")
class BisheMethodV2StageAnchor16LiteFirst32Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(192, 192), (144, 144), (96, 96), (64, 64), (32, 32), (16, 16)]  # 576 -> 384 -> 240 -> 144 -> 80 -> 48 -> 32
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_a3_q010")
class BisheMethodV2StageAnchor16LiteFirstA3Q010Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["qk_mix"] = 0.10
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_a3_q030")
class BisheMethodV2StageAnchor16LiteFirstA3Q030Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["qk_mix"] = 0.30
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_a3_p000")
class BisheMethodV2StageAnchor16LiteFirstA3P000Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["pos_mix"] = 0.00
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_a3_p002")
class BisheMethodV2StageAnchor16LiteFirstA3P002Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["pos_mix"] = 0.02
        return cfg


@register_strategy("bishemethod_v2stage_anchor16_litefirst_a3_p008")
class BisheMethodV2StageAnchor16LiteFirstA3P008Strategy(BisheMethodV2StageAnchor16LiteFirstStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["pos_mix"] = 0.08
        return cfg


# =========================
# Ablation 1: Metric Switch
# =========================
# A1-a: All-S1 (use the Stage-1 composite metric for all merge stages)
# A1-b: All-S2 (use the Stage-2 metric for all merge stages)
# A1-c: Ours (stage-aware) -> `bishemethod_v2stage`


@register_strategy("bishemethod_v2stage_a1_all_s1")
class BisheMethodV2StageA1AllS1Strategy(BisheMethodV2Strategy):
    """Ablation1 A1-a (All-S1).

    Use the same composite (q/k + optional pos) metric across all merge stages.
    This is equivalent to "no stage switch" under the v2 backbone.
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        # Match the default hyperparams of `bishemethod_v2stage` so ablation only changes stage switching.
        return {
            "merge_steps": [(272, 272), (148, 148), (60, 60)],  # 576 -> 96
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
            "protected_penalty": 0.07,
            "qk_mix": 0.20,
            "pos_mix": 0.04,
        }


@register_strategy("bishemethod_v2stage_a1_all_s2")
class BisheMethodV2StageA1AllS2Strategy(BisheMethodSoftStrategy):
    """Ablation1 A1-b (All-S2).

    Use the Stage-2 metric for all merge stages. In this repo's implementation,
    Stage-2 corresponds to using the merged image features as the matching metric
    (i.e. no q/k mixing and no position augmentation).
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "merge_steps": [(272, 272), (148, 148), (60, 60)],  # 576 -> 96
            "anchor_points": ["tl", "tr", "bl", "br", "center"],
            "protected_penalty": 0.07,
        }

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bipartite_merge is None:
            raise RuntimeError("bishemethod_v2stage_a1_all_s2 requires bipartite_soft_matching_merge to be available")

        # Stage-2 metric: use features directly as the matching metric.
        metric = image_features

        orig_num = image_features.shape[1]
        protected_idx = self._anchor_indices(orig_num, image_features.device)

        first = True
        for r_feat, r_metric in self.config.get("merge_steps", [(272, 272), (148, 148), (60, 60)]):
            if r_feat <= 0 or r_metric <= 0:
                continue
            if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
                break

            kwargs_merge = {}
            if first and protected_idx.numel() > 0:
                kwargs_merge = {"protected_idx": protected_idx, "protected_penalty": self.protected_penalty}

            merged_feat = self.bipartite_merge(metric=metric, r=r_feat, x=image_features, **kwargs_merge)
            if merged_feat is None:
                break

            # Keep the metric consistent with "Stage-2": always use merged features.
            image_features = merged_feat
            metric = image_features
            first = False

        return image_features


def _coerce_metric_to_image_tokens(metric: Optional[torch.Tensor], image_features: torch.Tensor) -> torch.Tensor:
    """Align a candidate metric tensor to the image feature token layout."""
    if metric is None:
        return image_features
    if metric.shape[1] == image_features.shape[1] + 1:
        metric = metric[:, 1:, :]
    if metric.shape[1] != image_features.shape[1]:
        return image_features
    return metric


def _extract_qk_tensors(
    model: nn.Module,
    images: torch.Tensor,
    image_features: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Collect q/k projections from the penultimate CLIP layer for ablation metrics."""
    q = None
    k = None
    if model is None or images is None:
        return q, k

    try:
        if isinstance(images, list) and len(images) > 0:
            images = images[0]
        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0)

        if isinstance(images, torch.Tensor) and images.dim() == 4:
            vision_tower = getattr(model, "vision_tower", None)
            inner = getattr(vision_tower, "vision_tower", vision_tower)
            vision_model = getattr(inner, "vision_model", None)
            if vision_model is not None:
                outputs = {}

                def hook_k(module, input, output):
                    outputs["desired_k"] = output

                def hook_q(module, input, output):
                    outputs["desired_q"] = output

                layers = vision_model.encoder.layers
                hk = layers[-2].self_attn.k_proj.register_forward_hook(hook_k)
                hq = layers[-2].self_attn.q_proj.register_forward_hook(hook_q)
                _ = inner(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
                hk.remove()
                hq.remove()
                k = outputs.get("desired_k", None)
                q = outputs.get("desired_q", None)
    except Exception:
        q = None
        k = None

    q = _coerce_metric_to_image_tokens(q, image_features) if q is not None else None
    k = _coerce_metric_to_image_tokens(k, image_features) if k is not None else None
    return q, k


def _scalar_metric_from_values(values: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
    """Convert a scalar per-token score into a cosine-comparable token metric."""
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True).clamp_min(1e-6)
    z = ((values - mean) / std).clamp(-10.0, 10.0)
    ones = torch.ones_like(z)
    metric = torch.cat([z, ones], dim=-1)
    return metric.to(device=image_features.device, dtype=image_features.dtype)


def _build_attention_only_metric(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    image_features: torch.Tensor,
) -> torch.Tensor:
    """Use token attention profiles as the Stage-1 metric."""
    if q is None or k is None or q.shape != k.shape:
        return image_features

    qf = torch.nan_to_num(q.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    kf = torch.nan_to_num(k.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    scale = 1.0 / max(qf.shape[-1] ** 0.5, 1.0)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    attn = torch.softmax(attn, dim=-1)
    return attn.to(device=image_features.device, dtype=image_features.dtype)


def _build_key_norm_only_metric(
    k: Optional[torch.Tensor],
    image_features: torch.Tensor,
) -> torch.Tensor:
    """Use only the per-token key norm as the Stage-1 signal."""
    if k is None:
        return image_features
    key_norm = torch.norm(k.float(), dim=-1, keepdim=True)
    return _scalar_metric_from_values(key_norm, image_features)


def _build_position_only_metric(
    strategy: "BisheMethodV2StageStrategy",
    image_features: torch.Tensor,
) -> torch.Tensor:
    """Use only the 2D spatial code as the Stage-1 signal."""
    pos_code = strategy._build_pos_code(image_features.shape[1], image_features.device, image_features.dtype)
    if pos_code is None:
        return image_features
    return pos_code.unsqueeze(0).expand(image_features.shape[0], -1, -1)


def _build_cls_similarity_metric(
    image_features: torch.Tensor,
) -> torch.Tensor:
    """Approximate CLS-similarity using a global pooled pseudo-CLS token."""
    proto = image_features.float().mean(dim=1, keepdim=True)
    proto = torch.nn.functional.normalize(proto, p=2, dim=-1)
    feats = torch.nn.functional.normalize(image_features.float(), p=2, dim=-1)
    sim = torch.matmul(feats, proto.transpose(-1, -2))
    return _scalar_metric_from_values(sim, image_features)


def _run_stage1_metric_ablation(
    strategy: "BisheMethodV2StageStrategy",
    image_features: torch.Tensor,
    stage1_metric: torch.Tensor,
    debug_name: str,
) -> torch.Tensor:
    """Run the default 3-stage merge, but only replace the Stage-1 metric."""
    if strategy.bipartite_merge is None:
        raise RuntimeError(f"{debug_name} requires bipartite_soft_matching_merge to be available")

    metric = _coerce_metric_to_image_tokens(stage1_metric, image_features)
    orig_num = image_features.shape[1]
    protected_idx = strategy._anchor_indices(orig_num, image_features.device)

    first = True
    for r_feat, r_metric in strategy.config.get("merge_steps", [(272, 272), (148, 148), (60, 60)]):
        if r_feat <= 0 or r_metric <= 0:
            continue
        if image_features.shape[1] <= 1 or metric.shape[1] <= 1:
            break

        kwargs_merge = {}
        if first and protected_idx.numel() > 0:
            kwargs_merge = {
                "protected_idx": protected_idx,
                "protected_penalty": strategy.protected_penalty,
            }

        merged_feat = strategy.bipartite_merge(metric=metric, r=r_feat, x=image_features, **kwargs_merge)
        if merged_feat is None:
            break

        merged_metric = strategy.bipartite_merge(metric=metric, r=r_metric, x=metric, **kwargs_merge)
        image_features = merged_feat
        if first:
            metric = image_features
        else:
            metric = merged_metric if merged_metric is not None and merged_metric.shape[1] == image_features.shape[1] else image_features
        first = False

    if image_features.shape[1] != orig_num:
        print(f"{debug_name} merged tokens: {orig_num} -> {image_features.shape[1]}")
    return image_features


# =========================
# Ablation 2: Soft Anchor Lambda
# =========================


@register_strategy("bishemethod_v2stage_a2_lambda_000")
class BisheMethodV2StageA2Lambda000Strategy(BisheMethodV2StageStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["protected_penalty"] = 0.00
        return cfg


@register_strategy("bishemethod_v2stage_a2_lambda_003")
class BisheMethodV2StageA2Lambda003Strategy(BisheMethodV2StageStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["protected_penalty"] = 0.03
        return cfg


@register_strategy("bishemethod_v2stage_a2_lambda_005")
class BisheMethodV2StageA2Lambda005Strategy(BisheMethodV2StageStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["protected_penalty"] = 0.05
        return cfg


@register_strategy("bishemethod_v2stage_a2_lambda_010")
class BisheMethodV2StageA2Lambda010Strategy(BisheMethodV2StageStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["protected_penalty"] = 0.10
        return cfg


@register_strategy("bishemethod_v2stage_a2_lambda_015")
class BisheMethodV2StageA2Lambda015Strategy(BisheMethodV2StageStrategy):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["protected_penalty"] = 0.15
        return cfg


# =========================
# Ablation 5: Stage-1 Metric Comparisons
# =========================


@register_strategy("bishemethod_v2stage_a5_attn_only")
class BisheMethodV2StageA5AttnOnlyStrategy(BisheMethodV2StageStrategy):
    """A5-a: Stage-1 uses only attention profiles; later stages stay unchanged."""

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        q, k = _extract_qk_tensors(model, images, image_features)
        stage1_metric = _build_attention_only_metric(q, k, image_features)
        return _run_stage1_metric_ablation(self, image_features, stage1_metric, "BisheMethodV2StageA5AttnOnly")


@register_strategy("bishemethod_v2stage_a5_keynorm_only")
class BisheMethodV2StageA5KeyNormOnlyStrategy(BisheMethodV2StageStrategy):
    """A5-b: Stage-1 uses only key norm."""

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        _, k = _extract_qk_tensors(model, images, image_features)
        stage1_metric = _build_key_norm_only_metric(k, image_features)
        return _run_stage1_metric_ablation(self, image_features, stage1_metric, "BisheMethodV2StageA5KeyNormOnly")


@register_strategy("bishemethod_v2stage_a5_position_only")
class BisheMethodV2StageA5PositionOnlyStrategy(BisheMethodV2StageStrategy):
    """A5-c: Stage-1 uses only spatial position code."""

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        stage1_metric = _build_position_only_metric(self, image_features)
        return _run_stage1_metric_ablation(self, image_features, stage1_metric, "BisheMethodV2StageA5PositionOnly")


@register_strategy("bishemethod_v2stage_a5_clssim")
class BisheMethodV2StageA5CLSSimStrategy(BisheMethodV2StageStrategy):
    """A5-d: Stage-1 uses pseudo-CLS similarity."""

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        stage1_metric = _build_cls_similarity_metric(image_features)
        return _run_stage1_metric_ablation(self, image_features, stage1_metric, "BisheMethodV2StageA5CLSSim")


@register_strategy("bishemethod_v2stage_192")
class BisheMethodV2Stage192Strategy(BisheMethodV2StageStrategy):
    """BisheMethod v2stage budget=192 tokens.

    Schedule: 576 -> 304 -> 192
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(272, 272), (112, 112)]
        return cfg


@register_strategy("bishemethod_v2stage_144")
class BisheMethodV2Stage144Strategy(BisheMethodV2StageStrategy):
    """BisheMethod v2stage budget=144 tokens.

    Schedule: 576 -> 304 -> 152 -> 144
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(272, 272), (152, 152), (8, 8)]
        return cfg


@register_strategy("bishemethod_v2stage_64")
class BisheMethodV2Stage64Strategy(BisheMethodV2StageStrategy):
    """BisheMethod v2stage budget=64 tokens.

    Schedule: 576 -> 304 -> 152 -> 76 -> 64
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(272, 272), (152, 152), (76, 76), (12, 12)]
        return cfg


@register_strategy("bishemethod_v2stage_32")
class BisheMethodV2Stage32Strategy(BisheMethodV2StageStrategy):
    """BisheMethod v2stage budget=32 tokens.

    Schedule: 576 -> 304 -> 152 -> 76 -> 38 -> 32
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg["merge_steps"] = [(272, 272), (152, 152), (76, 76), (38, 38), (6, 6)]
        return cfg


@register_strategy("prumerge_advanced")
class PruMergeAdvancedStrategy(TokenPruneStrategy):
    """LLaVA-PruMerge Advanced 策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.if_adaptive = config.get("if_adaptive", True)
        self.reduction_ratio = config.get("reduction_ratio", 1/8)

    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        # 检查模型是否有 PruMerge 方法
        if hasattr(model, 'token_prune_merge_advanced'):
            return model.token_prune_merge_advanced(
                images, 
                if_adaptive=self.if_adaptive, 
                reduction_ratio=self.reduction_ratio
            )
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "if_adaptive": True,
            "reduction_ratio": 1/8
        }


@register_strategy("prumerge_plus")
class PruMergePlusStrategy(TokenPruneStrategy):
    """LLaVA-PruMerge Advanced Plus 策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.if_adaptive = config.get("if_adaptive", True)
        self.reduction_ratio = config.get("reduction_ratio", 1/8)

    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        # 检查模型是否有 PruMerge Plus 方法
        if hasattr(model, 'token_prune_merge_advanced_plus'):
            return model.token_prune_merge_advanced_plus(
                images, 
                if_adaptive=self.if_adaptive, 
                reduction_ratio=self.reduction_ratio
            )
        return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "if_adaptive": True,
            "reduction_ratio": 1/8
        }


@register_strategy("visionzip")
class VisionZipStrategy(TokenPruneStrategy):
    """VisionZip 策略 - Dominant + Contextual tokens选择
    
    为 SigLIP 视觉编码器重新实现的简化版本
    在视觉特征输出后、mm_projector 之前应用
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dominant = config.get("dominant", 54)
        self.contextual = config.get("contextual", 10)

    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        应用 VisionZip 策略到 SigLIP 输出的视觉特征
        
        Args:
            model: 视觉模型 (CLIPVisionTower)
            images: 输入图像张量
            image_features: 原始图像特征 [batch_size, num_tokens, hidden_dim]
            
        Returns:
            剪枝后的图像特征
        """
        return self._apply_visionzip(image_features)
    
    def apply_after_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        在 mm_projector 之后应用 VisionZip 策略
        
        Args:
            image_features: 经过 mm_projector 的图像特征 [batch_size, num_tokens, hidden_dim]
            
        Returns:
            剪枝后的图像特征
        """
        return self._apply_visionzip(image_features)
    
    def _apply_visionzip(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        内部方法：应用 VisionZip 剪枝逻辑
        
        Args:
            image_features: 图像特征 [batch_size, num_tokens, hidden_dim]
            
        Returns:
            剪枝后的图像特征
        """
        try:
            batch_size, num_tokens, hidden_dim = image_features.shape
            
            # 使用特征的范数作为重要性指标（替代 attention weights）
            token_importance = torch.norm(image_features, dim=-1)  # [batch_size, num_tokens]
            
            # Step 1: 选择 Dominant tokens
            # 选择 top-k dominant tokens
            num_dominant = min(self.dominant, num_tokens)
            _, topk_indices = torch.topk(token_importance, num_dominant, dim=1, largest=True)
            
            # 创建 mask 选择 dominant tokens
            dominant_mask = torch.zeros_like(token_importance, dtype=torch.bool)
            dominant_mask.scatter_(1, topk_indices, True)
            
            # 提取 dominant tokens
            dominant_tokens = image_features.masked_select(
                dominant_mask.unsqueeze(-1)
            ).view(batch_size, num_dominant, hidden_dim)
            
            # Step 2: 选择和聚合 Contextual tokens
            # 获取剩余的 tokens
            contextual_mask = ~dominant_mask
            
            contextual_hidden = image_features.masked_select(
                contextual_mask.unsqueeze(-1)
            ).view(batch_size, -1, hidden_dim)  # [batch_size, num_contextual_candidates, hidden_dim]
            
            num_contextual_candidates = contextual_hidden.shape[1]
            
            if num_contextual_candidates > 0 and self.contextual > 0:
                # 归一化用于计算相似度
                dominant_normalized = torch.nn.functional.normalize(dominant_tokens, p=2, dim=-1)
                contextual_normalized = torch.nn.functional.normalize(contextual_hidden, p=2, dim=-1)
                
                # 计算相似度矩阵
                similarity = torch.bmm(contextual_normalized, dominant_normalized.transpose(1, 2))  # [batch_size, num_contextual_candidates, num_dominant]
                
                # 为每个 contextual token 找到最相似的 dominant token
                max_similarity, best_dominant_idx = similarity.max(dim=2)  # [batch_size, num_contextual_candidates]
                
                # 选择 contextual tokens：基于相似度选择 top-k
                num_contextual = min(self.contextual, num_contextual_candidates)
                
                if num_contextual_candidates > num_contextual:
                    # 基于相似度选择 top-k contextual tokens
                    _, contextual_indices = torch.topk(max_similarity, num_contextual, dim=1, largest=True)
                    contextual_hidden = contextual_hidden.gather(
                        1, contextual_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
                    )
                
                contextual_tokens = contextual_hidden
            else:
                contextual_tokens = torch.empty(batch_size, 0, hidden_dim, device=image_features.device, dtype=image_features.dtype)
            
            # Step 3: 合并 Dominant 和 Contextual tokens
            pruned_features = torch.cat([dominant_tokens, contextual_tokens], dim=1)
            
            print(f"VisionZip: pruned from {num_tokens} to {pruned_features.shape[1]} tokens "
                  f"(dominant={self.dominant}, contextual={self.contextual})")
            
            return pruned_features
            
        except Exception as e:
            print(f"VisionZip application failed, falling back to original features: {e}")
            import traceback
            traceback.print_exc()
            return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "dominant": 54,
            "contextual": 10
        }


@register_strategy("visionzipplus")
class VisionZipPlusStrategy(TokenPruneStrategy):
    """VisionZipPlus 策略 - VisionZip 的改进版本
    
    改进:
    1. 对四个角 + 中心的 patch 加入轻微空间偏置，确保不会完全丢失关键位置
    2. 使用 softmax 加权相似度聚合 contextual tokens，而非简单的硬选择
    
    在视觉特征输出后、mm_projector 之前应用
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dominant = config.get("dominant", 54)
        self.contextual = config.get("contextual", 10)
        self.spatial_bias = config.get("spatial_bias", 0.02)
        self.similarity_temp = config.get("similarity_temp", 3.0)
        # Avoid log spam during long evals; print only first N calls.
        self._debug_print_limit = int(config.get("debug_print_limit", 3))
        self._debug_print_count = 0

    def apply(self, model: nn.Module, images: torch.Tensor, 
              image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        应用 VisionZipPlus 策略到视觉特征
        
        Args:
            model: 视觉模型 (CLIPVisionTower)
            images: 输入图像张量
            image_features: 原始图像特征 [batch_size, num_tokens, hidden_dim]
            
        Returns:
            剪枝后的图像特征
        """
        return self._apply_visionzip_plus(image_features)
    
    def apply_after_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        在 mm_projector 之后应用 VisionZipPlus 策略
        """
        return self._apply_visionzip_plus(image_features)
    
    def _get_spatial_bonus_mask(self, batch_size: int, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        为四个角和中心位置的 patch 生成空间偏置 bonus
        
        Args:
            batch_size: 批次大小
            num_tokens: token 总数 (包括 [CLS] token)
            device: 设备
            dtype: 数据类型
            
        Returns:
            空间偏置 tensor [batch_size, num_tokens]
        """
        bonus = torch.zeros(batch_size, num_tokens, device=device, dtype=dtype)

        # LLaVA often uses patch-only features (no CLS): num_tokens == side^2 (e.g. 576).
        # Some vision towers output CLS+patches: num_tokens == 1 + side^2 (e.g. 577).
        if num_tokens <= 0:
            return bonus

        has_cls = False
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            # patch-only
            num_spatial = num_tokens
            offset = 0
        else:
            # maybe CLS+patches
            num_spatial = num_tokens - 1
            side = int(num_spatial ** 0.5)
            if side * side != num_spatial or num_spatial <= 0:
                return bonus
            has_cls = True
            offset = 1

        # Four corners + center (in spatial grid)
        corner_indices_spatial = [
            0,
            side - 1,
            (side - 1) * side,
            (side - 1) * side + side - 1,
            (side // 2) * side + side // 2,
        ]

        for idx in corner_indices_spatial:
            if 0 <= idx < num_spatial:
                bonus[:, idx + offset] += self.spatial_bias

        # Optionally also keep CLS slightly (if present) by giving it tiny bias.
        if has_cls:
            bonus[:, 0] += self.spatial_bias * 0.25

        return bonus
    
    def _apply_visionzip_plus(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        内部方法：应用 VisionZipPlus 剪枝逻辑
        
        Args:
            image_features: 图像特征 [batch_size, num_tokens, hidden_dim]
            
        Returns:
            剪枝后的图像特征
        """
        try:
            batch_size, num_tokens, hidden_dim = image_features.shape
            
            # 使用特征的范数作为重要性指标
            token_importance = torch.norm(image_features, dim=-1)  # [batch_size, num_tokens]
            
            # 加上空间偏置
            spatial_bonus = self._get_spatial_bonus_mask(batch_size, num_tokens, image_features.device, token_importance.dtype)
            token_importance += spatial_bonus
            
            # Step 1: 选择 Dominant tokens
            num_dominant = min(self.dominant, num_tokens)
            _, topk_indices = torch.topk(token_importance, num_dominant, dim=1, largest=True)
            
            dominant_mask = torch.zeros_like(token_importance, dtype=torch.bool)
            dominant_mask.scatter_(1, topk_indices, True)
            
            dominant_tokens = image_features.masked_select(
                dominant_mask.unsqueeze(-1)
            ).view(batch_size, num_dominant, hidden_dim)
            
            # Step 2: 选择和聚合 Contextual tokens (使用 softmax 加权)
            contextual_mask = ~dominant_mask
            
            contextual_hidden = image_features.masked_select(
                contextual_mask.unsqueeze(-1)
            ).view(batch_size, -1, hidden_dim)
            
            num_contextual_candidates = contextual_hidden.shape[1]
            
            if num_contextual_candidates > 0 and self.contextual > 0:
                dominant_normalized = torch.nn.functional.normalize(dominant_tokens, p=2, dim=-1)
                contextual_normalized = torch.nn.functional.normalize(contextual_hidden, p=2, dim=-1)
                
                similarity = torch.bmm(contextual_normalized, dominant_normalized.transpose(1, 2))
                
                num_contextual = min(self.contextual, num_contextual_candidates)
                
                # 先选择候选 tokens，再聚合
                max_similarity = similarity.max(dim=2).values
                _, selected_contextual_indices = torch.topk(max_similarity, num_contextual, dim=1, largest=True)
                
                selected_contextual = contextual_hidden.gather(
                    1, selected_contextual_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
                )
                
                # 对剩余的 tokens，计算 softmax 加权聚合
                if num_contextual_candidates > num_contextual:
                    unselected_mask = torch.ones(num_contextual_candidates, dtype=torch.bool, device=contextual_hidden.device)
                    unselected_mask[selected_contextual_indices[0]] = False
                    unselected_hidden = contextual_hidden[:, unselected_mask]
                    
                    if unselected_hidden.shape[1] > 0:
                        unselected_normalized = torch.nn.functional.normalize(unselected_hidden, p=2, dim=-1)
                        sim_to_selected = torch.bmm(unselected_normalized, 
                                                    torch.nn.functional.normalize(selected_contextual, p=2, dim=-1).transpose(1,2))
                        
                        # softmax 加权聚合
                        softmax_weights = torch.softmax(sim_to_selected * self.similarity_temp, dim=2)
                        aggregated = torch.bmm(softmax_weights.transpose(1,2), unselected_hidden)
                        selected_contextual = selected_contextual + aggregated
                
                contextual_tokens = selected_contextual
            else:
                contextual_tokens = torch.empty(batch_size, 0, hidden_dim, device=image_features.device, dtype=image_features.dtype)
            
            pruned_features = torch.cat([dominant_tokens, contextual_tokens], dim=1)

            if self._debug_print_count < self._debug_print_limit:
                print(
                    f"VisionZipPlus: pruned from {num_tokens} to {pruned_features.shape[1]} tokens "
                    f"(dominant={self.dominant}, contextual={self.contextual})"
                )
                self._debug_print_count += 1
            
            return pruned_features
            
        except Exception as e:
            print(f"VisionZipPlus application failed, falling back to original features: {e}")
            import traceback
            traceback.print_exc()
            return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "dominant": 54,
            "contextual": 10,
            "spatial_bias": 0.02,
            "similarity_temp": 3.0,
            "debug_print_limit": 3
        }


@register_strategy("visionzipplus_adaptive")
class VisionZipPlusAdaptiveStrategy(TokenPruneStrategy):
    """VisionZipPlusAdaptive: 自适应 token budget + 空间覆盖约束的剪枝策略。

    目标：在明显提速（token reduction）的同时，把 GQA 等 VQA 任务的性能拉回到可用区间。
    核心改进点（适合写论文）：
    1) 兼容 patch-only（576）与 CLS+patch（577）两种特征形态的空间先验；
    2) 用重要性分布熵估计冗余度，自适应确定保留 token 数；
    3) Contextual 部分采用“空间分桶取 top”保证覆盖，避免 top-k 全挤在局部区域。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_keep = int(config.get("min_keep", 128))
        self.max_keep = int(config.get("max_keep", 256))
        self.dominant_ratio = float(config.get("dominant_ratio", 0.75))
        self.spatial_bias = float(config.get("spatial_bias", 0.02))
        self.entropy_temp = float(config.get("entropy_temp", 1.0))
        self.grid_bins = int(config.get("grid_bins", 6))  # for 24x24, 6 bins gives 4x4 patches per bin
        self._debug_print_limit = int(config.get("debug_print_limit", 3))
        self._debug_print_count = 0

    def apply(
        self,
        model: nn.Module,
        images: torch.Tensor,
        image_features: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Kept for API compatibility; pruning uses only features.
        return self._apply(image_features)

    def apply_after_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        return self._apply(image_features)

    def _spatial_layout(self, num_tokens: int):
        # Returns (num_spatial, offset, side) or (None, None, None) if unknown layout.
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            return num_tokens, 0, side
        num_spatial = num_tokens - 1
        if num_spatial <= 0:
            return None, None, None
        side = int(num_spatial ** 0.5)
        if side * side != num_spatial:
            return None, None, None
        return num_spatial, 1, side

    def _spatial_bonus(self, batch_size: int, num_tokens: int, device, dtype) -> torch.Tensor:
        bonus = torch.zeros(batch_size, num_tokens, device=device, dtype=dtype)
        num_spatial, offset, side = self._spatial_layout(num_tokens)
        if num_spatial is None:
            return bonus
        idxs = [
            0,
            side - 1,
            (side - 1) * side,
            (side - 1) * side + side - 1,
            (side // 2) * side + side // 2,
        ]
        for i in idxs:
            if 0 <= i < num_spatial:
                bonus[:, i + offset] += self.spatial_bias
        if offset == 1:
            bonus[:, 0] += self.spatial_bias * 0.25
        return bonus

    def _adaptive_keep(self, token_importance: torch.Tensor) -> int:
        # token_importance: [B, T]
        # Use normalized entropy as a proxy for redundancy: higher entropy => flatter => keep more.
        # Compute on batch mean to get a single keep for this image.
        # Guard against NaN/Inf in importance (mixed precision / numerical issues).
        token_importance = torch.nan_to_num(token_importance, nan=0.0, posinf=1e4, neginf=0.0)
        p = torch.softmax(token_importance / max(self.entropy_temp, 1e-6), dim=1)  # [B, T]
        eps = 1e-12
        H = -(p * (p + eps).log()).sum(dim=1)  # [B]
        # Normalize by log(T) to get [0,1] range. token_importance.shape[1] is an int.
        import math
        denom = max(math.log(float(token_importance.shape[1])), 1e-6)
        Hn = (H / denom).mean()
        Hn = torch.nan_to_num(Hn, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).item()
        keep = int(round(self.min_keep + (self.max_keep - self.min_keep) * Hn))
        return max(1, min(keep, token_importance.shape[1]))

    def _apply(self, image_features: torch.Tensor) -> torch.Tensor:
        try:
            b, t, d = image_features.shape
            imp = torch.norm(image_features, dim=-1)  # [B, T]
            imp = torch.nan_to_num(imp, nan=0.0, posinf=1e4, neginf=0.0)
            imp = imp + self._spatial_bonus(b, t, image_features.device, imp.dtype)

            keep_total = self._adaptive_keep(imp)
            keep_total = max(self.min_keep, min(self.max_keep, keep_total))
            keep_total = min(keep_total, t)

            num_dom = int(round(keep_total * self.dominant_ratio))
            num_dom = max(1, min(num_dom, keep_total))
            num_ctx = keep_total - num_dom

            # Dominant: top-k by importance
            _, dom_idx = torch.topk(imp, num_dom, dim=1, largest=True)
            dom_mask = torch.zeros_like(imp, dtype=torch.bool)
            dom_mask.scatter_(1, dom_idx, True)
            dom_tokens = image_features.masked_select(dom_mask.unsqueeze(-1)).view(b, num_dom, d)

            if num_ctx <= 0:
                out = dom_tokens
            else:
                # Remaining candidates
                rem_mask = ~dom_mask
                rem_feats = image_features.masked_select(rem_mask.unsqueeze(-1)).view(b, -1, d)
                rem_imp = imp.masked_select(rem_mask).view(b, -1)  # [B, R]
                R = rem_feats.shape[1]

                if R == 0:
                    out = dom_tokens
                else:
                    # Spatial coverage selection if we can infer grid layout.
                    num_spatial, offset, side = self._spatial_layout(t)
                    if num_spatial is None or side is None or side <= 1:
                        # Fallback: top by remaining importance
                        k = min(num_ctx, R)
                        _, ctx_pick = torch.topk(rem_imp, k, dim=1, largest=True)
                        ctx_tokens = rem_feats.gather(1, ctx_pick.unsqueeze(-1).expand(-1, -1, d))
                        out = torch.cat([dom_tokens, ctx_tokens], dim=1)
                    else:
                        # Adaptive binning: with smaller token budgets, fewer bins works better.
                        # Use sqrt(num_ctx) as a heuristic target for bins per side.
                        import math
                        bin_n = max(2, min(self.grid_bins, int(round(math.sqrt(max(1, num_ctx))))))
                        bin_size = max(1, side // bin_n)

                        orig_indices = (~dom_mask).nonzero(as_tuple=False)  # [B*R, 2] => (batch, token_idx)
                        ctx_tokens_all = []
                        for bi in range(b):
                            bi_rows = orig_indices[orig_indices[:, 0] == bi][:, 1]  # [R]
                            if bi_rows.numel() == 0:
                                ctx_tokens_all.append(rem_feats[bi : bi + 1, :0, :])
                                continue

                            spatial_idx = bi_rows - offset
                            valid = (spatial_idx >= 0) & (spatial_idx < side * side)

                            if not valid.any():
                                k = min(num_ctx, R)
                                _, ctx_pick = torch.topk(rem_imp[bi : bi + 1], k, dim=1, largest=True)
                                ctx_tokens_all.append(
                                    rem_feats[bi : bi + 1].gather(1, ctx_pick.unsqueeze(-1).expand(-1, -1, d))
                                )
                                continue

                            spatial_valid = spatial_idx[valid]
                            ys = spatial_valid // side
                            xs = spatial_valid % side
                            by = (ys // bin_size).clamp(0, bin_n - 1)
                            bx = (xs // bin_size).clamp(0, bin_n - 1)
                            bin_id = by * bin_n + bx  # [R_valid]

                            # Map "valid positions" back to rem indices
                            rem_pos_valid = valid.nonzero(as_tuple=False).squeeze(1)  # indices into rem_feats/rem_imp

                            picked = []
                            for bid in range(bin_n * bin_n):
                                m = (bin_id == bid)
                                if not m.any():
                                    continue
                                cand_pos = rem_pos_valid[m.nonzero(as_tuple=False).squeeze(1)]
                                cand_scores = rem_imp[bi, cand_pos]
                                best = cand_pos[torch.argmax(cand_scores)]
                                picked.append(int(best.item()))

                            if len(picked) == 0:
                                k = min(num_ctx, R)
                                _, ctx_pick = torch.topk(rem_imp[bi : bi + 1], k, dim=1, largest=True)
                                ctx_tokens_all.append(
                                    rem_feats[bi : bi + 1].gather(1, ctx_pick.unsqueeze(-1).expand(-1, -1, d))
                                )
                                continue

                            # Dedup and rank
                            picked = list(dict.fromkeys(picked))
                            picked_t = torch.tensor(picked, device=rem_imp.device, dtype=torch.long)
                            picked_scores = rem_imp[bi, picked_t]
                            k = min(num_ctx, picked_t.numel())
                            topk = torch.topk(picked_scores, k, largest=True).indices
                            final_idx = picked_t[topk]
                            ctx = rem_feats[bi, final_idx].unsqueeze(0)

                            # Fill if不足
                            if ctx.shape[1] < num_ctx:
                                fill = num_ctx - ctx.shape[1]
                                mask = torch.ones(R, device=rem_imp.device, dtype=torch.bool)
                                mask[final_idx] = False
                                if mask.any():
                                    avail_imp = rem_imp[bi, mask]
                                    avail_feats = rem_feats[bi, mask]
                                    fk = min(fill, avail_feats.shape[0])
                                    _, fidx = torch.topk(avail_imp, fk, largest=True)
                                    ctx = torch.cat([ctx, avail_feats[fidx].unsqueeze(0)], dim=1)

                            ctx_tokens_all.append(ctx)

                        ctx_tokens = torch.cat(ctx_tokens_all, dim=0) if b > 1 else ctx_tokens_all[0]
                        out = torch.cat([dom_tokens, ctx_tokens], dim=1)

            if self._debug_print_count < self._debug_print_limit:
                print(
                    f"VisionZipPlusAdaptive: pruned from {t} to {out.shape[1]} tokens "
                    f"(keep_total={keep_total}, min={self.min_keep}, max={self.max_keep})"
                )
                self._debug_print_count += 1

            return out
        except Exception as e:
            # Avoid flooding logs on long evals.
            if self._debug_print_count < self._debug_print_limit:
                print(f"VisionZipPlusAdaptive failed, falling back to original features: {e}")
                import traceback
                traceback.print_exc()
                self._debug_print_count += 1
            return image_features

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            # Smaller budget by default: more aggressive but still usually safe for VQA.
            "min_keep": 64,
            "max_keep": 128,
            "dominant_ratio": 0.75,
            "spatial_bias": 0.02,
            "entropy_temp": 1.0,
            "grid_bins": 6,
            "debug_print_limit": 3,
        }


@register_strategy("visionzipplus_diverse")
class VisionZipPlusDiverseStrategy(TokenPruneStrategy):
    """VisionZipPlusDiverse: 64~128 预算内的“重要性-多样性”联合选点剪枝。

    面向论文的创新点：
    - 在固定/小预算下，用贪心次模目标近似（importance + diversity）做 token 选择，
      避免简单 top-k 导致 token 空间覆盖差、信息丢失严重（这是 VQA 掉分的主因之一）。
    - 可选空间分桶先验，保证全图覆盖。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_keep = int(config.get("min_keep", 64))
        self.max_keep = int(config.get("max_keep", 128))
        self.pool_mult = int(config.get("pool_mult", 4))  # candidate pool = keep_total * pool_mult
        self.alpha = float(config.get("alpha", 0.7))      # weight for importance vs diversity
        self.spatial_bias = float(config.get("spatial_bias", 0.02))
        self.grid_bins = int(config.get("grid_bins", 6))
        self._debug_print_limit = int(config.get("debug_print_limit", 3))
        self._debug_print_count = 0

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._apply(image_features)

    def apply_after_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        return self._apply(image_features)

    def _spatial_layout(self, num_tokens: int):
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            return num_tokens, 0, side
        num_spatial = num_tokens - 1
        if num_spatial <= 0:
            return None, None, None
        side = int(num_spatial ** 0.5)
        if side * side != num_spatial:
            return None, None, None
        return num_spatial, 1, side

    def _spatial_bonus(self, batch_size: int, num_tokens: int, device, dtype) -> torch.Tensor:
        bonus = torch.zeros(batch_size, num_tokens, device=device, dtype=dtype)
        num_spatial, offset, side = self._spatial_layout(num_tokens)
        if num_spatial is None:
            return bonus
        idxs = [
            0,
            side - 1,
            (side - 1) * side,
            (side - 1) * side + side - 1,
            (side // 2) * side + side // 2,
        ]
        for i in idxs:
            if 0 <= i < num_spatial:
                bonus[:, i + offset] += self.spatial_bias
        if offset == 1:
            bonus[:, 0] += self.spatial_bias * 0.25
        return bonus

    def _pick_with_spatial_bins(self, imp: torch.Tensor, keep_total: int, side: int, offset: int) -> torch.Tensor:
        # imp: [T]
        # Return indices prioritized by per-bin top scores.
        import math
        num_ctx = keep_total
        bin_n = max(2, min(self.grid_bins, int(round(math.sqrt(max(1, num_ctx))))))
        bin_size = max(1, side // bin_n)
        spatial_idx = torch.arange(imp.shape[0], device=imp.device) - offset
        valid = (spatial_idx >= 0) & (spatial_idx < side * side)
        if not valid.any():
            return torch.topk(imp, min(keep_total, imp.shape[0]), largest=True).indices
        sv = spatial_idx[valid]
        ys = sv // side
        xs = sv % side
        by = (ys // bin_size).clamp(0, bin_n - 1)
        bx = (xs // bin_size).clamp(0, bin_n - 1)
        bid = by * bin_n + bx
        valid_pos = valid.nonzero(as_tuple=False).squeeze(1)
        picked = []
        for b in range(bin_n * bin_n):
            m = (bid == b)
            if not m.any():
                continue
            cand_pos = valid_pos[m.nonzero(as_tuple=False).squeeze(1)]
            cand_scores = imp[cand_pos]
            best = cand_pos[torch.argmax(cand_scores)]
            picked.append(int(best.item()))
        if len(picked) == 0:
            return torch.topk(imp, min(keep_total, imp.shape[0]), largest=True).indices
        picked = list(dict.fromkeys(picked))
        picked_t = torch.tensor(picked, device=imp.device, dtype=torch.long)
        # sort by score
        scores = imp[picked_t]
        topk = torch.topk(scores, min(keep_total, picked_t.numel()), largest=True).indices
        return picked_t[topk]

    def _apply(self, image_features: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: do NOT fallback silently. If pruning fails, raise and fail the eval.
        # Also: do selection math in FP32 to avoid FP16 overflow/underflow.
        b, t, d = image_features.shape
        # LLaVA eval uses batch=1 almost always; support b>1 but do per-sample loop.
        outs = []
        for bi in range(b):
            feats = image_features[bi]  # [T, D] (likely fp16/bf16)
            feats_f = feats.float()
            # Importance (FP32)
            imp = torch.norm(feats_f, dim=-1)
            imp = torch.nan_to_num(imp, nan=0.0, posinf=1e4, neginf=0.0)
            imp = imp + self._spatial_bonus(1, t, feats.device, imp.dtype).squeeze(0)

            # Choose budget within [min,max] based on distribution flatness (cheap heuristic).
            # If importance is very peaky => keep less; if flat => keep more.
            imp_std = float(torch.std(imp).item())
            imp_mean = float(torch.mean(imp).item()) + 1e-6
            cv = imp_std / imp_mean  # coefficient of variation
            # cv high => peaky => smaller keep
            keep_total = int(round(self.max_keep - (self.max_keep - self.min_keep) * min(max(cv, 0.0), 1.0)))
            keep_total = max(self.min_keep, min(self.max_keep, keep_total))
            keep_total = min(keep_total, t)

            # Candidate pool
            pool_k = min(t, keep_total * max(2, self.pool_mult))
            # Optional: get a spatially covered candidate set first, then expand with top importance.
            num_spatial, offset, side = self._spatial_layout(t)
            if num_spatial is not None and side is not None and side > 1:
                bins_idx = self._pick_with_spatial_bins(imp, min(pool_k, keep_total * 2), side, offset)
                top_idx = torch.topk(imp, pool_k, largest=True).indices
                cand_idx = torch.unique(torch.cat([bins_idx, top_idx], dim=0))[:pool_k]
            else:
                cand_idx = torch.topk(imp, pool_k, largest=True).indices

            cand = feats_f[cand_idx]  # [P, D] FP32 for stable selection
            cand_imp = imp[cand_idx]  # [P]
            cand_norm = torch.nn.functional.normalize(cand, p=2, dim=-1)

            # Greedy selection: objective = alpha * normalized_imp + (1-alpha) * diversity_gain
            # diversity_gain approximated by farthest-point (maximize min distance to selected)
            # Start from best importance
            sel = [int(torch.argmax(cand_imp).item())]
            # Track min distance to selected for each candidate
            # cosine distance = 1 - cosine_sim
            min_dist = 1.0 - torch.matmul(cand_norm, cand_norm[sel[0]].unsqueeze(-1)).squeeze(-1)
            # Normalize importance to [0,1]
            imp_min = float(torch.min(cand_imp).item())
            imp_max = float(torch.max(cand_imp).item())
            imp_norm = (cand_imp - imp_min) / max(imp_max - imp_min, 1e-6)

            while len(sel) < keep_total:
                # score for each candidate not selected
                score = self.alpha * imp_norm + (1.0 - self.alpha) * min_dist
                # Mask selected indices with a very negative FP32 value (avoid fp16 overflow).
                score[sel] = -1e9
                nxt = int(torch.argmax(score).item())
                if nxt < 0:
                    break
                sel.append(nxt)
                # update min_dist
                dist_to_new = 1.0 - torch.matmul(cand_norm, cand_norm[nxt].unsqueeze(-1)).squeeze(-1)
                min_dist = torch.minimum(min_dist, dist_to_new)

            sel_idx = cand_idx[torch.tensor(sel, device=cand_idx.device, dtype=torch.long)]
            # Return original dtype features (do not upcast the model inputs).
            out = feats[sel_idx].unsqueeze(0)  # [1, K, D]

            if self._debug_print_count < self._debug_print_limit:
                print(f"VisionZipPlusDiverse: pruned from {t} to {out.shape[1]} tokens (keep_total={keep_total})")
                self._debug_print_count += 1

            outs.append(out)

        return torch.cat(outs, dim=0)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "min_keep": 64,
            "max_keep": 128,
            "pool_mult": 4,
            "alpha": 0.7,
            "spatial_bias": 0.02,
            "grid_bins": 6,
            "debug_print_limit": 3,
        }


@register_strategy("visionzipplus_anchor")
class VisionZipPlusAnchorStrategy(TokenPruneStrategy):
    """Anchor-preserving diverse pruning.

    相比 `visionzipplus_diverse`，额外显式保留若干空间锚点 token，
    然后在剩余预算上做 importance + diversity 贪心选择。
    适合 GQA 这类需要更强全局空间覆盖的任务。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_keep = int(config.get("min_keep", 80))
        self.max_keep = int(config.get("max_keep", 128))
        self.pool_mult = int(config.get("pool_mult", 4))
        self.alpha = float(config.get("alpha", 0.65))
        self.spatial_bias = float(config.get("spatial_bias", 0.03))
        self.grid_bins = int(config.get("grid_bins", 6))
        self.anchor_points = config.get(
            "anchor_points",
            ["tl", "tr", "bl", "br", "center", "top", "bottom", "left", "right"],
        )
        self._debug_print_limit = int(config.get("debug_print_limit", 3))
        self._debug_print_count = 0

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._apply(image_features)

    def apply_after_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        return self._apply(image_features)

    def _spatial_layout(self, num_tokens: int):
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            return num_tokens, 0, side
        num_spatial = num_tokens - 1
        if num_spatial <= 0:
            return None, None, None
        side = int(num_spatial ** 0.5)
        if side * side != num_spatial:
            return None, None, None
        return num_spatial, 1, side

    def _anchor_indices(self, num_tokens: int, device) -> torch.Tensor:
        num_spatial, offset, side = self._spatial_layout(num_tokens)
        if num_spatial is None:
            return torch.empty(0, device=device, dtype=torch.long)

        point_to_idx = {
            "tl": 0,
            "tr": side - 1,
            "bl": (side - 1) * side,
            "br": (side - 1) * side + side - 1,
            "center": (side // 2) * side + side // 2,
            "top": side // 2,
            "bottom": (side - 1) * side + side // 2,
            "left": (side // 2) * side,
            "right": (side // 2) * side + side - 1,
        }
        indices = []
        for name in self.anchor_points:
            idx = point_to_idx.get(name)
            if idx is not None and 0 <= idx < num_spatial:
                indices.append(idx + offset)
        if offset == 1:
            indices.append(0)  # optionally keep CLS-like token if present
        if not indices:
            return torch.empty(0, device=device, dtype=torch.long)
        indices = sorted(set(indices))
        return torch.tensor(indices, device=device, dtype=torch.long)

    def _spatial_bonus(self, num_tokens: int, device, dtype) -> torch.Tensor:
        bonus = torch.zeros(num_tokens, device=device, dtype=dtype)
        anchor_idx = self._anchor_indices(num_tokens, device)
        if anchor_idx.numel() > 0:
            bonus[anchor_idx] += self.spatial_bias
        return bonus

    def _pick_with_spatial_bins(self, imp: torch.Tensor, keep_total: int, side: int, offset: int) -> torch.Tensor:
        import math
        bin_n = max(2, min(self.grid_bins, int(round(math.sqrt(max(1, keep_total))))))
        bin_size = max(1, side // bin_n)
        spatial_idx = torch.arange(imp.shape[0], device=imp.device) - offset
        valid = (spatial_idx >= 0) & (spatial_idx < side * side)
        if not valid.any():
            return torch.topk(imp, min(keep_total, imp.shape[0]), largest=True).indices
        sv = spatial_idx[valid]
        ys = sv // side
        xs = sv % side
        by = (ys // bin_size).clamp(0, bin_n - 1)
        bx = (xs // bin_size).clamp(0, bin_n - 1)
        bid = by * bin_n + bx
        valid_pos = valid.nonzero(as_tuple=False).squeeze(1)
        picked = []
        for b in range(bin_n * bin_n):
            m = bid == b
            if not m.any():
                continue
            cand_pos = valid_pos[m.nonzero(as_tuple=False).squeeze(1)]
            cand_scores = imp[cand_pos]
            best = cand_pos[torch.argmax(cand_scores)]
            picked.append(int(best.item()))
        if not picked:
            return torch.topk(imp, min(keep_total, imp.shape[0]), largest=True).indices
        picked_t = torch.tensor(sorted(set(picked)), device=imp.device, dtype=torch.long)
        scores = imp[picked_t]
        topk = torch.topk(scores, min(keep_total, picked_t.numel()), largest=True).indices
        return picked_t[topk]

    def _apply(self, image_features: torch.Tensor) -> torch.Tensor:
        b, t, d = image_features.shape
        outs = []
        for bi in range(b):
            feats = image_features[bi]
            feats_f = feats.float()
            imp = torch.norm(feats_f, dim=-1)
            imp = torch.nan_to_num(imp, nan=0.0, posinf=1e4, neginf=0.0)
            imp = imp + self._spatial_bonus(t, feats.device, imp.dtype)

            imp_std = float(torch.std(imp).item())
            imp_mean = float(torch.mean(imp).item()) + 1e-6
            cv = imp_std / imp_mean
            keep_total = int(round(self.max_keep - (self.max_keep - self.min_keep) * min(max(cv, 0.0), 1.0)))
            keep_total = max(self.min_keep, min(self.max_keep, keep_total))
            keep_total = min(keep_total, t)

            anchor_idx = self._anchor_indices(t, feats.device)
            if anchor_idx.numel() > keep_total:
                anchor_scores = imp[anchor_idx]
                keep_anchor = torch.topk(anchor_scores, keep_total, largest=True).indices
                anchor_idx = anchor_idx[keep_anchor]

            remaining_budget = max(0, keep_total - anchor_idx.numel())
            if remaining_budget == 0:
                sel_idx = anchor_idx
            else:
                mask = torch.ones(t, device=feats.device, dtype=torch.bool)
                if anchor_idx.numel() > 0:
                    mask[anchor_idx] = False
                rem_idx = mask.nonzero(as_tuple=False).squeeze(1)
                rem_imp_full = imp[rem_idx]

                pool_k = min(rem_idx.numel(), remaining_budget * max(2, self.pool_mult))
                num_spatial, offset, side = self._spatial_layout(t)
                if num_spatial is not None and side is not None and side > 1:
                    bins_local = self._pick_with_spatial_bins(rem_imp_full, min(pool_k, remaining_budget * 2), side, max(0, offset - int(anchor_idx.numel() > 0)))
                    top_local = torch.topk(rem_imp_full, pool_k, largest=True).indices
                    cand_local = torch.unique(torch.cat([bins_local, top_local], dim=0))[:pool_k]
                else:
                    cand_local = torch.topk(rem_imp_full, pool_k, largest=True).indices

                cand_idx = rem_idx[cand_local]
                cand = feats_f[cand_idx]
                cand_imp = imp[cand_idx]
                cand_norm = torch.nn.functional.normalize(cand, p=2, dim=-1)

                imp_min = float(torch.min(cand_imp).item())
                imp_max = float(torch.max(cand_imp).item())
                imp_norm = (cand_imp - imp_min) / max(imp_max - imp_min, 1e-6)

                if anchor_idx.numel() > 0:
                    anchor_norm = torch.nn.functional.normalize(feats_f[anchor_idx], p=2, dim=-1)
                    sim = torch.matmul(cand_norm, anchor_norm.transpose(0, 1))
                    min_dist = 1.0 - sim.max(dim=1).values
                else:
                    first = int(torch.argmax(cand_imp).item())
                    min_dist = 1.0 - torch.matmul(cand_norm, cand_norm[first].unsqueeze(-1)).squeeze(-1)

                sel_local = []
                while len(sel_local) < remaining_budget:
                    score = self.alpha * imp_norm + (1.0 - self.alpha) * min_dist
                    if sel_local:
                        score[sel_local] = -1e9
                    nxt = int(torch.argmax(score).item())
                    if nxt < 0:
                        break
                    sel_local.append(nxt)
                    dist_to_new = 1.0 - torch.matmul(cand_norm, cand_norm[nxt].unsqueeze(-1)).squeeze(-1)
                    min_dist = torch.minimum(min_dist, dist_to_new)

                diverse_idx = cand_idx[torch.tensor(sel_local, device=cand_idx.device, dtype=torch.long)]
                sel_idx = torch.cat([anchor_idx, diverse_idx], dim=0) if anchor_idx.numel() > 0 else diverse_idx

            out = feats[sel_idx].unsqueeze(0)
            if self._debug_print_count < self._debug_print_limit:
                print(
                    f"VisionZipPlusAnchor: pruned from {t} to {out.shape[1]} tokens "
                    f"(keep_total={keep_total}, anchors={anchor_idx.numel()})"
                )
                self._debug_print_count += 1
            outs.append(out)
        return torch.cat(outs, dim=0)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "min_keep": 80,
            "max_keep": 128,
            "pool_mult": 4,
            "alpha": 0.65,
            "spatial_bias": 0.03,
            "grid_bins": 6,
            "anchor_points": ["tl", "tr", "bl", "br", "center", "top", "bottom", "left", "right"],
            "debug_print_limit": 3,
        }


@register_strategy("visionzipplus_anchor_merge")
class VisionZipPlusAnchorMergeStrategy(TokenPruneStrategy):
    """Anchor + soft merge.

    Pipeline:
    1) 用 anchor 方法选出 K 个保留 tokens（K 在小预算内自适应）。
    2) 对被丢弃 tokens，不直接丢掉；按与保留 token 的相似度做 soft assignment，
       把信息“融回去”（weighted sum merge），token 数仍然是 K。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_keep = int(config.get("min_keep", 80))
        self.max_keep = int(config.get("max_keep", 128))
        self.pool_mult = int(config.get("pool_mult", 4))
        self.alpha = float(config.get("alpha", 0.65))
        self.spatial_bias = float(config.get("spatial_bias", 0.03))
        self.grid_bins = int(config.get("grid_bins", 6))
        self.anchor_points = config.get(
            "anchor_points",
            ["tl", "tr", "bl", "br", "center", "top", "bottom", "left", "right"],
        )
        self.merge_temp = float(config.get("merge_temp", 5.0))  # higher => sharper assignment
        self._debug_print_limit = int(config.get("debug_print_limit", 3))
        self._debug_print_count = 0

    def apply(self, model: nn.Module, images: torch.Tensor, image_features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._apply(image_features)

    def apply_after_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        return self._apply(image_features)

    # Reuse the anchor implementation by instantiating a helper with same params.
    def _select_keep_indices(self, image_features: torch.Tensor) -> torch.Tensor:
        helper = VisionZipPlusAnchorStrategy(
            {
                "min_keep": self.min_keep,
                "max_keep": self.max_keep,
                "pool_mult": self.pool_mult,
                "alpha": self.alpha,
                "spatial_bias": self.spatial_bias,
                "grid_bins": self.grid_bins,
                "anchor_points": self.anchor_points,
                "debug_print_limit": 0,
            }
        )
        # helper._apply returns [B, K, D], but we need indices; replicate minimal parts:
        # We'll re-run the selection logic but return sel_idx.
        b, t, d = image_features.shape
        keep_all = []
        for bi in range(b):
            feats = image_features[bi]
            feats_f = feats.float()
            imp = torch.norm(feats_f, dim=-1)
            imp = torch.nan_to_num(imp, nan=0.0, posinf=1e4, neginf=0.0)
            imp = imp + helper._spatial_bonus(t, feats.device, imp.dtype)

            imp_std = float(torch.std(imp).item())
            imp_mean = float(torch.mean(imp).item()) + 1e-6
            cv = imp_std / imp_mean
            keep_total = int(round(self.max_keep - (self.max_keep - self.min_keep) * min(max(cv, 0.0), 1.0)))
            keep_total = max(self.min_keep, min(self.max_keep, keep_total))
            keep_total = min(keep_total, t)

            anchor_idx = helper._anchor_indices(t, feats.device)
            if anchor_idx.numel() > keep_total:
                anchor_scores = imp[anchor_idx]
                keep_anchor = torch.topk(anchor_scores, keep_total, largest=True).indices
                anchor_idx = anchor_idx[keep_anchor]

            remaining_budget = max(0, keep_total - anchor_idx.numel())
            if remaining_budget == 0:
                sel_idx = anchor_idx
            else:
                mask = torch.ones(t, device=feats.device, dtype=torch.bool)
                if anchor_idx.numel() > 0:
                    mask[anchor_idx] = False
                rem_idx = mask.nonzero(as_tuple=False).squeeze(1)
                rem_imp_full = imp[rem_idx]

                pool_k = min(rem_idx.numel(), remaining_budget * max(2, self.pool_mult))
                num_spatial, offset, side = helper._spatial_layout(t)
                if num_spatial is not None and side is not None and side > 1:
                    bins_local = helper._pick_with_spatial_bins(
                        rem_imp_full, min(pool_k, remaining_budget * 2), side, max(0, offset - int(anchor_idx.numel() > 0))
                    )
                    top_local = torch.topk(rem_imp_full, pool_k, largest=True).indices
                    cand_local = torch.unique(torch.cat([bins_local, top_local], dim=0))[:pool_k]
                else:
                    cand_local = torch.topk(rem_imp_full, pool_k, largest=True).indices

                cand_idx = rem_idx[cand_local]
                cand = feats_f[cand_idx]
                cand_imp = imp[cand_idx]
                cand_norm = torch.nn.functional.normalize(cand, p=2, dim=-1)

                imp_min = float(torch.min(cand_imp).item())
                imp_max = float(torch.max(cand_imp).item())
                imp_norm = (cand_imp - imp_min) / max(imp_max - imp_min, 1e-6)

                if anchor_idx.numel() > 0:
                    anchor_norm = torch.nn.functional.normalize(feats_f[anchor_idx], p=2, dim=-1)
                    sim = torch.matmul(cand_norm, anchor_norm.transpose(0, 1))
                    min_dist = 1.0 - sim.max(dim=1).values
                else:
                    first = int(torch.argmax(cand_imp).item())
                    min_dist = 1.0 - torch.matmul(cand_norm, cand_norm[first].unsqueeze(-1)).squeeze(-1)

                sel_local = []
                while len(sel_local) < remaining_budget:
                    score = self.alpha * imp_norm + (1.0 - self.alpha) * min_dist
                    if sel_local:
                        score[sel_local] = -1e9
                    nxt = int(torch.argmax(score).item())
                    if nxt < 0:
                        break
                    sel_local.append(nxt)
                    dist_to_new = 1.0 - torch.matmul(cand_norm, cand_norm[nxt].unsqueeze(-1)).squeeze(-1)
                    min_dist = torch.minimum(min_dist, dist_to_new)

                diverse_idx = cand_idx[torch.tensor(sel_local, device=cand_idx.device, dtype=torch.long)]
                sel_idx = torch.cat([anchor_idx, diverse_idx], dim=0) if anchor_idx.numel() > 0 else diverse_idx

            keep_all.append(sel_idx)
        # Pad keep indices per batch (rarely b>1). For b==1, just return [K].
        return keep_all[0] if b == 1 else keep_all

    def _apply(self, image_features: torch.Tensor) -> torch.Tensor:
        b, t, d = image_features.shape
        outs = []
        for bi in range(b):
            feats = image_features[bi]              # [T, D] original dtype
            feats_f = feats.float()                 # [T, D] FP32 for similarity/merge
            keep_idx = self._select_keep_indices(feats.unsqueeze(0))
            if isinstance(keep_idx, list):
                keep_idx = keep_idx[bi]
            keep_idx = keep_idx.to(feats.device)

            kept = feats_f[keep_idx]                # [K, D]
            dropped_mask = torch.ones(t, device=feats.device, dtype=torch.bool)
            dropped_mask[keep_idx] = False
            drop_idx = dropped_mask.nonzero(as_tuple=False).squeeze(1)
            if drop_idx.numel() == 0:
                out = feats[keep_idx].unsqueeze(0)
                outs.append(out)
                continue

            dropped = feats_f[drop_idx]              # [R, D]
            kept_n = torch.nn.functional.normalize(kept, p=2, dim=-1)
            drop_n = torch.nn.functional.normalize(dropped, p=2, dim=-1)
            sim = torch.matmul(drop_n, kept_n.transpose(0, 1))  # [R, K]
            w = torch.softmax(sim * self.merge_temp, dim=1)     # [R, K]
            # Aggregate dropped tokens into kept tokens: KxD
            agg = torch.matmul(w.transpose(0, 1), dropped)      # [K, D]
            merged = kept + agg

            out = merged.to(dtype=feats.dtype).unsqueeze(0)     # [1, K, D]
            if self._debug_print_count < self._debug_print_limit:
                print(f"VisionZipPlusAnchorMerge: pruned from {t} to {out.shape[1]} tokens (merge_temp={self.merge_temp})")
                self._debug_print_count += 1
            outs.append(out)
        return torch.cat(outs, dim=0)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "min_keep": 80,
            "max_keep": 128,
            "pool_mult": 4,
            "alpha": 0.65,
            "spatial_bias": 0.03,
            "grid_bins": 6,
            "anchor_points": ["tl", "tr", "bl", "br", "center", "top", "bottom", "left", "right"],
            "merge_temp": 5.0,
            "debug_print_limit": 3,
        }


def apply_strategy_by_name(model: nn.Module, strategy_name: str, 
                          config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    应用指定名称的策略到模型
    
    Args:
        model: 模型
        strategy_name: 策略名称
        config: 策略配置
        
    Returns:
        应用策略后的模型
    """
    global _GLOBAL_STRATEGY, _GLOBAL_STRATEGY_NAME
    
    if strategy_name == "visionzip":
        try:
            # 设置全局策略实例
            _GLOBAL_STRATEGY_NAME = strategy_name
            cfg = config or VisionZipStrategy.get_default_config()
            _GLOBAL_STRATEGY = VisionZipStrategy(cfg)
            
            dominant = cfg.get('dominant', 54)
            contextual = cfg.get('contextual', 10)
            print(f"Applied VisionZip strategy: dominant={dominant}, contextual={contextual}")
            
        except Exception as e:
            print(f"Failed to apply VisionZip: {e}")
            import traceback
            traceback.print_exc()
    elif strategy_name == "visionzipplus":
        try:
            _GLOBAL_STRATEGY_NAME = strategy_name
            cfg = config or VisionZipPlusStrategy.get_default_config()
            _GLOBAL_STRATEGY = VisionZipPlusStrategy(cfg)
            
            dominant = cfg.get('dominant', 54)
            contextual = cfg.get('contextual', 10)
            print(f"Applied VisionZipPlus strategy: dominant={dominant}, contextual={contextual}")
            
        except Exception as e:
            print(f"Failed to apply VisionZipPlus: {e}")
            import traceback
            traceback.print_exc()
    elif strategy_name == "aim":
        try:
            _GLOBAL_STRATEGY_NAME = strategy_name
            cfg = config or AIMStrategy.get_default_config()
            _GLOBAL_STRATEGY = AIMStrategy(cfg)
            print(f"Applied AIM strategy")
        except Exception as e:
            print(f"Failed to apply AIM: {e}")
    elif strategy_name != "none":
        try:
            _GLOBAL_STRATEGY_NAME = strategy_name
            _GLOBAL_STRATEGY = get_strategy(strategy_name, config)
            print(f"Applied strategy: {strategy_name}")
        except Exception as e:
            print(f"Failed to apply strategy {strategy_name}: {e}")
    
    return model


def get_global_strategy() -> Optional[TokenPruneStrategy]:
    """获取当前全局策略"""
    return _GLOBAL_STRATEGY


def get_global_strategy_name() -> Optional[str]:
    """获取当前全局策略名称"""
    return _GLOBAL_STRATEGY_NAME


def apply_global_strategy(model: nn.Module, images: torch.Tensor, 
                         image_features: torch.Tensor) -> torch.Tensor:
    """
    应用全局策略到图像特征
    
    Args:
        model: 模型
        images: 输入图像
        image_features: 原始图像特征
        
    Returns:
        处理后的图像特征
    """
    global _GLOBAL_STRATEGY
    if _GLOBAL_STRATEGY is not None:
        try:
            return _GLOBAL_STRATEGY.apply(model, images, image_features)
        except Exception as e:
            print(f"Error applying global strategy: {e}")
    return image_features


# 导出可用策略列表
def list_strategies() -> list:
    """列出所有可用的策略"""
    return list(STRATEGY_REGISTRY.keys())
