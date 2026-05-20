"""Qwen3-VL token reduction patch used by AIM.

这个模块不再使用旧的 VisionZip soft-mask 方案，而是直接在 AIM 仓库内
移植一套 PixelPrune 风格的 **hard-prune** 逻辑：

1. 基于像素级 patch 内容计算每张图的 merged-token keep indices；
2. 在 ViT 输入层（或指定的中间层）真实裁剪 patch token；
3. 同步裁剪 LLM 侧的 image placeholders / input_ids / attention_mask；
4. 重新对齐 Qwen3-VL 的 3D position ids / mRoPE。

对外仍保留 ``apply_visionzip_to_qwen3_vl`` 这个历史 API 名称，
方便 AIM 现有脚本继续通过 ``token_prune_strategy=visionzip`` 进入新实现。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


_PATCH_CONFIG: Dict[str, Any] = {
    "selector": "pixelprune",
    "method": "pred_2d",
    "metric": "max",
    "threshold": 0.0,
    "vit_prune_layer": 0,
    "anchored": True,
    "verbose": False,
    "visionzip_dominant": 64,
    "visionzip_contextual": 16,
    "visionzip_target_keep": 80,
    "bishe_target_keep": 96,
    "bishe_merge_steps": [(192, 192), (144, 144), (96, 96), (48, 48)],
    "bishe_anchor_points": ["grid4x4"],
    "bishe_protected_penalty": 0.09,
    "bishe_qk_mix": 0.20,
    "bishe_pos_mix": 0.05,
    "bishe_metric_layer": -2,
    "bishe_anchor_order_mode": "block",
    "bishe_anchor_order_group_size": 3,
    "bishe_anchor_order_first_stage_only": True,
}

_ORIG_VISION_FORWARD = None
_ORIG_MODEL_GET_IMAGE_FEATURES = None
_ORIG_MODEL_FORWARD = None
_ORIG_CG_FORWARD = None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_pixel_values_for_selector(pixel_values: torch.Tensor) -> torch.Tensor:
    """把 Qwen3-VL patch pixel values 恢复到 selector 使用的 [0,1] 空间。"""
    c, t, h, w = 3, 2, 16, 16
    pv_reshaped = pixel_values.view(-1, c, t, h, w)
    frame_0 = pv_reshaped[:, :, 0, :, :]
    normalized = (frame_0 * 0.5) + 0.5
    return normalized.reshape(pixel_values.shape[0], -1)


def _sim2d(x: torch.Tensor, y: torch.Tensor, method: str, threshold: float) -> torch.Tensor:
    diff = x - y
    if method in ("max", "exact"):
        dist = diff.abs().amax(dim=-1)
        thr = 0.0 if method == "exact" else threshold
    elif method == "mae":
        dist = diff.abs().mean(dim=-1)
        thr = threshold
    elif method == "rmse":
        dist = diff.pow(2).mean(dim=-1).sqrt()
        thr = threshold
    else:
        raise ValueError(f"Unsupported token reduction metric: {method!r}")
    return dist <= thr


def _prepare_merged(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, List[int]]:
    block_size = spatial_merge_size * spatial_merge_size
    merged_pv = pixel_values.reshape(-1, pixel_values.shape[-1] * block_size)
    merged_lengths = [int(t * h * w) // block_size for t, h, w in image_grid_thw.tolist()]
    return merged_pv, merged_lengths


def _select_2d_loco_fast(
    tokens: torch.Tensor,
    h: int,
    w: int,
    method: str,
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    d = tokens.shape[-1]
    g = tokens.view(h, w, d)
    keep = torch.zeros(h, w, dtype=torch.bool, device=device)
    keep[0, 0] = True

    if w > 1:
        keep[0, 1:] = ~_sim2d(g[0:1, 1:], g[0:1, :-1], method, threshold)[0]
    if h > 1:
        keep[1:, 0] = ~_sim2d(g[1:, 0:1], g[:-1, 0:1], method, threshold)[:, 0]
    if h > 1 and w > 1:
        x = g[1:, 1:]
        a = g[1:, :-1]
        b = g[:-1, 1:]
        c = g[:-1, :-1]
        cb = _sim2d(c, b, method, threshold)
        ca = _sim2d(c, a, method, threshold)
        use_b = ca & ~cb
        pred = torch.where(use_b.unsqueeze(-1), b, a)
        keep[1:, 1:] = ~_sim2d(x, pred, method, threshold)
    return keep.flatten().nonzero(as_tuple=False)[:, 0]


def _select_2d_loco_anchored(
    tokens: torch.Tensor,
    h: int,
    w: int,
    method: str,
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    d = tokens.shape[-1]
    g = tokens.view(h, w, d)
    keep = torch.zeros(h, w, dtype=torch.bool, device=device)
    keep[0, 0] = True
    anchor_grid = g.clone()

    def sim(a: torch.Tensor, b: torch.Tensor) -> bool:
        return bool(_sim2d(a.unsqueeze(0), b.unsqueeze(0), method, threshold).item())

    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                continue
            if c > 0:
                a = anchor_grid[r, c - 1]
            elif r > 0:
                a = anchor_grid[r - 1, 0]
            else:
                a = None

            b = anchor_grid[r - 1, c] if r > 0 else a
            c_tok = anchor_grid[r - 1, c - 1] if (r > 0 and c > 0) else b

            if r > 0 and c > 0:
                cb = sim(c_tok, b)
                ca = sim(c_tok, a)
                pred = b if (ca and not cb) else a
            else:
                pred = a

            x = g[r, c]
            if not sim(x, pred):
                keep[r, c] = True
                anchor_grid[r, c] = x
            else:
                anchor_grid[r, c] = pred
    return keep.flatten().nonzero(as_tuple=False)[:, 0]


def _compute_merged_keep_indices(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> List[torch.Tensor]:
    method = str(_PATCH_CONFIG.get("method", "pred_2d")).lower()
    metric = str(_PATCH_CONFIG.get("metric", "max")).lower()
    threshold = float(_PATCH_CONFIG.get("threshold", 0.0))
    anchored = bool(_PATCH_CONFIG.get("anchored", True))
    if method not in {"pred_2d", "pred2d", "loco", "pixelprune"}:
        raise ValueError(
            f"AIM 目前只支持移植后的 pred_2d / pixelprune 风格实现，收到 method={method!r}"
        )

    merged_pv, merged_lengths = _prepare_merged(pixel_values, image_grid_thw, spatial_merge_size)
    device = pixel_values.device
    merged_indices_list = []
    offset = 0
    for length, (t, h, w) in zip(merged_lengths, image_grid_thw.tolist()):
        img_merged = merged_pv[offset: offset + length]
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        if metric == "exact" or (metric == "max" and threshold == 0.0) or not anchored:
            indices = _select_2d_loco_fast(img_merged, merged_h, merged_w, metric, threshold, device)
        else:
            indices = _select_2d_loco_anchored(img_merged, merged_h, merged_w, metric, threshold, device)
        merged_indices_list.append(indices.to(device))
        offset += length
    return merged_indices_list


def _merged_indices_to_patch_indices(
    merged_indices_list: List[torch.Tensor],
    block_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    patch_indices_list: List[torch.Tensor] = []
    for merged_indices in merged_indices_list:
        patch_indices = []
        for midx in merged_indices.tolist():
            base = int(midx) * block_size
            patch_indices.extend(range(base, base + block_size))
        patch_indices_list.append(torch.tensor(patch_indices, device=device, dtype=torch.long))
    return patch_indices_list


def _build_grid_pos_code(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def _grid_anchor_indices(h: int, w: int, anchor_points: List[str], device: torch.device) -> torch.Tensor:
    point_to_idx = {
        "tl": 0,
        "tr": w - 1,
        "bl": (h - 1) * w,
        "br": (h - 1) * w + w - 1,
        "center": (h // 2) * w + w // 2,
        "top": w // 2,
        "bottom": (h - 1) * w + w // 2,
        "left": (h // 2) * w,
        "right": (h // 2) * w + w - 1,
    }

    def grid_points(side_len: int, n: int) -> List[int]:
        if n <= 1:
            return [side_len // 2]
        return [round(i * (side_len - 1) / (n - 1)) for i in range(n)]

    idxs: List[int] = []
    for name in anchor_points:
        if name.startswith("grid") and "x" in name:
            parts = name[len("grid"):].split("x", 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                gy = int(parts[0])
                gx = int(parts[1])
                ys = grid_points(h, gy)
                xs = grid_points(w, gx)
                for y in ys:
                    for x in xs:
                        idxs.append(y * w + x)
                continue
        idx = point_to_idx.get(name)
        if idx is not None:
            idxs.append(idx)

    idxs = sorted(set(i for i in idxs if 0 <= i < h * w))
    return torch.tensor(idxs, device=device, dtype=torch.long)


def _build_anchor_aware_order(
    num_tokens: int,
    anchor_idx: torch.Tensor,
    device: torch.device,
    mode: str,
    group_size: int,
) -> Optional[torch.Tensor]:
    if num_tokens <= 1 or anchor_idx.numel() == 0:
        return None

    all_idx = torch.arange(num_tokens, device=device, dtype=torch.long)
    is_anchor = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    is_anchor[anchor_idx.clamp(min=0, max=num_tokens - 1)] = True
    anchors = all_idx[is_anchor]
    non_anchors = all_idx[~is_anchor]

    if mode == "alternate":
        out: List[torch.Tensor] = []
        ia = 0
        inon = 0
        while ia < anchors.numel() or inon < non_anchors.numel():
            if ia < anchors.numel():
                out.append(anchors[ia])
                ia += 1
            if inon < non_anchors.numel():
                out.append(non_anchors[inon])
                inon += 1
        return torch.stack(out) if out else None

    if mode == "balanced_pairs":
        out = []
        ia = 0
        inon = 0
        while ia < anchors.numel() or inon < non_anchors.numel():
            if ia < anchors.numel():
                out.append(anchors[ia])
                ia += 1
            elif inon < non_anchors.numel():
                out.append(non_anchors[inon])
                inon += 1
            if inon < non_anchors.numel():
                out.append(non_anchors[inon])
                inon += 1
            elif ia < anchors.numel():
                out.append(anchors[ia])
                ia += 1
        return torch.stack(out) if out else None

    if mode == "block":
        g = max(1, int(group_size))
        out = []
        ia = 0
        inon = 0
        while ia < anchors.numel() or inon < non_anchors.numel():
            for _ in range(g):
                if ia < anchors.numel():
                    out.append(anchors[ia])
                    ia += 1
            for _ in range(g):
                if inon < non_anchors.numel():
                    out.append(non_anchors[inon])
                    inon += 1
        return torch.stack(out) if out else None

    return None


def _dynamic_merge_steps(num_tokens: int, target_keep: int) -> List[Tuple[int, int]]:
    steps: List[Tuple[int, int]] = []
    cur = int(num_tokens)
    target_keep = max(1, min(int(target_keep), cur))
    while cur > target_keep:
        r = min(cur // 2, cur - target_keep)
        if r <= 0:
            break
        steps.append((r, r))
        cur -= r
    return steps


def _resolve_bishe_merge_steps(num_tokens: int, target_keep: int) -> List[Tuple[int, int]]:
    steps: List[Tuple[int, int]] = []
    cur = int(num_tokens)
    target_keep = max(1, min(int(target_keep), cur))

    configured = _PATCH_CONFIG.get("bishe_merge_steps") or []
    for pair in configured:
        if cur <= target_keep:
            break
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        r_feat = min(int(pair[0]), cur // 2, cur - target_keep)
        r_metric = min(int(pair[1]), cur // 2, cur - target_keep)
        if r_feat <= 0 or r_metric <= 0:
            continue
        steps.append((r_feat, r_metric))
        cur -= r_feat

    if cur > target_keep:
        steps.extend(_dynamic_merge_steps(cur, target_keep))
    return steps


def _pool_patch_tokens_to_merged(
    packed_tensor: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> List[torch.Tensor]:
    pooled_list: List[torch.Tensor] = []
    offset = 0
    block_size = spatial_merge_size ** 2
    for t, h, w in image_grid_thw.tolist():
        seq_len = int(t) * int(h) * int(w)
        seq = packed_tensor[offset: offset + seq_len]
        if seq.shape[0] == 0:
            pooled = seq
        elif seq.shape[0] % block_size != 0:
            pooled = seq
        else:
            pooled = seq.view(-1, block_size, seq.shape[-1]).mean(dim=1)
        pooled_list.append(pooled)
        offset += seq_len
    return pooled_list


def _extract_qk_metric_from_qwen3_visual(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """从 Qwen3-VL vision tower 抽取 stage-1 的 q/k 度量。

    注意 Qwen3-VL vision attention 使用单个 `qkv` 线性层，而不是 `q_proj` / `k_proj`。
    之前 bishe 的 Qwen3 hard-prune 适配层没有真正使用 q/k，只是拿 merger 后的
    image features + pos 作为 metric，这会让 `bishe_qk_mix` 形同虚设。
    """
    try:
        visual = model.visual
        pixel_values_typed = pixel_values.type(visual.dtype)
        hidden_states = visual.patch_embed(pixel_values_typed)
        pos_embeds = visual.fast_pos_embed_interpolate(image_grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = visual.rot_pos_emb(image_grid_thw).reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(hidden_states.device)

        num_blocks = len(visual.blocks)
        metric_layer = int(_PATCH_CONFIG.get("bishe_metric_layer", -2))
        if metric_layer < 0:
            metric_layer = num_blocks + metric_layer
        metric_layer = max(0, min(metric_layer, num_blocks - 1))

        q = None
        k = None
        for layer_idx, blk in enumerate(visual.blocks):
            if layer_idx == metric_layer:
                attn_input = blk.norm1(hidden_states)
                qkv = blk.attn.qkv(attn_input).reshape(attn_input.shape[0], 3, blk.attn.num_heads, -1)
                q = qkv[:, 0].reshape(attn_input.shape[0], -1)
                k = qkv[:, 1].reshape(attn_input.shape[0], -1)
                break
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens.int(),
                position_embeddings=position_embeddings,
            )

        if q is None or k is None:
            return None, None

        spatial_merge_size = getattr(visual.config, "spatial_merge_size", 2)
        q_list = _pool_patch_tokens_to_merged(q, image_grid_thw, spatial_merge_size)
        k_list = _pool_patch_tokens_to_merged(k, image_grid_thw, spatial_merge_size)
        return q_list, k_list
    except Exception:
        return None, None


def _bipartite_merge_with_indices(
    metric: torch.Tensor,
    r: int,
    x: torch.Tensor,
    orig_idx: torch.Tensor,
    protected_orig_idx: Optional[torch.Tensor] = None,
    protected_penalty: Optional[float] = None,
    token_order: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    t = metric.shape[0]
    r = min(int(r), t // 2)
    if r <= 0:
        return None, None

    if token_order is not None:
        token_order = token_order.to(device=metric.device, dtype=torch.long)
        metric = metric.index_select(0, token_order)
        x = x.index_select(0, token_order)
        orig_idx = orig_idx.index_select(0, token_order)

    metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a, b = metric[::2], metric[1::2]
    if a.numel() == 0 or b.numel() == 0:
        return None, None
    scores = a @ b.transpose(0, 1)
    node_max, node_idx = scores.max(dim=-1)

    if protected_orig_idx is not None and protected_orig_idx.numel() > 0:
        prot = protected_orig_idx.to(device=metric.device, dtype=orig_idx.dtype)
        a_orig = orig_idx[::2]
        a_mask = (a_orig.unsqueeze(0) == prot.unsqueeze(1)).any(dim=0)
        node_max_rank = node_max.float()
        if a_mask.any():
            if protected_penalty is None:
                node_max_rank[a_mask] = -1e9
            else:
                node_max_rank[a_mask] -= float(protected_penalty)
        edge_idx = node_max_rank.argsort(dim=-1, descending=True)
    else:
        edge_idx = node_max.argsort(dim=-1, descending=True)

    unm_idx = edge_idx[r:]
    src_idx = edge_idx[:r]
    dst_idx = node_idx.gather(0, src_idx)

    src, dst = x[::2], x[1::2]
    unm = src.index_select(0, unm_idx)
    src_sel = src.index_select(0, src_idx)
    dst = dst.clone().scatter_reduce(
        0,
        dst_idx.unsqueeze(-1).expand(-1, dst.shape[-1]),
        src_sel,
        reduce="mean",
        include_self=True,
    )
    new_x = torch.cat([unm, dst], dim=0)
    new_orig_idx = torch.cat([orig_idx[::2].index_select(0, unm_idx), orig_idx[1::2]], dim=0)
    return new_x, new_orig_idx


def _build_bipartite_merge_plan(
    metric: torch.Tensor,
    r: int,
    orig_idx: torch.Tensor,
    protected_orig_idx: Optional[torch.Tensor] = None,
    protected_penalty: Optional[float] = None,
    token_order: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    t = metric.shape[0]
    r = min(int(r), t // 2)
    if r <= 0:
        return None

    reordered_metric = metric
    reordered_orig_idx = orig_idx
    reordered_token_order = None
    if token_order is not None:
        reordered_token_order = token_order.to(device=metric.device, dtype=torch.long)
        reordered_metric = reordered_metric.index_select(0, reordered_token_order)
        reordered_orig_idx = reordered_orig_idx.index_select(0, reordered_token_order)

    reordered_metric = reordered_metric / reordered_metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a, b = reordered_metric[::2], reordered_metric[1::2]
    if a.numel() == 0 or b.numel() == 0:
        return None

    scores = a @ b.transpose(0, 1)
    node_max, node_idx = scores.max(dim=-1)

    if protected_orig_idx is not None and protected_orig_idx.numel() > 0:
        prot = protected_orig_idx.to(device=metric.device, dtype=reordered_orig_idx.dtype)
        a_orig = reordered_orig_idx[::2]
        a_mask = (a_orig.unsqueeze(0) == prot.unsqueeze(1)).any(dim=0)
        node_max_rank = node_max.float()
        if a_mask.any():
            if protected_penalty is None:
                node_max_rank[a_mask] = -1e9
            else:
                node_max_rank[a_mask] -= float(protected_penalty)
        edge_idx = node_max_rank.argsort(dim=-1, descending=True)
    else:
        edge_idx = node_max.argsort(dim=-1, descending=True)

    unm_idx = edge_idx[r:]
    src_idx = edge_idx[:r]
    dst_idx = node_idx.gather(0, src_idx)
    new_orig_idx = torch.cat([reordered_orig_idx[::2].index_select(0, unm_idx), reordered_orig_idx[1::2]], dim=0)

    return {
        "token_order": reordered_token_order,
        "unm_idx": unm_idx,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "new_orig_idx": new_orig_idx,
    }


def _apply_bipartite_merge_plan(x: torch.Tensor, plan: Dict[str, torch.Tensor]) -> torch.Tensor:
    token_order = plan.get("token_order")
    if token_order is not None:
        x = x.index_select(0, token_order.to(device=x.device, dtype=torch.long))

    src, dst = x[::2], x[1::2]
    unm_idx = plan["unm_idx"].to(device=x.device, dtype=torch.long)
    src_idx = plan["src_idx"].to(device=x.device, dtype=torch.long)
    dst_idx = plan["dst_idx"].to(device=x.device, dtype=torch.long)

    unm = src.index_select(0, unm_idx)
    src_sel = src.index_select(0, src_idx)
    dst = dst.clone().scatter_reduce(
        0,
        dst_idx.unsqueeze(-1).expand(-1, dst.shape[-1]),
        src_sel,
        reduce="mean",
        include_self=True,
    )
    return torch.cat([unm, dst], dim=0)


def _split_packed_per_image(
    packed_tensor: torch.Tensor,
    image_grid_thw: torch.Tensor,
    block_size: int,
) -> List[torch.Tensor]:
    split_sizes = (image_grid_thw.prod(-1) // block_size).tolist()
    return list(torch.split(packed_tensor, split_sizes))


def _compute_bishe_soft_merged_outputs(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], List[torch.Tensor]]:
    image_outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    image_embeds = list(image_outputs.pooler_output)
    deepstack_image_embeds = image_outputs.deepstack_features
    q_list, k_list = _extract_qk_metric_from_qwen3_visual(model, pixel_values, image_grid_thw)

    device = pixel_values.device
    target_keep = int(_PATCH_CONFIG.get("bishe_target_keep", 96))
    anchor_points = list(_PATCH_CONFIG.get("bishe_anchor_points", ["grid4x4"]))
    protected_penalty = float(_PATCH_CONFIG.get("bishe_protected_penalty", 0.09))
    qk_mix = float(_PATCH_CONFIG.get("bishe_qk_mix", 0.20))
    pos_mix = float(_PATCH_CONFIG.get("bishe_pos_mix", 0.05))
    order_mode = str(_PATCH_CONFIG.get("bishe_anchor_order_mode", "block"))
    order_group_size = int(_PATCH_CONFIG.get("bishe_anchor_order_group_size", 3))
    first_stage_only = bool(_PATCH_CONFIG.get("bishe_anchor_order_first_stage_only", True))
    verbose = bool(_PATCH_CONFIG.get("verbose", False))
    spatial_merge_size = getattr(model.visual.config, "spatial_merge_size", 2)
    block_size = spatial_merge_size ** 2

    deepstack_per_layer = None
    if deepstack_image_embeds:
        deepstack_per_layer = [
            _split_packed_per_image(ds_layer_embeds, image_grid_thw, block_size)
            for ds_layer_embeds in deepstack_image_embeds
        ]

    merged_image_embeds: List[torch.Tensor] = []
    merged_orig_indices: List[torch.Tensor] = []
    merged_deepstack_per_image: Optional[List[List[torch.Tensor]]] = [] if deepstack_per_layer is not None else None

    for img_idx, (seq_embeds, (_t, h, w)) in enumerate(zip(image_embeds, image_grid_thw.tolist())):
        merged_h = int(h) // spatial_merge_size
        merged_w = int(w) // spatial_merge_size
        image_features = seq_embeds.to(device)
        orig_num = image_features.shape[0]

        ds_features_this_img = None
        if deepstack_per_layer is not None:
            ds_features_this_img = [ds_layer[img_idx].to(device) for ds_layer in deepstack_per_layer]

        if orig_num <= target_keep:
            final_orig_idx = torch.arange(orig_num, device=device, dtype=torch.long)
            merged_image_embeds.append(image_features)
            merged_orig_indices.append(final_orig_idx)
            if merged_deepstack_per_image is not None:
                merged_deepstack_per_image.append(ds_features_this_img or [])
            continue

        metric_base = image_features
        q_metric = q_list[img_idx] if q_list is not None and img_idx < len(q_list) else None
        k_metric = k_list[img_idx] if k_list is not None and img_idx < len(k_list) else None
        if q_metric is not None:
            q_metric = q_metric.to(device=image_features.device, dtype=image_features.dtype)
        if k_metric is not None:
            k_metric = k_metric.to(device=image_features.device, dtype=image_features.dtype)
        if q_metric is not None and k_metric is not None and q_metric.shape == k_metric.shape == image_features.shape:
            metric_base = (1.0 - qk_mix) * k_metric + qk_mix * q_metric
        elif k_metric is not None and k_metric.shape == image_features.shape:
            metric_base = k_metric
        elif q_metric is not None and q_metric.shape == image_features.shape:
            metric_base = q_metric

        pos_code = _build_grid_pos_code(merged_h, merged_w, image_features.device, image_features.dtype)
        metric_first = metric_base
        if pos_mix > 0 and pos_code.shape[0] == image_features.shape[0]:
            metric_first = torch.cat([metric_base, pos_mix * pos_code], dim=-1)

        protected_orig_idx = _grid_anchor_indices(merged_h, merged_w, anchor_points, image_features.device)
        current_feat = image_features
        current_metric = metric_first
        current_orig_idx = torch.arange(orig_num, device=image_features.device, dtype=torch.long)
        current_ds = ds_features_this_img
        first = True

        for r_feat, _r_metric in _resolve_bishe_merge_steps(orig_num, target_keep):
            if current_feat.shape[0] <= target_keep or current_metric.shape[0] <= 1:
                break

            token_order = None
            if protected_orig_idx.numel() > 0 and (first or (not first_stage_only)):
                current_protected = torch.nonzero(
                    (current_orig_idx.unsqueeze(0) == protected_orig_idx.unsqueeze(1)).any(dim=0),
                    as_tuple=False,
                ).squeeze(1)
                token_order = _build_anchor_aware_order(
                    current_feat.shape[0],
                    current_protected,
                    current_feat.device,
                    order_mode,
                    order_group_size,
                )

            plan = _build_bipartite_merge_plan(
                metric=current_metric,
                r=min(r_feat, current_feat.shape[0] - target_keep),
                orig_idx=current_orig_idx,
                protected_orig_idx=protected_orig_idx if first else None,
                protected_penalty=protected_penalty if first else None,
                token_order=token_order,
            )
            if plan is None:
                break

            current_feat = _apply_bipartite_merge_plan(current_feat, plan)
            if first:
                current_metric = current_feat
            else:
                current_metric = _apply_bipartite_merge_plan(current_metric, plan)
                if current_metric.shape[0] != current_feat.shape[0]:
                    current_metric = current_feat
            current_orig_idx = plan["new_orig_idx"].to(device=current_feat.device, dtype=torch.long)
            if current_ds is not None:
                current_ds = [_apply_bipartite_merge_plan(ds_feat, plan) for ds_feat in current_ds]
            first = False

        sort_order = current_orig_idx.argsort()
        final_orig_idx = current_orig_idx.index_select(0, sort_order).to(device=device, dtype=torch.long)
        final_feat = current_feat.index_select(0, sort_order).to(device)
        merged_image_embeds.append(final_feat)
        merged_orig_indices.append(final_orig_idx)
        if merged_deepstack_per_image is not None:
            final_ds = []
            if current_ds is not None:
                final_ds = [ds_feat.index_select(0, sort_order).to(device) for ds_feat in current_ds]
            merged_deepstack_per_image.append(final_ds)

        if verbose:
            print(
                "[AIM Qwen3-VL token reduction] bishe soft-merged tokens "
                f"{orig_num} -> {final_feat.shape[0]} | target_keep={target_keep} "
                f"order={order_mode}:{order_group_size} anchors={anchor_points} "
                f"qk_mix={qk_mix} pos_mix={pos_mix} penalty={protected_penalty}"
            )

    merged_deepstack_packed = None
    if merged_deepstack_per_image is not None:
        num_layers = len(merged_deepstack_per_image[0]) if merged_deepstack_per_image else 0
        merged_deepstack_packed = []
        for layer_idx in range(num_layers):
            merged_deepstack_packed.append(
                torch.cat([img_layers[layer_idx] for img_layers in merged_deepstack_per_image], dim=0)
            )

    return merged_image_embeds, merged_deepstack_packed, merged_orig_indices


def _compute_bishe_merged_keep_indices(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> List[torch.Tensor]:
    image_outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    image_embeds = list(image_outputs.pooler_output)
    q_list, k_list = _extract_qk_metric_from_qwen3_visual(model, pixel_values, image_grid_thw)
    device = pixel_values.device
    target_keep = int(_PATCH_CONFIG.get("bishe_target_keep", 96))
    anchor_points = list(_PATCH_CONFIG.get("bishe_anchor_points", ["grid4x4"]))
    protected_penalty = float(_PATCH_CONFIG.get("bishe_protected_penalty", 0.09))
    qk_mix = float(_PATCH_CONFIG.get("bishe_qk_mix", 0.20))
    pos_mix = float(_PATCH_CONFIG.get("bishe_pos_mix", 0.05))
    order_mode = str(_PATCH_CONFIG.get("bishe_anchor_order_mode", "block"))
    order_group_size = int(_PATCH_CONFIG.get("bishe_anchor_order_group_size", 3))
    first_stage_only = bool(_PATCH_CONFIG.get("bishe_anchor_order_first_stage_only", True))
    verbose = bool(_PATCH_CONFIG.get("verbose", False))

    keep_indices_list: List[torch.Tensor] = []
    for img_idx, (seq_embeds, (_t, h, w)) in enumerate(zip(image_embeds, image_grid_thw.tolist())):
        merged_h = int(h) // getattr(model.visual.config, "spatial_merge_size", 2)
        merged_w = int(w) // getattr(model.visual.config, "spatial_merge_size", 2)
        image_features = seq_embeds.to(device)
        orig_num = image_features.shape[0]
        if orig_num <= target_keep:
            keep_indices_list.append(torch.arange(orig_num, device=device, dtype=torch.long))
            continue

        metric_base = image_features
        q_metric = q_list[img_idx] if q_list is not None and img_idx < len(q_list) else None
        k_metric = k_list[img_idx] if k_list is not None and img_idx < len(k_list) else None
        if q_metric is not None:
            q_metric = q_metric.to(device=image_features.device, dtype=image_features.dtype)
        if k_metric is not None:
            k_metric = k_metric.to(device=image_features.device, dtype=image_features.dtype)
        if q_metric is not None and k_metric is not None and q_metric.shape == k_metric.shape == image_features.shape:
            metric_base = (1.0 - qk_mix) * k_metric + qk_mix * q_metric
        elif k_metric is not None and k_metric.shape == image_features.shape:
            metric_base = k_metric
        elif q_metric is not None and q_metric.shape == image_features.shape:
            metric_base = q_metric

        pos_code = _build_grid_pos_code(merged_h, merged_w, image_features.device, image_features.dtype)
        metric_first = metric_base
        if pos_mix > 0 and pos_code.shape[0] == image_features.shape[0]:
            metric_first = torch.cat([metric_base, pos_mix * pos_code], dim=-1)

        protected_orig_idx = _grid_anchor_indices(merged_h, merged_w, anchor_points, image_features.device)
        current_feat = image_features
        current_orig_idx = torch.arange(orig_num, device=image_features.device, dtype=torch.long)
        metric = metric_first
        first = True
        for r_feat, r_metric in _dynamic_merge_steps(orig_num, target_keep):
            if current_feat.shape[0] <= target_keep:
                break
            token_order = None
            if protected_orig_idx.numel() > 0 and (first or (not first_stage_only)):
                current_protected = torch.nonzero(
                    (current_orig_idx.unsqueeze(0) == protected_orig_idx.unsqueeze(1)).any(dim=0),
                    as_tuple=False,
                ).squeeze(1)
                token_order = _build_anchor_aware_order(
                    current_feat.shape[0],
                    current_protected,
                    current_feat.device,
                    order_mode,
                    order_group_size,
                )

            merged_feat, merged_orig_idx = _bipartite_merge_with_indices(
                metric=metric,
                r=min(r_feat, current_feat.shape[0] - target_keep),
                x=current_feat,
                orig_idx=current_orig_idx,
                protected_orig_idx=protected_orig_idx if first else None,
                protected_penalty=protected_penalty if first else None,
                token_order=token_order,
            )
            if merged_feat is None or merged_orig_idx is None:
                break

            current_feat = merged_feat
            current_orig_idx = merged_orig_idx
            metric = current_feat
            first = False

        keep_indices = current_orig_idx.unique(sorted=True)
        if verbose:
            print(
                "[AIM Qwen3-VL token reduction] bishe merged tokens "
                f"{orig_num} -> {len(keep_indices)} | target_keep={target_keep} "
                f"order={order_mode}:{order_group_size} anchors={anchor_points} "
                f"qk_mix={qk_mix} pos_mix={pos_mix} penalty={protected_penalty}"
            )
        keep_indices_list.append(keep_indices.to(device=device, dtype=torch.long))
    return keep_indices_list


def _compute_visionzip_merged_keep_indices(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> List[torch.Tensor]:
    image_outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    image_embeds = list(image_outputs.pooler_output)
    device = pixel_values.device
    verbose = bool(_PATCH_CONFIG.get("verbose", False))
    target_keep = int(_PATCH_CONFIG.get("visionzip_target_keep", 80))
    dominant = int(_PATCH_CONFIG.get("visionzip_dominant", target_keep))
    contextual = int(_PATCH_CONFIG.get("visionzip_contextual", max(0, target_keep - dominant)))

    keep_indices_list: List[torch.Tensor] = []
    for seq_embeds in image_embeds:
        image_features = seq_embeds.to(device)
        num_tokens = int(image_features.shape[0])
        if num_tokens <= 0:
            keep_indices_list.append(torch.empty(0, device=device, dtype=torch.long))
            continue

        keep_total = max(1, min(target_keep, num_tokens))
        dom_keep = max(1, min(dominant, keep_total, num_tokens))

        token_importance = torch.norm(image_features, dim=-1)
        _, dominant_idx = torch.topk(token_importance, dom_keep, largest=True)
        keep_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        keep_mask[dominant_idx] = True

        ctx_keep = max(0, min(contextual, keep_total - dom_keep, num_tokens - dom_keep))
        if ctx_keep > 0 and dom_keep > 0:
            candidate_idx = torch.nonzero(~keep_mask, as_tuple=False).squeeze(1)
            if candidate_idx.numel() > 0:
                dominant_tokens = image_features.index_select(0, dominant_idx)
                contextual_tokens = image_features.index_select(0, candidate_idx)
                dominant_normalized = F.normalize(dominant_tokens, p=2, dim=-1)
                contextual_normalized = F.normalize(contextual_tokens, p=2, dim=-1)
                similarity = contextual_normalized @ dominant_normalized.transpose(0, 1)
                max_similarity = similarity.max(dim=1).values
                _, selected_ctx_rel = torch.topk(
                    max_similarity,
                    k=min(ctx_keep, candidate_idx.numel()),
                    largest=True,
                )
                selected_ctx_idx = candidate_idx.index_select(0, selected_ctx_rel)
                keep_mask[selected_ctx_idx] = True

        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        keep_indices, _ = keep_indices.sort()
        if keep_indices.numel() > keep_total:
            keep_indices = keep_indices[:keep_total]
        if verbose:
            print(
                "[AIM Qwen3-VL token reduction] visionzip merged tokens "
                f"{num_tokens} -> {keep_indices.numel()} | dominant={dom_keep} contextual={ctx_keep} target_keep={keep_total}"
            )
        keep_indices_list.append(keep_indices.to(device=device, dtype=torch.long))
    return keep_indices_list


def _select_packed_by_indices(
    tensor: torch.Tensor,
    grid_thw: torch.Tensor,
    keep_indices: List[torch.Tensor],
) -> torch.Tensor:
    selected = []
    offset = 0
    grid_list = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
    for seq_idx, (t, h, w) in enumerate(grid_list):
        seq_length = int(t) * int(h) * int(w)
        indices = keep_indices[seq_idx].to(tensor.device)
        selected.append(tensor[offset: offset + seq_length][indices])
        offset += seq_length
    return torch.cat(selected, dim=0)


def _adjust_inputs_for_dedup(
    self: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    image_embeds: List[torch.Tensor],
    image_grid_thw: torch.Tensor,
    merged_indices: List[torch.Tensor],
    padding_side: str = "left",
) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
    batch_size, _ = input_ids.shape
    image_token_id = self.config.image_token_id
    vision_start_token_id = self.config.vision_start_token_id
    spatial_merge_size = getattr(self.visual.config, "spatial_merge_size", 2)

    keep_mask = torch.ones(batch_size, input_ids.shape[1], dtype=torch.bool, device="cpu")
    new_inputs_embeds_list = []
    new_input_ids_list = []

    for batch_idx in range(batch_size):
        sample_input_ids = input_ids[batch_idx]
        sample_inputs_embeds = inputs_embeds[batch_idx]
        vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
        if len(vision_start_indices) == 0:
            new_inputs_embeds_list.append(sample_inputs_embeds)
            new_input_ids_list.append(sample_input_ids)
            continue

        image_positions = vision_start_indices[
            sample_input_ids[vision_start_indices + 1] == image_token_id
        ]
        if len(image_positions) == 0:
            new_inputs_embeds_list.append(sample_inputs_embeds)
            new_input_ids_list.append(sample_input_ids)
            continue

        sample_keep_mask = torch.ones(len(sample_input_ids), dtype=torch.bool, device="cpu")
        for img_idx, img_pos in enumerate(image_positions):
            t, h, w = (int(x) for x in image_grid_thw[img_idx])
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            num_original_llm_tokens = t * llm_grid_h * llm_grid_w
            visual_start = int((img_pos + 1).item())
            sample_keep_mask[visual_start: visual_start + num_original_llm_tokens] = False
            sample_keep_mask[visual_start + merged_indices[img_idx].cpu().long()] = True

        keep_mask[batch_idx] = sample_keep_mask
        new_inputs_embeds_list.append(sample_inputs_embeds[sample_keep_mask])
        new_input_ids_list.append(sample_input_ids[sample_keep_mask])

    max_len = max(ids.shape[0] for ids in new_input_ids_list)
    pad_token_id = getattr(self.config, "pad_token_id", 0)
    padded_inputs_embeds = []
    padded_input_ids = []
    for embeds, ids in zip(new_inputs_embeds_list, new_input_ids_list):
        if embeds.shape[0] < max_len:
            pad_len = max_len - embeds.shape[0]
            if padding_side == "left":
                embeds = F.pad(embeds, (0, 0, pad_len, 0), value=0)
                ids = F.pad(ids, (pad_len, 0), value=pad_token_id)
            else:
                embeds = F.pad(embeds, (0, 0, 0, pad_len), value=0)
                ids = F.pad(ids, (0, pad_len), value=pad_token_id)
        padded_inputs_embeds.append(embeds)
        padded_input_ids.append(ids)

    return torch.stack(padded_inputs_embeds), torch.stack(padded_input_ids), keep_mask


def _vision_forward(
    self: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    keep_indices: Optional[List[torch.Tensor]] = None,
    **kwargs: Any,
) -> Any:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

    if keep_indices is None:
        return _ORIG_VISION_FORWARD(self, hidden_states, grid_thw=grid_thw, **kwargs)

    vit_prune_layer = int(_PATCH_CONFIG.get("vit_prune_layer", 0))
    num_blocks = len(self.blocks)
    if vit_prune_layer < 0 or vit_prune_layer > num_blocks:
        raise ValueError(f"vit_prune_layer={vit_prune_layer} out of range [0, {num_blocks}]")

    hidden_states_full = self.patch_embed(hidden_states)
    pos_embeds_full = self.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb_full = self.rot_pos_emb(grid_thw).reshape(hidden_states_full.shape[0], -1)

    if vit_prune_layer == 0:
        hidden_states = _select_packed_by_indices(hidden_states_full, grid_thw, keep_indices)
        pos_embeds = _select_packed_by_indices(pos_embeds_full, grid_thw, keep_indices)
        rotary_pos_emb = _select_packed_by_indices(rotary_pos_emb_full, grid_thw, keep_indices)
        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        seq_lengths = [len(idx) for idx in keep_indices]
        cu_seqlens = F.pad(
            torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0),
            (1, 0),
            value=0,
        )
    else:
        hidden_states = hidden_states_full + pos_embeds_full
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        emb_full = torch.cat((rotary_pos_emb_full, rotary_pos_emb_full), dim=-1)
        position_embeddings = (emb_full.cos(), emb_full.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(hidden_states.device)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens.int(),
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

        if layer_num + 1 == vit_prune_layer:
            hidden_states = _select_packed_by_indices(hidden_states, grid_thw, keep_indices)
            cos_pruned = _select_packed_by_indices(position_embeddings[0], grid_thw, keep_indices)
            sin_pruned = _select_packed_by_indices(position_embeddings[1], grid_thw, keep_indices)
            position_embeddings = (cos_pruned, sin_pruned)
            seq_lengths = [len(idx) for idx in keep_indices]
            cu_seqlens = F.pad(
                torch.tensor(seq_lengths, device=hidden_states.device, dtype=torch.int32).cumsum(0),
                (1, 0),
                value=0,
            )

    merged_hidden_states = self.merger(hidden_states)

    ds_indexes = getattr(self, "deepstack_visual_indexes", [])
    if deepstack_feature_lists and ds_indexes and vit_prune_layer > max(ds_indexes):
        spatial_merge_size = getattr(self.config, "spatial_merge_size", 2)
        block_size = spatial_merge_size ** 2
        merged_indices = [idx[::block_size] // block_size for idx in keep_indices]
        split_sizes = (grid_thw.prod(-1) // block_size).tolist()
        for i, ds_feat in enumerate(deepstack_feature_lists):
            ds_per_img = torch.split(ds_feat, split_sizes)
            deepstack_feature_lists[i] = torch.cat(
                [
                    ds_per_img[j][merged_indices[j].to(ds_per_img[j].device)]
                    for j in range(len(ds_per_img))
                ],
                dim=0,
            )

    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        deepstack_features=deepstack_feature_lists,
    )


def _model_get_image_features(
    self: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    keep_indices: Optional[List[torch.Tensor]] = None,
    **kwargs: Any,
) -> Any:
    pixel_values_typed = pixel_values.type(self.visual.dtype)
    return_dict = kwargs.pop("return_dict", True)
    vision_output = self.visual(
        pixel_values_typed,
        grid_thw=image_grid_thw,
        keep_indices=keep_indices,
        return_dict=True,
        **kwargs,
    )
    image_embeds = vision_output.pooler_output
    merge_size_sq = self.visual.spatial_merge_size ** 2
    if keep_indices is not None:
        split_sizes = [len(idx) // merge_size_sq for idx in keep_indices]
    else:
        split_sizes = (image_grid_thw.prod(-1) // merge_size_sq).tolist()
    vision_output.pooler_output = torch.split(image_embeds, split_sizes)
    return vision_output if return_dict else vision_output.to_tuple()


def _model_forward(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    keep_indices: Optional[List[torch.Tensor]] = None,
    soft_merged_image_embeds: Optional[List[torch.Tensor]] = None,
    soft_merged_deepstack: Optional[List[torch.Tensor]] = None,
    soft_merged_indices: Optional[List[torch.Tensor]] = None,
    **kwargs: Any,
) -> Any:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
    from transformers.utils import is_torchdynamo_compiling

    if pixel_values_videos is not None:
        return _ORIG_MODEL_FORWARD(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            **kwargs,
        )

    if keep_indices is None and soft_merged_image_embeds is None:
        return _ORIG_MODEL_FORWARD(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            **kwargs,
        )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    device = inputs_embeds.device

    if soft_merged_image_embeds is not None:
        merged_keep_indices = [idx.to(device=device, dtype=torch.long) for idx in soft_merged_indices or []]
        image_embeds = [seq_embeds.to(device) for seq_embeds in soft_merged_image_embeds]
        deepstack_image_embeds = soft_merged_deepstack
    else:
        spatial_merge_size = getattr(self.visual.config, "spatial_merge_size", 2)
        block_size = spatial_merge_size ** 2
        merged_keep_indices = [idx[::block_size] // block_size for idx in keep_indices]
        vit_prune_layer = int(_PATCH_CONFIG.get("vit_prune_layer", 0))
        vit_keep_indices = keep_indices if vit_prune_layer != -1 else None
        image_outputs = self.get_image_features(
            pixel_values,
            image_grid_thw,
            keep_indices=vit_keep_indices,
            return_dict=True,
        )
        image_embeds = list(image_outputs.pooler_output)
        deepstack_image_embeds = image_outputs.deepstack_features

        if vit_prune_layer == -1:
            image_embeds = [
                seq_embeds[merged_keep_indices[idx].to(seq_embeds.device)]
                for idx, seq_embeds in enumerate(image_embeds)
            ]
            if deepstack_image_embeds:
                split_sizes = (image_grid_thw.prod(-1) // block_size).tolist()
                deepstack_image_embeds = [
                    torch.cat(
                        [
                            torch.split(ds_layer_embeds, split_sizes)[i][merged_keep_indices[i].to(ds_layer_embeds.device)]
                            for i in range(len(split_sizes))
                        ],
                        dim=0,
                    )
                    for ds_layer_embeds in deepstack_image_embeds
                ]

    image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

    original_input_ids = input_ids.clone() if input_ids is not None else None
    if attention_mask is None:
        original_attention_mask = None
    elif isinstance(attention_mask, torch.Tensor):
        original_attention_mask = attention_mask.clone()
    else:
        original_attention_mask = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in attention_mask.items()
        }

    inputs_embeds, input_ids, keep_mask = _adjust_inputs_for_dedup(
        self,
        inputs_embeds,
        input_ids,
        image_embeds,
        image_grid_thw,
        merged_indices=merged_keep_indices,
        padding_side="left",
    )
    if cache_position is not None:
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=cache_position.device,
            dtype=cache_position.dtype,
        )

    image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
    visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
    deepstack_visual_embeds = deepstack_image_embeds

    mask_for_rope = original_attention_mask if original_attention_mask is not None else attention_mask
    attention_mask_tensor = mask_for_rope if not isinstance(mask_for_rope, dict) else mask_for_rope.get("full_attention")
    if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
        attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
        if attention_mask_tensor.dtype.is_floating_point:
            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

    prefill_compiled = is_torchdynamo_compiling() and (
        (input_ids is not None and input_ids.shape[1] != 1)
        or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
    )
    prefill_noncompiled = not is_torchdynamo_compiling() and (
        (cache_position is not None and cache_position[0] == 0)
        or past_key_values is None
        or past_key_values.get_seq_length() == 0
    )

    if (prefill_compiled or prefill_noncompiled) or self.rope_deltas is None:
        ids_for_rope = original_input_ids if original_input_ids is not None else input_ids
        position_ids, rope_deltas = self.get_rope_index(
            ids_for_rope,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask_tensor,
        )
        if keep_mask is not None:
            batch_size = position_ids.shape[1]
            adjusted_pos_ids = [position_ids[:, b, keep_mask[b]] for b in range(batch_size)]
            max_len = max(p.shape[1] for p in adjusted_pos_ids)
            padded_pos_ids = [F.pad(pos_ids, (max_len - pos_ids.shape[1], 0)) for pos_ids in adjusted_pos_ids]
            position_ids = torch.stack(padded_pos_ids, dim=1).to(inputs_embeds.device)
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                adjusted_attn = [attention_mask[b, keep_mask[b]] for b in range(batch_size)]
                padded_attn = [F.pad(m, (max_len - m.shape[0], 0), value=0) for m in adjusted_attn]
                attention_mask = torch.stack(padded_attn, dim=0).to(inputs_embeds.device)
        self.rope_deltas = rope_deltas
    elif position_ids is None:
        batch_size, seq_length, _ = inputs_embeds.shape
        delta = (
            (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            if cache_position is not None
            else 0
        )
        position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
        if cache_position is not None:
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


def _cg_forward(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Any,
) -> Any:
    keep_indices = None
    soft_merged_image_embeds = None
    soft_merged_deepstack = None
    soft_merged_indices = None
    if (
        pixel_values is not None
        and image_grid_thw is not None
        and pixel_values_videos is None
    ):
        spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)
        selector = str(_PATCH_CONFIG.get("selector", "pixelprune")).lower()
        if selector == "bishe_v2stage_anchor16_aware_v6":
            (
                soft_merged_image_embeds,
                soft_merged_deepstack,
                soft_merged_indices,
            ) = _compute_bishe_soft_merged_outputs(
                self.model,
                pixel_values,
                image_grid_thw,
            )
        elif selector in {"visionzip", "visionzip_topk", "visionzip_budget"}:
            merged_keep_indices = _compute_visionzip_merged_keep_indices(
                self.model,
                pixel_values,
                image_grid_thw,
            )
        else:
            pixel_values_norm = _normalize_pixel_values_for_selector(pixel_values)
            merged_keep_indices = _compute_merged_keep_indices(
                pixel_values_norm,
                image_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
        if soft_merged_indices is None:
            keep_indices = _merged_indices_to_patch_indices(
                merged_keep_indices,
                spatial_merge_size ** 2,
                pixel_values.device,
            )
        if _to_bool(_PATCH_CONFIG.get("verbose")):
            original = [int((t * h * w) // (spatial_merge_size ** 2)) for t, h, w in image_grid_thw.tolist()]
            reduced = [
                len(idx) for idx in (soft_merged_indices if soft_merged_indices is not None else merged_keep_indices)
            ]
            print(
                "[AIM Qwen3-VL token reduction] merged tokens "
                f"{original} -> {reduced} | selector={_PATCH_CONFIG['selector']} method={_PATCH_CONFIG['method']} "
                f"metric={_PATCH_CONFIG['metric']} threshold={_PATCH_CONFIG['threshold']}"
            )

    if soft_merged_image_embeds is not None:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            soft_merged_image_embeds=soft_merged_image_embeds,
            soft_merged_deepstack=soft_merged_deepstack,
            soft_merged_indices=soft_merged_indices,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    return _ORIG_CG_FORWARD(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        logits_to_keep=logits_to_keep,
        keep_indices=keep_indices,
        **kwargs,
    )


def apply_visionzip_to_qwen3_vl(model: Any, **kwargs: Any) -> None:
    """对 AIM 中的 Qwen3-VL 应用新的 hard-prune token reduction patch。

    支持参数：
    - method: 默认 ``pred_2d``
    - metric: ``max`` / ``exact`` / ``mae`` / ``rmse``
    - threshold: 相似度阈值
    - vit_prune_layer: 0 表示 ViT 输入层裁剪，-1 表示只在 merged token 层裁剪
    - anchored: 非 exact 模式下是否使用 anchored 版本
    - verbose: 打印简单保留率日志

    兼容旧 AIM 的 dominant/contextual/target_total_tokens 参数；当显式传入时，
    会切到 merged-token 级的 VisionZip top-k budget 选择。
    """
    from transformers.models.qwen3_vl import modeling_qwen3_vl as m

    dominant_arg = kwargs.pop("dominant", None)
    contextual_arg = kwargs.pop("contextual", None)
    target_total_tokens_arg = kwargs.pop("target_total_tokens", None)
    ignored = {
        k: kwargs.pop(k)
        for k in list(kwargs.keys())
        if k in {"protected_penalty", "qk_mix", "pos_mix"}
    }

    use_visionzip_budget = any(v is not None for v in (dominant_arg, contextual_arg, target_total_tokens_arg))
    selector_default = "visionzip_topk" if use_visionzip_budget else _PATCH_CONFIG["selector"]

    if target_total_tokens_arg is not None:
        target_keep = int(target_total_tokens_arg)
        dominant = int(dominant_arg) if dominant_arg is not None else target_keep
        contextual = int(contextual_arg) if contextual_arg is not None else max(0, target_keep - dominant)
    else:
        dominant = int(dominant_arg) if dominant_arg is not None else _PATCH_CONFIG["visionzip_dominant"]
        contextual = int(contextual_arg) if contextual_arg is not None else _PATCH_CONFIG["visionzip_contextual"]
        target_keep = max(1, dominant + contextual)

    _PATCH_CONFIG.update(
        {
            "selector": str(kwargs.pop("selector", selector_default)).lower(),
            "method": str(kwargs.pop("method", _PATCH_CONFIG["method"])).lower(),
            "metric": str(kwargs.pop("metric", _PATCH_CONFIG["metric"])).lower(),
            "threshold": float(kwargs.pop("threshold", _PATCH_CONFIG["threshold"])),
            "vit_prune_layer": int(kwargs.pop("vit_prune_layer", _PATCH_CONFIG["vit_prune_layer"])),
            "anchored": _to_bool(kwargs.pop("anchored", _PATCH_CONFIG["anchored"]), True),
            "verbose": _to_bool(kwargs.pop("verbose", _PATCH_CONFIG["verbose"]), False),
            "visionzip_dominant": dominant,
            "visionzip_contextual": contextual,
            "visionzip_target_keep": int(target_keep),
            "bishe_target_keep": int(kwargs.pop("bishe_target_keep", _PATCH_CONFIG["bishe_target_keep"])),
            "bishe_merge_steps": kwargs.pop("bishe_merge_steps", _PATCH_CONFIG["bishe_merge_steps"]),
            "bishe_anchor_points": kwargs.pop("bishe_anchor_points", _PATCH_CONFIG["bishe_anchor_points"]),
            "bishe_protected_penalty": float(
                kwargs.pop("bishe_protected_penalty", _PATCH_CONFIG["bishe_protected_penalty"])
            ),
            "bishe_qk_mix": float(kwargs.pop("bishe_qk_mix", _PATCH_CONFIG["bishe_qk_mix"])),
            "bishe_pos_mix": float(kwargs.pop("bishe_pos_mix", _PATCH_CONFIG["bishe_pos_mix"])),
            "bishe_metric_layer": int(kwargs.pop("bishe_metric_layer", _PATCH_CONFIG["bishe_metric_layer"])),
            "bishe_anchor_order_mode": str(
                kwargs.pop("bishe_anchor_order_mode", _PATCH_CONFIG["bishe_anchor_order_mode"])
            ).lower(),
            "bishe_anchor_order_group_size": int(
                kwargs.pop("bishe_anchor_order_group_size", _PATCH_CONFIG["bishe_anchor_order_group_size"])
            ),
            "bishe_anchor_order_first_stage_only": _to_bool(
                kwargs.pop(
                    "bishe_anchor_order_first_stage_only",
                    _PATCH_CONFIG["bishe_anchor_order_first_stage_only"],
                ),
                True,
            ),
        }
    )

    if kwargs:
        ignored.update(kwargs)

    global _ORIG_VISION_FORWARD, _ORIG_MODEL_GET_IMAGE_FEATURES, _ORIG_MODEL_FORWARD, _ORIG_CG_FORWARD

    if not getattr(m.Qwen3VLVisionModel, "__aim_qwen3_token_reduction_patched__", False):
        _ORIG_VISION_FORWARD = m.Qwen3VLVisionModel.forward
        m.Qwen3VLVisionModel.forward = _vision_forward

        _ORIG_MODEL_GET_IMAGE_FEATURES = m.Qwen3VLModel.get_image_features
        m.Qwen3VLModel.get_image_features = _model_get_image_features

        _ORIG_MODEL_FORWARD = m.Qwen3VLModel.forward
        m.Qwen3VLModel.forward = _model_forward

        _ORIG_CG_FORWARD = m.Qwen3VLForConditionalGeneration.forward
        m.Qwen3VLForConditionalGeneration.forward = _cg_forward

        m.Qwen3VLVisionModel.__aim_qwen3_token_reduction_patched__ = True
        m.Qwen3VLModel.__aim_qwen3_token_reduction_patched__ = True
        m.Qwen3VLForConditionalGeneration.__aim_qwen3_token_reduction_patched__ = True

    if ignored:
        print(f"[AIM Qwen3-VL token reduction] ignored legacy args: {sorted(ignored.keys())}")

    patch_mode = "soft-merge" if _PATCH_CONFIG.get("selector") == "bishe_v2stage_anchor16_aware_v6" else "hard-prune"
    print(
        f"[AIM Qwen3-VL token reduction] patched with {patch_mode} path: "
        f"selector={_PATCH_CONFIG['selector']}, method={_PATCH_CONFIG['method']}, metric={_PATCH_CONFIG['metric']}, "
        f"threshold={_PATCH_CONFIG['threshold']}, vit_prune_layer={_PATCH_CONFIG['vit_prune_layer']}, "
        f"anchored={_PATCH_CONFIG['anchored']}"
    )


def apply_bishemethod_v2stage_anchor16_aware_v6_to_qwen3_vl(model: Any, **kwargs: Any) -> None:
    explicit_bishe_target_keep = "bishe_target_keep" in kwargs
    cfg = {
        "selector": kwargs.pop("selector", "bishe_v2stage_anchor16_aware_v6"),
        "method": kwargs.pop("method", "pred_2d"),
        "metric": kwargs.pop("metric", "max"),
        "threshold": kwargs.pop("threshold", 0.0),
        "vit_prune_layer": kwargs.pop("vit_prune_layer", 0),
        "anchored": kwargs.pop("anchored", True),
        "bishe_target_keep": kwargs.pop("bishe_target_keep", 96),
        "bishe_merge_steps": kwargs.pop("bishe_merge_steps", [(192, 192), (144, 144), (96, 96), (48, 48)]),
        "bishe_anchor_points": kwargs.pop("bishe_anchor_points", ["grid4x4"]),
        "bishe_protected_penalty": kwargs.pop("bishe_protected_penalty", 0.09),
        "bishe_qk_mix": kwargs.pop("bishe_qk_mix", 0.20),
        "bishe_pos_mix": kwargs.pop("bishe_pos_mix", 0.05),
        "bishe_metric_layer": kwargs.pop("bishe_metric_layer", -2),
        "bishe_anchor_order_mode": kwargs.pop("bishe_anchor_order_mode", "block"),
        "bishe_anchor_order_group_size": kwargs.pop("bishe_anchor_order_group_size", 3),
        "bishe_anchor_order_first_stage_only": kwargs.pop("bishe_anchor_order_first_stage_only", True),
    }

    target_total_tokens = kwargs.pop("target_total_tokens", None)
    if target_total_tokens is not None and not explicit_bishe_target_keep:
        cfg["bishe_target_keep"] = int(target_total_tokens)

    cfg.update(kwargs)
    apply_visionzip_to_qwen3_vl(model, **cfg)
    print(
        "[AIM Qwen3-VL token reduction] bishemethod_v2stage_anchor16_aware_v6 patched with soft-merge path"
    )
