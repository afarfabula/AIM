import math
from typing import Callable, Tuple, Optional
import torch

def bipartite_soft_matching_merge(
    metric: torch.Tensor,
    r: int,
    x: torch.Tensor,
    mode: str = "mean",
    protected_idx: Optional[torch.Tensor] = None,
    protected_penalty: Optional[float] = None,
) -> torch.Tensor:
    """
    Modified from the implementation of paper https://arxiv.org/abs/2210.09461
    Batch token merging with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    # Backward-compatible "protected" behavior:
    # If protected_idx is provided (token indices in [0, t)), we will bias the matching
    # so these tokens are less likely to be selected as merge sources (a-nodes).
    # We can only reduce by a maximum of 50% tokens.
    # Do NOT reduce r based on protected tokens; protection is handled by ranking bias below.
    t = metric.shape[1]
    r = min(r, t // 2)

    if r <= 0:
        return None

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :] # a: source, b: dst
        scores = a @ b.transpose(-1, -2) # row: source, col: dst

        node_max, node_idx = scores.max(dim=-1) # col index (dst nodes): for each node (row), find a best matched node (col)

        if protected_idx is not None and protected_idx.numel() > 0:
            # Map protected token indices to "a" (source) positions (even indices only).
            # Those source nodes get forced to the end of edge_idx ranking, making them unmerged.
            t = metric.shape[1]
            a_orig = torch.arange(0, t, 2, device=metric.device, dtype=protected_idx.dtype)  # [t1]
            prot = protected_idx.to(device=metric.device, dtype=a_orig.dtype).unique()
            # a_mask[i]=True if original token index of this a-node is protected
            a_mask = (a_orig.unsqueeze(0) == prot.unsqueeze(1)).any(dim=0)  # [t1]
            if a_mask.any():
                # node_max is often fp16/bf16. Do ranking on fp32 to avoid overflow.
                node_max_fp32 = node_max.float()
                if protected_penalty is None:
                    # Hard protection: never merge these source nodes.
                    node_max_fp32[:, a_mask] = -1e9
                else:
                    # Soft protection: lower their rank without forbidding matching.
                    node_max_fp32[:, a_mask] -= float(protected_penalty)
                edge_idx = node_max_fp32.argsort(dim=-1, descending=True)[..., None]
            else:
                edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        else:
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # row index (source nodes): rank best-matched pairs
        ################################# maintain relative order of unmerged tokens #################################
        # unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, the source nodes that do not related to merging operation (low similarity part)
        # unm_idx, _ = unm_idx.sort(dim=-2, descending=False)
        ################################# maintain relative order of unmerged tokens #################################
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, the source nodes that do not related to merging operation (low similarity part)
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, the source nodes to be merged (high similarity part)
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # the selected values in node_idx are the dst nodes during merging operation, the values can be duplicated

        # merge tokens
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # the source nodes that do not related to merging operation 
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # the source nodes to be merged
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # add source nodes to the indices in dst_idx, and merge them to the nodes in dst via 'mode'

        return torch.cat([unm, dst], dim=1) # [the source nodes that do not relate to merging, the dst nodes that already merged with matched source nodes], first item was sorted 
