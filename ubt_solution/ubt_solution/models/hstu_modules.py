import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

TIMESTAMPS_KEY = "timestamps"

# 相对注意力偏置基类
class RelativeAttentionBiasModule(nn.Module):
    @abc.abstractmethod
    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass

# 相对位置偏置
class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]

# 基于相对时间和位置的桶化偏置
class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._ts_w = nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = bucketization_fn

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias

# HSTU缓存状态类型
HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# HSTU注意力计算函数
def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
):
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    
    # 处理缓存
    if delta_x_offsets is not None:
        padded_q, padded_k = cached_q, cached_k
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        # 将不规则张量转换为填充的密集张量
        padded_q = torch.zeros(B, n, q.size(-1), device=q.device)
        padded_k = torch.zeros(B, n, k.size(-1), device=k.device)
        
        # 手动实现jagged_to_padded_dense
        offset_starts = x_offsets[:-1].tolist()
        offset_ends = x_offsets[1:].tolist()
        
        for i in range(B):
            start, end = offset_starts[i], offset_ends[i]
            length = min(end - start, n)
            if length > 0:
                padded_q[i, :length] = q[start:start+length]
                padded_k[i, :length] = k[start:start+length]

    # 重塑为多头注意力格式
    padded_q = padded_q.view(B, n, num_heads, attention_dim)
    padded_k = padded_k.view(B, n, num_heads, attention_dim)
    
    # 计算注意力分数
    qk_attn = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k)
    
    # 添加相对时间偏置
    if all_timestamps is not None and rel_attn_bias is not None:
        qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)
    
    # 使用标准 scaled-dot 注意力 + softmax
    scale = math.sqrt(attention_dim)
    qk_attn = qk_attn / scale
    qk_attn = qk_attn.masked_fill(invalid_attn_mask.unsqueeze(0).unsqueeze(0)==0, float('-inf'))
    qk_attn = F.softmax(qk_attn, dim=-1)
    
    # dropout
    qk_attn = F.dropout(qk_attn, p=rel_attn_bias._dropout if hasattr(rel_attn_bias, '_dropout') else 0.1, training=rel_attn_bias.training)
    
    # 将v转换为填充的密集张量
    padded_v = torch.zeros(B, n, v.size(-1), device=v.device)
    for i in range(B):
        start, end = offset_starts[i], offset_ends[i]
        length = min(end - start, n)
        if length > 0:
            padded_v[i, :length] = v[start:start+length]
    
    # 重塑为多头格式
    padded_v = padded_v.reshape(B, n, num_heads, linear_dim)
    
    # 计算注意力输出
    attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, padded_v)
    attn_output = attn_output.reshape(B, n, num_heads * linear_dim)
    
    # 将密集张量转换回不规则张量
    jagged_output = []
    for i in range(B):
        start, end = offset_starts[i], offset_ends[i]
        length = min(end - start, n)
        if length > 0:
            jagged_output.append(attn_output[i, :length])
    
    if jagged_output:
        attn_output_jagged = torch.cat(jagged_output, dim=0)
    else:
        attn_output_jagged = torch.zeros(0, num_heads * linear_dim, device=q.device)
    
    return attn_output_jagged, padded_q, padded_k

# 序列转导单元
class SequentialTransductionUnitJagged(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int, # This is dv in rails (dimension of V and FFN intermediate for one head)
        attention_dim: int,   # This is dqk in rails (dimension of Q, K per head)
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias", # Not directly used if LayerNorm is explicit
        linear_config: str = "uvqk",   # Currently only "uvqk" is supported by this refactor
        concat_ua: bool = False,       # Not used in this Pre-LN refactor
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim_per_head: int = linear_hidden_dim # dv
        self._attention_dim_per_head: int = attention_dim # dqk
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = relative_attention_bias_module
        
        if linear_config != "uvqk":
            raise ValueError(f"This refactored STU currently only supports linear_config 'uvqk'")

        # Pre-LN: LayerNorm before Multi-Head Attention
        self.norm1 = nn.LayerNorm(embedding_dim, eps=epsilon)

        # Combined projection for Q, K, V for multi-head attention
        # Total dimension for Q and K is num_heads * attention_dim_per_head
        # Total dimension for V is num_heads * linear_dim_per_head
        self.q_proj = nn.Linear(embedding_dim, num_heads * self._attention_dim_per_head)
        self.k_proj = nn.Linear(embedding_dim, num_heads * self._attention_dim_per_head)
        self.v_proj = nn.Linear(embedding_dim, num_heads * self._linear_dim_per_head)
            
        self._linear_activation_fn = getattr(F, linear_activation, None)
        if self._linear_activation_fn is None:
            raise ValueError(f"Unknown linear_activation {linear_activation}")
            
        # Output projection after attention
        self.attn_out_proj = nn.Linear(num_heads * self._linear_dim_per_head, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout_ratio) # Dropout for attention output

        # Pre-LN: LayerNorm before FFN
        self.norm2 = nn.LayerNorm(embedding_dim, eps=epsilon)
        
        # FFN layers
        ffn_hidden_dim = embedding_dim * 4 # Standard FFN expansion
        self.ffn1 = nn.Linear(embedding_dim, ffn_hidden_dim)
        self.ffn2 = nn.Linear(ffn_hidden_dim, embedding_dim)
        self.ffn_dropout = nn.Dropout(dropout_ratio) # Dropout for FFN output

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, HSTUCacheState]:
        
        # ----- Multi-Head Attention Sub-layer (Pre-LN) -----
        normed_x_attn = self.norm1(x)
        
        q = self.q_proj(normed_x_attn)
        k = self.k_proj(normed_x_attn)
        v = self.v_proj(normed_x_attn)
        
        # _hstu_attention_maybe_from_cache expects q,k,v with per-head dimensions for its internal logic.
        attn_output_jagged, new_cached_q, new_cached_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
            attention_dim=self._attention_dim_per_head, 
            linear_dim=self._linear_dim_per_head,   
            q=q, k=k, v=v,
            cached_q=cache[0] if cache else None,
            cached_k=cache[1] if cache else None,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        
        # attn_output_jagged is (total_elements, num_heads * linear_dim_per_head)
        attn_output_projected = self.attn_out_proj(attn_output_jagged)
        attn_output_dropped = self.attn_dropout(attn_output_projected)
        
        # First residual connection
        x = x + attn_output_dropped
        
        # ----- FFN Sub-layer (Pre-LN) -----
        normed_x_ffn = self.norm2(x)
        
        ffn_intermediate = self.ffn1(normed_x_ffn)
        ffn_activated = self._linear_activation_fn(ffn_intermediate)
        ffn_output = self.ffn2(ffn_activated) # Output of FFN before dropout
        ffn_output_dropped = self.ffn_dropout(ffn_output) # Dropout on the final output of FFN sub-block

        # Second residual connection
        x = x + ffn_output_dropped
        
        # Placeholder for full cache state management if needed for inference
        new_cache_state = (new_cached_q, new_cached_k, None, None) 

        return x, new_cache_state

# HSTU模型主体
class HSTUJagged(nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self._attention_layers: nn.ModuleList = nn.ModuleList(modules=modules)
        self._autocast_dtype: torch.dtype = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        # """
        # Args:
        #     x: (\sum_i N_i, D) x float
        #     x_offsets: (B + 1) x int32
        #     all_timestamps: (B, N) x int64
        #     invalid_attn_mask: (N, N) x float, each element in {0, 1}
        #     return_cache_states: bool. True if we should return cache states.
        # Returns:
        #     x' = f(x),                                                  (\sum_i N_i, D) x float
        # """
        cache_states: List[HSTUCacheState] = []

        # 使用自动混合精度
        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                )
                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        
        # 如果输入是密集张量，转换为不规则张量
        is_dense_input = len(x.size()) == 3
        if is_dense_input:
            B, N, D = x.size()
            jagged_x = []
            offset_starts = x_offsets[:-1].tolist()
            offset_ends = x_offsets[1:].tolist()
            
            for i in range(B):
                length = min(offset_ends[i] - offset_starts[i], N)
                if length > 0:
                    jagged_x.append(x[i, :length])
            
            if jagged_x:
                x = torch.cat(jagged_x, dim=0)
            else:
                x = torch.zeros(0, D, device=x.device)

        # 前向传播
        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        
        # 如果输入是密集张量，将结果转回密集张量
        if is_dense_input:
            B = x_offsets.size(0) - 1
            N = invalid_attn_mask.size(0)
            D = jagged_x.size(-1)
            
            y = torch.zeros(B, N, D, device=jagged_x.device)
            
            offset_starts = x_offsets[:-1].tolist()
            offset_ends = x_offsets[1:].tolist()
            
            current_idx = 0
            for i in range(B):
                length = min(offset_ends[i] - offset_starts[i], N)
                if length > 0:
                    y[i, :length] = jagged_x[current_idx:current_idx+length]
                    current_idx += length
            
            return y, cache_states
        else:
            return jagged_x, cache_states 