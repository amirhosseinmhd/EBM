"""
Transformer building blocks for Energy-Based Transformer (EBT).

This module implements the core components following Section C.3 of the paper:
"Autoregressive Causal Energy-Based Transformers Efficient Implementation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================================
# RMSNorm - Root Mean Square Layer Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable than LayerNorm and used in LLaMA models.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE (Rotary Position Embeddings).
    
    Args:
        dim: Dimension of the frequency tensor (should be head_dim)
        end: Maximum sequence length
        theta: Scaling factor for frequency computation
    
    Returns:
        Precomputed complex frequency tensor of shape (end, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with input tensor.
    
    Args:
        freqs_cis: Frequency tensor to reshape
        x: Target tensor for broadcasting compatibility
    
    Returns:
        Reshaped frequency tensor
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequency tensor
    
    Returns:
        Tuple of (query, key) tensors with rotary embeddings applied
    """
    # Reshape to complex numbers: last dimension must be even
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape frequencies for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation and convert back to real
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for grouped-query attention (GQA).
    If n_rep == 1, this is standard multi-head attention.
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ============================================================================
# EBT Attention - Section C.3 Implementation
# ============================================================================


class EBTAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
            self,
            dim: int,
            n_heads: int,
            n_kv_heads: Optional[int] = None,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Model dimension.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of key/value heads. If None, defaults to n_heads.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = 1  # NOTE this is hardcoded since we are using DDP
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # init_whole_model_weights(self.wq, args.weight_initialization,
        #                          weight_initialization_gain=args.weight_initialization_gain)

        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # init_whole_model_weights(self.wk, args.weight_initialization,
        #                          weight_initialization_gain=args.weight_initialization_gain)

        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # init_whole_model_weights(self.wv, args.weight_initialization,
        #                          weight_initialization_gain=args.weight_initialization_gain)

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        # init_whole_model_weights(self.wo, args.weight_initialization,
        #                          weight_initialization_gain=args.weight_initialization_gain)
        # self.wq = ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        # NOTE the usage of S-1/S/S+1 is messed up and confusing here, I recommend checking the paper
        bsz, full_seqlen, _ = x.shape  # full_seqlen includes real embeds and pred embeds
        original_seqlen = full_seqlen // 2  # length of original sequence without next pred
        context_length = original_seqlen + 1  # actual context length of model
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)

        # _o is for original attention stuff
        xq_o = xq[:, :original_seqlen, :, :]  # B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]

        # _p is for predicted attention stuff
        xq_p = xq[:, original_seqlen:, :, :]  # B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]

        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:original_seqlen])

        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[
            1:context_length])  # use 1 since are the next preds and thus need to condition on a frame
        # I tested this compared to prepending row on S dimension and the tensors were the same

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        # # repeat k/v heads if n_kv_heads < n_heads # this does nothing since self.n_rep = 1
        # keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # original attn calc is more normal############################################

        # seqlen here is S-1 which = original_seqlen
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)  # B, N, S-1, S-1
        if mask is not None:
            # this mask needs to be seqlen, seqlen, was S, S
            o_mask = mask[:-1, :-1]  # set to S-1, S-1 like 0 -inf -inf; 0 0 -inf, etc
            scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # has B, S-1, D after

        # pred sequence attn calc is for energy-based transformer ########################################################################################

        # seqlen here is S-1 which = original_seqlen
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        values_p = xv_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(
            self.head_dim)  # B, N, S-1, S-1; this uses xq_p and keys_o since for every next pred calcs similarity to all prev words
        temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype,
                                  device=scores_p.device)  # B, N, S-1, 1; is used since context_length = original_length +1, superdiag needs this
        scores_p = torch.cat((scores_p, temp_append),
                             dim=-1)  # is B, N, S-1, S; represents for each next pred (S-1 row) attending to all previous words (S-1) and then itself +1

        insertion_superdiagonal = (xq_p * keys_p).sum(dim=3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)  # for if using non 32 precision
        # bs, n, s-1 ; this calcs attn score of next preds with themselves, is like grabbing diag of matmul

        superdiag_rows = torch.arange(scores_p.shape[2])  # [0, ..., S-2] (len 15)
        superdiag_cols = torch.arange(1, scores_p.shape[3])  # [1, ..., S-1] (len 15)
        # use [3] last line since is [2]+1 and scores_p is wider than is tall as has B, N, S-1, S

        # first remove superdiagonal values so doesnt use attention to future tokens--prevents leakage of probability mass
        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype,
                                          device=scores_p.device)  # for zeroing out superdiag since dont want to include in matmul, do this in differentiable way
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask

        # then set diagonal to next pred self attention scores in differentiable way
        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask

        if mask is not None:
            p_mask = mask[1:, :]  # S-1, S like 0 0 -inf -inf; 0 0 0, -inf, etc
            scores_p = scores_p + p_mask
        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)

        # Q: why do I need to extract superdiagonal why cant i just do matmul after? A: its bc would need same subsequence in value matrix but dont have it, have original subsequence and then seperately all next preds
        scores_p_superdiagonal = scores_p.diagonal(offset=1, dim1=2,
                                                   dim2=3).clone()  # is B, N, S-1; basically how much each token on this superdiag should attent to itself; clone since dont want mask to change this

        scores_p = scores_p * diagonal_removal_mask  # keeps scores_p as is except for superdiagonal which is next preds attention to selves, cant multiply these naively by values_p or values_o

        scores_p = scores_p[
            :, :, :, :-1]  # B, N, S-1, S-1 now; next preds/scores_p_superdiagonal was why needed extra col earlier (temp_append)
        output_p = torch.matmul(scores_p,
                                values_o)  # B, N, S-1, H; is how next preds attend to all original previous tokens;

        # next_pred_self_attention is to get self attention based on extracted superdiagonal and the values matrix (for predictions)
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(
            dim=-1)  # B, N, S-1, H this is for weighted sum of each next pred to its final embed rep.

        output_p = output_p + next_pred_self_attention  # B, N, S-1, H adding this is adding the aspect of each next pred embedding attending to itself
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # after this is B, S-1, D

        # return linear projection of concatted outputs ########################################################################################

        output = torch.cat((output_o, output_p), dim=1)  # B, 2(S-1), D
        return self.wo(output)


# ============================================================================
# FeedForward with SwiGLU
# ============================================================================

class FeedForward(nn.Module):
    """
    FeedForward network with SwiGLU activation.
    SwiGLU: Swish(xW) ⊙ (xV) where ⊙ is element-wise multiplication.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        
        # Calculate hidden dimension
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: Swish(xW1) ⊙ (xW3)"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer block with EBT attention.
    
    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        norm_eps: float = 1e-5,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.attention = EBTAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch, 2*seq_len, dim)
            freqs_cis: Precomputed RoPE frequencies
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, 2*seq_len, dim)
        """
        # Attention with residual
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        
        # FeedForward with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out