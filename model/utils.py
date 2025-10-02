"""
Model utilities for EBT-S1.

This module contains:
- Model configuration dataclass
- Weight initialization functions
- Helper utilities
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class EBTModelArgs:
    """
    Configuration for Energy-Based Transformer model.
    
    Model sizes from Table D.1 in paper:
    - xxs: 6.18M params, 6 layers, 384 dim, 6 heads
    - xs: 12.4M params, 12 layers, 384 dim, 6 heads
    - small: 48.8M params, 12 layers, 768 dim, 12 heads
    - medium: 176M params, 24 layers, 1024 dim, 16 heads
    - large: 396M params, 24 layers, 1536 dim, 16 heads
    - xl: 708M params, 24 layers, 2048 dim, 32 heads
    """
    # Architecture
    dim: int = 384                      # Embedding dimension
    n_layers: int = 6                   # Number of transformer blocks
    n_heads: int = 6                    # Number of attention heads
    n_kv_heads: Optional[int] = None    # For grouped-query attention
    ffn_dim_multiplier: Optional[float] = None  # FFN hidden dim multiplier
    
    # Normalization
    norm_eps: float = 1e-5              # Epsilon for numerical stability
    
    # Sequence parameters
    max_batch_size: int = 32            # Maximum batch size
    max_seq_len: int = 257              # Max sequence length (256 + 1 for next token)
    
    # Initialization
    weight_initialization: str = "xavier"  # "xavier" or "he"
    weight_initialization_gain: float = 1.0
    
    # EBT-specific (not used in S1 model architecture, but kept for compatibility)
    dyt_alpha_init: float = 500.0       # Initial step size (α)
    ebt_norm: str = "rms"               # Normalization type
    ebt_act_func: str = "silu"          # Activation function


# Model size presets
MODEL_SIZES = {
    'xxs': {'dim': 384, 'n_layers': 6, 'n_heads': 6},
    'xs': {'dim': 384, 'n_layers': 12, 'n_heads': 6},
    'small': {'dim': 768, 'n_layers': 12, 'n_heads': 12},
    'medium': {'dim': 1024, 'n_layers': 24, 'n_heads': 16},
    'large': {'dim': 1536, 'n_layers': 24, 'n_heads': 16},
    'xl': {'dim': 2048, 'n_layers': 24, 'n_heads': 32},
}


def get_model_config(size: str = 'xxs', **kwargs) -> EBTModelArgs:
    """
    Get model configuration for a specific size.
    
    Args:
        size: Model size ('xxs', 'xs', 'small', 'medium', 'large', 'xl')
        **kwargs: Additional arguments to override defaults
    
    Returns:
        EBTModelArgs configuration
    
    Example:
        >>> config = get_model_config('xxs', max_seq_len=512)
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(MODEL_SIZES.keys())}")
    
    params = MODEL_SIZES[size].copy()
    params.update(kwargs)
    return EBTModelArgs(**params)


# ============================================================================
# Weight Initialization
# ============================================================================

def init_whole_model_weights(
    model: nn.Module,
    weight_initialization_method: str = "xavier",
    nonlinearity: str = 'linear',
    weight_initialization_gain: float = 1.0
):
    """
    Initialize all weights in a model using specified method.
    Directly from their implementation in model_utils.py.
    
    Args:
        model: PyTorch model or layer to initialize
        weight_initialization_method: "xavier" or "he"
        nonlinearity: For He init ('linear', 'relu', 'leaky_relu', 'selu', 'tanh')
        weight_initialization_gain: Scaling factor for initialized weights
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                valid_nonlinearities = ['linear', 'relu', 'leaky_relu', 'selu', 'tanh']
                if nonlinearity not in valid_nonlinearities:
                    raise ValueError(
                        f"Unsupported nonlinearity: {nonlinearity}. "
                        f"Must be one of {valid_nonlinearities}"
                    )
                
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                    
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            else:
                raise ValueError(
                    f"Unknown weight init method: {weight_initialization_method}"
                )
        
        elif isinstance(m, nn.Embedding):
            if weight_initialization_method == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
    
    model.apply(init_weights)


# ============================================================================
# Helper Functions
# ============================================================================

def count_parameters(model: nn.Module, exclude_embeddings: bool = True) -> int:
    """
    Count total parameters in model.
    
    Args:
        model: PyTorch model
        exclude_embeddings: If True, don't count embedding parameters
    
    Returns:
        Total parameter count
    """
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if exclude_embeddings and 'embedding' in name.lower():
                continue
            total += param.numel()
    return total


def print_model_info(model: nn.Module, name: str = "Model"):
    """
    Print model information including size and parameter count.
    
    Args:
        model: PyTorch model
        name: Name to display
    """
    total_params = count_parameters(model, exclude_embeddings=False)
    non_embedding_params = count_parameters(model, exclude_embeddings=True)
    
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-embedding parameters: {non_embedding_params:,}")
    print(f"Total size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"{'='*60}\n")


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        Causal mask of shape (seq_len, seq_len) with -inf above diagonal
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device),
        diagonal=1
    )
    return mask


def create_ebt_causal_mask(
    original_seq_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create special causal mask for EBT attention.
    The mask is (2*seq_len, 2*seq_len) to handle both observed and predicted tokens.
    
    Args:
        original_seq_len: Length of observed sequence
        device: Device to create mask on
    
    Returns:
        EBT causal mask
    """
    context_length = original_seq_len + 1
    full_len = 2 * original_seq_len
    
    mask = torch.zeros((full_len, full_len), device=device)
    
    # Observed tokens: standard causal mask
    mask[:original_seq_len, :original_seq_len] = torch.triu(
        torch.full((original_seq_len, original_seq_len), float('-inf'), device=device),
        diagonal=1
    )
    
    # Predicted tokens can't see each other (except self on superdiagonal)
    # This is handled in the attention mechanism itself
    mask[original_seq_len:, original_seq_len:] = float('-inf')
    
    # Predicted tokens can see observed tokens (causal)
    for i in range(original_seq_len):
        mask[original_seq_len + i, :i+1] = 0.0  # Can see up to position i
    
    return mask