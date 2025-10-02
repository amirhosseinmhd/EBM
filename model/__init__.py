"""
Energy-Based Transformer (EBT) Model Package.

This package contains the implementation of EBT-S1 model for next-token prediction.
"""

from .utils import (
    EBTModelArgs,
    MODEL_SIZES,
    get_model_config,
    init_whole_model_weights,
    count_parameters,
    print_model_info,
    create_causal_mask,
    create_ebt_causal_mask,
)

from .transformer_blocks import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    EBTAttention,
    FeedForward,
    TransformerBlock,
)

from .ebt_s1 import EBTS1Model

__all__ = [
    # Configuration
    'EBTModelArgs',
    'MODEL_SIZES',
    'get_model_config',
    
    # Utilities
    'init_whole_model_weights',
    'count_parameters',
    'print_model_info',
    'create_causal_mask',
    'create_ebt_causal_mask',
    
    # Transformer components
    'RMSNorm',
    'precompute_freqs_cis',
    'apply_rotary_emb',
    'EBTAttention',
    'FeedForward',
    'TransformerBlock',
    
    # Main model
    'EBTS1Model',
]

__version__ = '0.1.0'
