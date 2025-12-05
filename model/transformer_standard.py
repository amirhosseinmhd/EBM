"""
Standard Transformer for ListOps Classification.

A typical encoder-only transformer architecture compatible with the Long Range Arena
ListOps task. This implementation follows the classic "Attention is All You Need"
architecture with modern improvements.

Reference: https://github.com/google-research/long-range-arena
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional mask (batch, seq_len) or (batch, seq_len, seq_len)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and apply output projection
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    FFN(x) = ReLU(xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Layer normalization and residual connections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class StandardTransformer(nn.Module):
    """
    Standard Transformer for sequence classification (ListOps).

    Architecture:
        1. Token embedding + positional encoding
        2. Stack of transformer encoder layers
        3. Global average pooling or CLS token
        4. Classification head

    This is sized to be runnable on modern GPUs with reasonable performance
    on the ListOps task from Long Range Arena.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 10,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        pooling: str = 'mean'  # 'mean', 'max', or 'cls'
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes (10 for ListOps)
            d_model: Model dimension (embedding size)
            n_layers: Number of transformer encoder layers
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: ID of padding token
            pooling: Pooling strategy ('mean', 'max', or 'cls')
        """
        super().__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.pooling = pooling

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.token_embedding.padding_idx is not None:
            nn.init.constant_(self.token_embedding.weight[self.token_embedding.padding_idx], 0)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)
                            1 for tokens to attend, 0 for padding
        Returns:
            Logits (batch, num_classes)
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Token embedding + positional encoding
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Pooling
        if self.pooling == 'mean':
            # Global average pooling (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling == 'max':
            # Global max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).float()
            x = x.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = torch.max(x, dim=1)[0]
        elif self.pooling == 'cls':
            # Use first token (CLS token)
            pooled = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits

    def get_num_params(self, exclude_embeddings: bool = False) -> int:
        """Count model parameters."""
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if exclude_embeddings and 'embedding' in name:
                    continue
                total += param.numel()
        return total


def create_standard_transformer(
    vocab_size: int = 18,
    num_classes: int = 10,
    model_size: str = 'small'
) -> StandardTransformer:
    """
    Factory function to create transformers of different sizes.

    Args:
        vocab_size: Size of vocabulary (18 for ListOps)
        num_classes: Number of output classes (10 for ListOps)
        model_size: 'tiny', 'small', 'medium', or 'large'

    Returns:
        StandardTransformer model
    """
    configs = {
        'tiny': {
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 512,
            'dropout': 0.1
        },
        'small': {
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 4,
            'd_ff': 1024,
            'dropout': 0.1
        },
        'medium': {
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 2048,
            'dropout': 0.1
        },
        'large': {
            'd_model': 768,
            'n_layers': 8,
            'n_heads': 12,
            'd_ff': 3072,
            'dropout': 0.1
        }
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    model = StandardTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        **config
    )

    return model


if __name__ == "__main__":
    # Example: Create and test a small transformer
    print("Creating Standard Transformer for ListOps")
    print("=" * 70)

    # Create model
    model = create_standard_transformer(vocab_size=18, num_classes=10, model_size='small')

    # Print model info
    total_params = model.get_num_params(exclude_embeddings=False)
    non_emb_params = model.get_num_params(exclude_embeddings=True)

    print(f"Model size: small")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-embedding parameters: {non_emb_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 100

    # Random input
    input_ids = torch.randint(0, 18, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output logits (first sample): {logits[0]}")

    print("\n✓ Model created successfully!")
