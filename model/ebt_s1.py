"""
Energy-Based Transformer S1 Model.

This implements the S1 variant of EBT with:
- 2 optimization steps
- Detachment between steps (for stability)
- Learnable step size (α)
- Loss calculated at each step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .utils import EBTModelArgs, init_whole_model_weights, create_ebt_causal_mask
from .transformer_blocks import (
    RMSNorm,
    precompute_freqs_cis,
    EBTAttention,
    FeedForward,
    TransformerBlock,
)


class EBTTransformer(nn.Module):
    """
    Core EBT Transformer (stack of transformer blocks).
    Based on ar_ebt_default.py from the official implementation.
    """
    
    def __init__(self, args: EBTModelArgs):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        self.vocab_size = None  # Will be set by parent model
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=args.dim,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                norm_eps=args.norm_eps,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )
            for _ in range(args.n_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            dim=args.dim // args.n_heads,
            end=args.max_seq_len,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through transformer blocks.
        
        Args:
            x: Input embeddings (batch, 2*seq_len, dim)
            start_pos: Starting position (for KV caching, not used in training)
        
        Returns:
            Output tensor (batch, 2*seq_len, dim)
        """
        # Move freqs_cis to same device as input
        freqs_cis = self.freqs_cis.to(x.device)
        
        # Create causal mask
        seqlen = (x.shape[1] + 2) // 2  # This gives you S
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1)

            # For KV caching (not needed in training but keeping for consistency)
            if start_pos > 0:
                mask = torch.hstack([
                    torch.zeros((seqlen, start_pos), device=x.device),
                    mask
                ]).type_as(x)

        # Pass through transformer blocks
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # Final normalization
        h = self.norm(h)
        
        return h


class EBTS1Model(nn.Module):
    """
    Energy-Based Transformer S1 Model for next-token prediction.
    
    Key S1 characteristics:
    - 2 optimization steps
    - Detachment between steps (stable training)
    - Learnable step size α ≈ 500
    - Loss calculated at each optimization step
    - No energy landscape regularization (replay buffer, Langevin dynamics, etc.)
    
    Training process:
        1. Embed observed tokens (context)
        2. Initialize predictions as random noise
        3. For each optimization step:
            a. Normalize predictions (softmax)
            b. Convert to embeddings
            c. Concatenate [observed_embeds, predicted_embeds]
            d. Forward through transformer → energy
            e. Compute gradient: ∇_pred energy
            f. Update: predictions -= α * ∇_pred energy
            g. **DETACH** predictions (S1 specific!)
            h. Calculate reconstruction loss
        4. Backpropagate accumulated loss
    """
    
    def __init__(
        self,
        args: EBTModelArgs,
        vocab_size: int,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, args.dim)
        init_whole_model_weights(
            self.embeddings,
            args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain
        )
        
        # Core transformer
        self.transformer = EBTTransformer(args)
        self.transformer.vocab_size = vocab_size
        
        # Output projection (embeddings → vocabulary logits)
        # Note: This is only used for predicted tokens
        # For observed tokens, we just pass embeddings through transformer
        self.output_proj = nn.Linear(args.dim, 1, bias=False)  # Energy head
        init_whole_model_weights(
            self.output_proj,
            args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain
        )
        
        # Learnable step size (α) for S1
        # Paper suggests α ≈ 500 for text, ≈ 30000 for video
        self.alpha = nn.Parameter(
            torch.tensor(args.dyt_alpha_init, dtype=torch.float32),
            requires_grad=True
        )
        
        # For softmax normalization
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mcmc_num_steps: int = 2,
        learning: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with S1 energy-based training.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            mcmc_num_steps: Number of optimization steps (default 2 for S1)
            learning: Whether in training mode (enables gradients)
        
        Returns:
            Tuple of:
            - predicted_distributions: List of predicted token distributions at each step
            - predicted_energies: List of energies at each step
            - predicted_embeddings_list: List of predicted embeddings at each step
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # ====================================================================
        # Step 1: Embed observed tokens (context)
        # ====================================================================
        # Remove last token to predict it
        context_ids = input_ids[:, :-1]  # (batch, seq_len-1)
        observed_embeds = self.embeddings(context_ids)  # (batch, seq_len-1, dim)
        
        original_seq_len = observed_embeds.shape[1]
        
        # ====================================================================
        # Step 2: Initialize predictions as random noise
        # ====================================================================
        # Initialize in logit space, then normalize
        predicted_logits = torch.randn(
            batch_size, original_seq_len, self.vocab_size,
            device=device,
            dtype=observed_embeds.dtype
        )
        
        # Storage for outputs
        predicted_distributions = []
        predicted_energies = []
        predicted_embeddings_list = []
        
        # ====================================================================
        # Step 3: Perform optimization steps
        # ====================================================================
        with torch.set_grad_enabled(learning):
            for step in range(mcmc_num_steps):
                # Enable gradients for predictions
                predicted_logits = predicted_logits.requires_grad_(True)

                # Normalize to probability distribution (stabilizes training)
                predicted_probs = self.softmax(predicted_logits)  # (batch, seq_len-1, vocab)

                # Convert to embeddings: prob_dist @ embedding_matrix
                # This is key: we predict in embedding space by weighted sum
                predicted_embeds = torch.matmul(
                    predicted_probs,
                    self.embeddings.weight
                )  # (batch, seq_len-1, dim)

                # ====================================================================
                # Step 4: Concatenate observed and predicted embeddings
                # ====================================================================
                # Shape: (batch, 2*(seq_len-1), dim)
                combined_embeds = torch.cat([observed_embeds, predicted_embeds], dim=1)
                
                # ====================================================================
                # Step 5: Forward through transformer → compute energy
                # ====================================================================
                transformer_output = self.transformer(combined_embeds, start_pos=0)
                
                # Extract predicted portion only
                predicted_output = transformer_output[:, original_seq_len:, :]
                
                # Compute energy (scalar per prediction)
                energy = self.output_proj(predicted_output).squeeze(-1)  # (batch, seq_len-1)

                # Store outputs (keep gradients for loss computation)
                predicted_distributions.append(predicted_logits)
                predicted_energies.append(energy)
                predicted_embeddings_list.append(predicted_embeds)

                # ====================================================================
                # Step 6: Compute gradient and update predictions
                # ====================================================================
                if step < mcmc_num_steps - 1:  # Don't update on last step
                    # Compute gradient of energy w.r.t. predictions
                    energy_sum = energy.sum()
                    pred_grad = torch.autograd.grad(
                        outputs=energy_sum,
                        inputs=predicted_logits,
                        retain_graph=False,  # Don't need to backprop through this
                        create_graph=False,  # S1: detachment means no second-order gradients
                    )[0]

                    # Gradient descent with detachment (S1 specific for stability)
                    # Detach here breaks gradient flow through iterations but not through energy computation
                    predicted_logits = (predicted_logits - self.alpha * pred_grad).detach()
        
        return predicted_distributions, predicted_energies, predicted_embeddings_list
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        predicted_distributions: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute reconstruction loss for S1 training.
        Loss is calculated at EACH optimization step (unlike S2 which only uses final step).
        
        Args:
            input_ids: Ground truth tokens (batch, seq_len)
            predicted_distributions: List of predicted logit distributions at each step
        
        Returns:
            Tuple of:
            - total_loss: Averaged loss across all steps
            - metrics: Dictionary of logging metrics
        """
        # Target tokens (what we're trying to predict)
        target_ids = input_ids[:, 1:]  # (batch, seq_len-1)
        target_flat = target_ids.reshape(-1)  # (batch * seq_len-1,)
        
        total_loss = 0.0
        num_steps = len(predicted_distributions)
        
        step_losses = []
        
        for step_idx, pred_logits in enumerate(predicted_distributions):
            # Reshape for cross-entropy
            pred_flat = pred_logits.reshape(-1, self.vocab_size)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                pred_flat,
                target_flat,
                ignore_index=self.pad_token_id,
                reduction='mean'
            )
            
            step_losses.append(loss.item())
            total_loss += loss
        
        # Average across steps (S1 specific)
        total_loss = total_loss / num_steps
        
        # Compute perplexity from final step
        final_loss = step_losses[-1]
        perplexity = math.exp(final_loss)
        
        metrics = {
            'loss': total_loss.item(),
            'final_loss': final_loss,
            'initial_loss': step_losses[0],
            'perplexity': perplexity,
            'alpha': self.alpha.item(),
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        mcmc_num_steps: int = 2,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using EBT.
        
        Args:
            input_ids: Prompt tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = no filtering)
            mcmc_num_steps: Number of optimization steps per token
        
        Returns:
            Generated tokens (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get predictions for next token
            predicted_dists, _, _ = self.forward(
                input_ids,
                mcmc_num_steps=mcmc_num_steps,
                learning=False
            )
            
            # Use final optimization step
            logits = predicted_dists[-1][:, -1, :]  # (batch, vocab)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, exclude_embeddings: bool = True) -> int:
        """Count model parameters."""
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if exclude_embeddings and 'embedding' in name:
                    continue
                total += param.numel()
        return total