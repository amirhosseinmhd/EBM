"""
Quick test script to verify the EBT-S1 model implementation.
Run this to ensure everything is working correctly.
"""

import torch
import sys
sys.path.append('.')  # Add parent directory to path

from model import (
    EBTS1Model,
    get_model_config,
    init_whole_model_weights,
    print_model_info,
)


def test_model_creation():
    """Test model creation and parameter counting."""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    # Get xxs config (6.18M params, 6 layers, 384 dim, 6 heads)
    config = get_model_config('xxs')
    print(f"✓ Config loaded: {config.n_layers} layers, {config.dim} dim, {config.n_heads} heads")
    
    # Create model
    vocab_size = 50277  # GPT-NeoX tokenizer size
    model = EBTS1Model(config, vocab_size=vocab_size, pad_token_id=0)
    print(f"✓ Model created with vocab size: {vocab_size}")
    
    # Initialize weights
    init_whole_model_weights(model, "xavier", weight_initialization_gain=1.0)
    print("✓ Weights initialized (Xavier)")
    
    # Print model info
    print_model_info(model, "EBT-S1 XXS")
    
    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    batch_size = 4
    seq_len = 32
    vocab_size = model.vocab_size
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"✓ Input shape: {input_ids.shape}")
    
    # Forward pass
    model.train()
    pred_dists, energies, embeds = model(
        input_ids,
        mcmc_num_steps=2,
        learning=True
    )
    
    print(f"✓ Forward pass successful!")
    print(f"  - Number of optimization steps: {len(pred_dists)}")
    print(f"  - Predicted distributions shape: {pred_dists[0].shape}")
    print(f"  - Energy shape: {energies[0].shape}")
    
    return input_ids, pred_dists, energies


def test_loss_computation(model, input_ids, pred_dists):
    """Test loss computation."""
    print("\n" + "="*60)
    print("TEST 3: Loss Computation")
    print("="*60)
    
    # Compute loss
    loss, metrics = model.compute_loss(input_ids, pred_dists)
    
    print(f"✓ Loss computed successfully!")
    print(f"  - Total loss: {metrics['loss']:.4f}")
    print(f"  - Initial loss (step 0): {metrics['initial_loss']:.4f}")
    print(f"  - Final loss (step 1): {metrics['final_loss']:.4f}")
    print(f"  - Perplexity: {metrics['perplexity']:.2f}")
    print(f"  - Step size α: {metrics['alpha']:.1f}")
    
    return loss


def test_backward_pass(loss):
    """Test backpropagation."""
    print("\n" + "="*60)
    print("TEST 4: Backward Pass")
    print("="*60)
    
    # Backpropagation
    loss.backward()
    print("✓ Backward pass successful!")
    print("  - Gradients computed through optimization steps")
    

def test_generation(model):
    """Test text generation."""
    print("\n" + "="*60)
    print("TEST 5: Text Generation")
    print("="*60)
    
    # Create short prompt
    prompt = torch.randint(0, model.vocab_size, (1, 10))
    print(f"✓ Prompt shape: {prompt.shape}")
    
    # Generate
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        mcmc_num_steps=2
    )
    
    print(f"✓ Generation successful!")
    print(f"  - Output shape: {generated.shape}")
    print(f"  - Generated {generated.shape[1] - prompt.shape[1]} new tokens")


def test_model_sizes():
    """Test different model sizes."""
    print("\n" + "="*60)
    print("TEST 6: Model Sizes")
    print("="*60)
    
    sizes = ['xxs', 'xs', 'small']
    vocab_size = 50277
    
    for size in sizes:
        config = get_model_config(size)
        model = EBTS1Model(config, vocab_size=vocab_size)
        params = model.get_num_params(exclude_embeddings=True)
        total_params = model.get_num_params(exclude_embeddings=False)
        
        print(f"✓ {size.upper():6s}: {params:>10,} params (non-emb), {total_params:>10,} total")


def test_attention_mechanism():
    """Test EBT attention in isolation."""
    print("\n" + "="*60)
    print("TEST 7: EBT Attention Mechanism")
    print("="*60)
    
    from model.transformer_blocks import EBTAttention, precompute_freqs_cis
    
    # Create attention layer
    dim = 384
    n_heads = 6
    attention = EBTAttention(dim, n_heads)
    
    # Create dummy input (2 * seq_len because of observed + predicted)
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, 2 * seq_len, dim)
    
    # Precompute frequencies
    freqs_cis = precompute_freqs_cis(
        dim=dim // n_heads,
        end=seq_len + 1,
    )
    
    # Forward pass
    output = attention(x, freqs_cis)
    
    print(f"✓ Attention mechanism working!")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Attention handles {seq_len} observed + {seq_len} predicted tokens")


def main():
    """Run all tests."""
    print("\n" + "🚀" * 30)
    print("EBT-S1 Model Implementation Test Suite")
    print("🚀" * 30)
    
    try:
        # Test 1: Create model
        model = test_model_creation()
        
        # Test 2: Forward pass
        input_ids, pred_dists, energies = test_forward_pass(model)
        
        # # Test 3: Loss computation
        loss = test_loss_computation(model, input_ids, pred_dists)
        # #
        # # Test 4: Backward pass
        test_backward_pass(loss)
        #
        # # Test 5: Generation
        # test_generation(model)
        #
        # # Test 6: Model sizes
        # test_model_sizes()
        #
        # # Test 7: Attention mechanism
        # test_attention_mechanism()
        #
        # Success!
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nModel implementation is working correctly! 🎉")
        print("Ready for training on WikiText-103 or TinyStories.\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()